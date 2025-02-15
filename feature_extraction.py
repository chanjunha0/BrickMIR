import numpy as np
import pandas as pd
import pywt

from collections import Counter
from scipy.signal import find_peaks, peak_widths
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


# Paths 
TEST_X_DIR = 'test_X/'
TRAIN_X_DIR_CLEAN = 'train_X_cleaned/'
TRAIN_Y_PATH = 'train_y_v0.1.0.csv'
SAMPLE_SUBMISSION_PATH = 'sample_submission_v0.1.0.csv.gz'

# Feature Extraction Function from a Single Time-Series File
def extract_features_from_dict(data_dict, filename):
    """
    Extracts statistical and advanced features from a dictionary containing 't' and 'v' keys.
    
    Parameters:
    - data_dict (dict): Dictionary with keys 't' (timestamps) and 'v' (values), both as NumPy arrays.
    - filename (str): Name of the file being processed (for debugging purposes).
    
    Returns:
    - dict or None: Extracted features or None if extraction fails.
    """
    features = {}
    
    values = data_dict['v']
    timestamps = data_dict['t']



    # Helper Functions

    def benford_correlation(x):
        if len(x) == 0 or np.all(x == 0):  # Edge case
            return np.nan
        try:
            x = np.array([int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))])
            benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
            data_distribution = np.array([(x == n).mean() for n in range(1, 10)])
            return np.corrcoef(benford_distribution, data_distribution)[0, 1]
        except Exception:
            return np.nan
        

    def _roll(a, shift):  # Helper function for rolling arrays
        return np.roll(a, shift, axis=0)



    def change_quantiles(x, ql, qh, isabs, f_agg):
        """
        Computes changes in a specified corridor defined by quantiles.
        """
        if len(x) < 2 or np.std(x) == 0:  # Edge case: insufficient data or constant values
            return np.nan
        if ql >= qh:  # Invalid quantile range
            return np.nan
        div = np.diff(x)
        if isabs:
            div = np.abs(div)
        try:
            bin_cat = pd.qcut(x, [ql, qh], labels=False, duplicates='drop')
            bin_cat_0 = bin_cat == 0
        except ValueError:  
            return np.nan
        
        # Identify valid indices
        ind = (bin_cat_0 & _roll(bin_cat_0, 1))[1:]  

        # No valid changes inside the corridor
        if np.sum(ind) == 0:  
            return np.nan
        ind_inside_corridor = np.where(ind == 1)
        aggregator = getattr(np, f_agg, None)

         # Invalid aggregation function
        if aggregator is None: 
            return np.nan
        return aggregator(div[ind_inside_corridor]) 


    def _into_subchunks(x, subchunk_length, every_n=1):
        """
        Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

        For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

            0  2  4
            1  3  5
            2  4  6

        with the settings subchunk_length = 3 and every_n = 2
        """
        len_x = len(x)

        assert subchunk_length > 1
        assert every_n > 0

        # how often can we shift a window of size subchunk_length over the input?
        num_shifts = (len_x - subchunk_length) // every_n + 1
        shift_starts = every_n * np.arange(num_shifts)
        indices = np.arange(subchunk_length)

        indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
        return np.asarray(x)[indexer]

    
    def permutation_entropy(x, tau, dimension):
        """
        Calculate the permutation entropy.

        Three steps are needed for this:

        1. chunk the data into sub-windows of length D starting every tau.
        Following the example from the reference, a vector

            x = [4, 7, 9, 10, 6, 11, 3

        with D = 3 and tau = 1 is turned into

            [[ 4,  7,  9],
                [ 7,  9, 10],
                [ 9, 10,  6],
                [10,  6, 11],
                [ 6, 11,  3]]

        2. replace each D-window by the permutation, that
        captures the ordinal ranking of the data.
        That gives

            [[0, 1, 2],
                [0, 1, 2],
                [1, 2, 0],
                [1, 0, 2],
                [1, 2, 0]]

        3. Now we just need to count the frequencies of every permutation
        and return their entropy (we use log_e and not log_2).

        Ref: https://www.aptech.com/blog/permutation-entropy/
            Bandt, Christoph and Bernd Pompe.
            “Permutation entropy: a natural complexity measure for time series.”
            Physical review letters 88 17 (2002): 174102 .
        """

        X = _into_subchunks(x, dimension, tau)
        if len(X) == 0:
            return np.nan

        permutations = np.argsort(np.argsort(X))
        # Count the number of occurences
        _, counts = np.unique(permutations, axis=0, return_counts=True)
        # turn them into frequencies
        probs = counts / len(permutations)
        # and return their entropy
        return -np.sum(probs * np.log(probs))
    

    def longest_one_run_ratio(vals, eps=1e-8):
        """
        Computes the longest consecutive run of values near 1
        as a ratio of the entire timeseries length.

        For example, if the array has length = 200, and the longest run
        of "near 1" is 50 in a row, it returns 50 / 200 = 0.25.
        
        Parameters
        ----------
        vals : array-like
            The time series or list of values.
        eps : float, optional
            Tolerance for deciding how close a value must be to 1. Defaults to 1e-8.

        Returns
        -------
        float
            The ratio (longest run of values near 1) / (total length).
        """
        n = len(vals)
        if n == 0:
            return 0.0

        # If the entire series is (almost) 1
        if np.allclose(vals, 1, atol=eps):
            return 1.0

        longest = 0
        current = 0

        for v in vals:
            if abs(v - 1) <= eps:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        # Convert to ratio
        return longest / n


    try:


        ##########################################
        # ----- Absolute Features -----
        ##########################################

        # Add Percentiles, capture high and lows, accounting for outliers
        features['ABS_p05_value'] = np.percentile(values, 5) if len(values) > 0 else 0.0 

        # 0>1 Changes
        # Check if values are within the range [0, 1]
        if not np.all((values >= 0) & (values <= 1)):
            # Calculate the number of changes between 0 and 1
            changes = np.abs(np.diff(values)) 

            # Only count transitions between 0 and 1
            changes = changes == 1  

            # Each value corresponds to 10 minutes, group into 24-hour periods (144 data points per day)
            changes_reshaped = changes[:len(changes) // 144 * 144].reshape(-1, 144)  
            
            # Sum changes for each day
            daily_change_counts = changes_reshaped.sum(axis=1)  
            
            # Calculate the 95th percentile of the count of number of changes
            percentile_95_changes = np.percentile(daily_change_counts, 95) if len(daily_change_counts) > 0 else np.nan
            
            features['ABS_95th_percentile_of_0_1_changes'] = percentile_95_changes
        else:
            features['ABS_95th_percentile_of_0_1_changes'] = 0.0





        ##########################################
        # ----- Ratio Features -----
        ##########################################
        
        # Find the mode (most frequent value)
        unique_counts = Counter(values)
        mode = unique_counts.most_common(1)[0][0] if unique_counts else None  

        # Mode / Len of values (fraction of time at most common value)
        features['RATIO_mode_DIV_total_len'] = unique_counts[mode] / len(values) if mode is not None else 0

        # Count the number of changes where the value is not equal to the mode
        changes = np.sum(np.array(values) != mode)

        # Ratio of p95 / p05
        features['RATIO_percentile_95_DIV_percentile_05'] = np.percentile(values, 95) / np.percentile(values, 5) if np.percentile(values, 5) != 0 else 0

        # number of unique values / mean
        if np.mean(values) == 0:
            features['RATIO_num_of_unique_DIV_mean'] = 0
        else:
            features['RATIO_num_of_unique_DIV_mean'] = len(np.unique(values)) / np.mean(values)

        # number of unique values / sd
        if np.std(values) == 0:
            features['RATIO_unique_DIV_sd'] = 0
        else:
            features['RATIO_unique_DIV_sd'] = len(np.unique(values)) / np.std(values)

        # sum of reoccurin values / number of unique values
        if len(np.unique(values)) != 0 and np.sum(values) != 0:
            features['RATIO_sum_reoccur_DIV_unique'] = np.sum(values) / len(np.unique(values))
        else:
            features['RATIO_sum_reoccur_DIV_unique'] = 0


        # lenght of the first min to the last min / length of values
        first_min = np.where(values == np.min(values))[0][0]
        last_min = np.where(values == np.min(values))[0][-1]
        features['RATIO_first_min_to_last_min_DIV_len'] = (last_min - first_min) / len(values)

        # ratio of longest one-run
        features['RATIO_longest_one_run_ratio'] = longest_one_run_ratio(values, eps=1e-8)  # FI 0.01

        # Use np.unique to calculate unique values and their frequencies
        unique_values, counts = np.unique(values, return_counts=True)

        # Get the most and least recurring values
        most_recurring_value = unique_values[np.argmax(counts)]  





        ##########################################
        # ----- Peak Features -----
        ##########################################

        def remove_false_peak(signal, p1, p2, maxDistance=2):
            """
            Remove 'false' peaks in p1 by comparing to p2, based on adjacency and ratio.
            """
            peak_diff = np.diff(p2)
            if len(peak_diff) == 0:
                return p1  

            ticks = []
            for i, d in enumerate(peak_diff):
                denom = signal[p2[i]]
                if denom == 0:
                    # If denominator is 0, skip or handle appropriately
                    continue
                
                ratio = signal[p2[i+1]] / denom

                # Example adjacency and ratio constraints:
                if d < maxDistance and -0.25 > ratio and ratio > -4:
                    ticks.append((p2[i], p2[i+1]))
            
            mask = np.ones(len(p1), dtype=bool)
            for i, j in ticks:
                # Example logic: remove p1 in [i, j+500]
                mask = mask & ((p1 < i) | (p1 > 500 + j))
            
            return p1[mask]

        def get_percentile_peaks_positive_only(signal, low_p=70, high_p=100, maxDistance=2):
            """
            1) Calculate two percentile-based thresholds (low_p, high_p) for the *positive* signal.
            2) Detect all peaks above pos_high_th (these become p2).
            3) Detect all peaks in [pos_low_th, pos_high_th] (these become p1).
            4) Use remove_false_peak to filter spurious peaks in p1 by referencing p2.
            5) Return final array of positive peak indices (p1) after removal.
            """
            # Compute low/high thresholds from the signal distribution
            pos_low_th  = np.percentile(signal, low_p) 
            pos_high_th = np.percentile(signal, high_p)

            # 1) Identify "strong" peaks > high threshold
            p2, _ = find_peaks(signal, height=pos_high_th)

            # 2) Identify "medium" peaks in [low_th, high_th]
            p1, _ = find_peaks(signal, height=(pos_low_th, pos_high_th))

            # 3) Filter out spurious peaks from p1
            p1_filtered = remove_false_peak(signal, p1, p2, maxDistance=maxDistance)

            return p1_filtered

        # Check if the entire signal is constant (could be all zeros or the same value)
        if len(values) == 0 or np.std(values) == 0 or len(np.unique(values)) == 2:
            # If no variation, no peaks. Return zeros.
            features["peak_height_median"] = 0
            features["peak_height_min"]    = 0

            features["peak_width_median"]  = 0
            features["peak_width_min"]     = 0
         

       

        else:    
            # 1) Get final (positive) peaks
            p_peaks = get_percentile_peaks_positive_only(values, low_p=70, high_p=100, maxDistance=2)
            num_p = len(p_peaks)

            # 3) Compute widths for the positive peaks
            widths_pos = peak_widths(values, p_peaks)[0]
            
            # 4) Compute peak heights (for positive peaks, you can just use the signal directly).
            #    If you're in the habit of taking absolute heights, you can still do so,
            #    but typically positive peaks means values[p_peaks] is already positive.
            heights_pos = values[p_peaks]  
            
            # 5) Calculate stats
            if len(heights_pos) == 0 or np.std(heights_pos) == 0:
                features["peak_height_median"] = 0
                features["peak_height_min"]    = 0
    
                features["peak_width_median"]  = 0
                features["peak_width_min"]     = 0
       

          
            else:
                features["peak_height_median"] = np.median(heights_pos)
                features["peak_height_min"]    = heights_pos.min()

                features["peak_width_median"]  = np.median(widths_pos) if len(widths_pos) else 0
                features["peak_width_min"]     = widths_pos.min()      if len(widths_pos) else 0
           


            # Ratio of peak height median to peak width median
            if features["peak_width_median"] == 0 or features["peak_height_median"] == 0:
                features["RATIO_peak_height_median_DIV_peak_width_median"] = 0
            else:
                features["RATIO_peak_height_median_DIV_peak_width_median"] = features["peak_height_median"] / features["peak_width_median"]


        del features["peak_height_median"]

      
        # -----------------------------------------------------------------------
        # 1) Estimate the sampling frequency from the first ~5 intervals
        # -----------------------------------------------------------------------
        n = len(values)
        if len(timestamps) < 2:
            features["missing_values_ratio"] = 0.0
            features["missing_runs_ratio"] = 0.0


        else:
            # If we have at least 2 timestamps but fewer than 6, use all of them.
            sample_t = timestamps[:6] if len(timestamps) >= 6 else timestamps
            deltas = np.diff(sample_t)
            
            # A robust estimate for "typical" sampling interval is often the median
            estimated_interval = np.median(deltas)  
            
            if estimated_interval <= 0:
                features["missing_values_ratio"] = 0.0
                features["missing_runs_ratio"] = 0.0

            else:
                # -----------------------------------------------------------------------
                # 2) Add a buffer (lateness_factor) to the estimated interval
                # -----------------------------------------------------------------------
                lateness_factor = 0.1  
                threshold_interval = estimated_interval * (1 + lateness_factor)
                
                total_missing = 0
                longest_run = 0
                current_run = 0
                
                # Track how many distinct missing runs we have
                missing_runs_count = 0
                in_missing_run = False

                # NEW: list to store each missing run’s length (in # of missing intervals)
                run_lengths = []
                
                # -----------------------------------------------------------------------
                # 3) Count missing intervals across the full time series
                # -----------------------------------------------------------------------
                for i in range(len(timestamps) - 1):
                    gap = timestamps[i+1] - timestamps[i]
                    
                    # If gap <= threshold_interval, we consider this "normal"
                    if gap <= threshold_interval:
                        missing_count = 0
                    else:
                        # If gap is bigger than the threshold, figure out how many intervals might be missing
                        missing_count = int(np.floor(gap / estimated_interval)) - 1
                        if missing_count < 0:
                            missing_count = 0
                    
                    # Add to total
                    total_missing += missing_count
                    
                    # Track runs
                    if missing_count > 0:
                        current_run += missing_count
                        if not in_missing_run:
                            # Start a new run
                            in_missing_run = True
                            missing_runs_count += 1
                    else:
                        if in_missing_run:
                            # We just ended a run; record its length
                            run_lengths.append(current_run)
                        current_run = 0
                        in_missing_run = False
                
                # Final check: if we ended while still in a run
                if in_missing_run:
                    run_lengths.append(current_run)
                
                # Update longest run if needed
                longest_run = max(longest_run, current_run)
                
                # -----------------------------------------------------------------------
                # 4) Convert missing counts to ratios
                # -----------------------------------------------------------------------
                features["missing_values_ratio"] = total_missing / n
                features["missing_runs_ratio"] = missing_runs_count / n
                


        # ########################################## 
        # # ----- Drop to 0 -----
        # ##########################################

        # Check if there are exactly 2 unique values and 0 is among them
        if len(np.unique(values)) == 2 and 0 in np.unique(values):
            # 1) Fraction of values that are zero
            num_zero = np.sum(values == 0)
            frac_zero = num_zero / len(values)
            features['fraction_of_values_that_are_zero'] = frac_zero

            # 3) Mean and SD of time intervals between consecutive 0s
            zero_timestamps = timestamps[values == 0]

            if len(zero_timestamps) > 1:
                intervals = np.diff(zero_timestamps)
                intervals_in_seconds = intervals / np.timedelta64(1, 's')

                mean_interval_0 = intervals_in_seconds.mean()
            else:
                # If there's 0 or 1 zero timestamps, no intervals exist
                mean_interval_0 = 0.0

            features['mean_interval_between_0s'] = mean_interval_0


        


        # # # ##########################################
        # # # # ----- Tsfresh -----
        # # # ##########################################

        abs_energy = np.dot(values, values)
        features['abs_energy_DIV_len'] = abs_energy / len(values) if len(values) > 0 else 0.0
        features['benford_corr'] = benford_correlation(values)
        features['change_quantiles_mean'] = change_quantiles(values, ql=0.1, qh=0.9, isabs=True, f_agg='mean')
        features['permutation_entropy_3'] = permutation_entropy(values, tau=3, dimension=3)


        ##########################################
        # ----- Trend and Seasonality -----
        ##########################################


        # Convert values to a pandas Series with datetime index
        if isinstance(timestamps, list) or isinstance(timestamps, pd.DatetimeIndex):
            ts = pd.Series(values, index=pd.to_datetime(timestamps))
        else:
            # If timestamps are not provided, assume a default frequency (e.g., daily)
            ts = pd.Series(values)
            ts.index = pd.date_range(start='2020-01-01', periods=len(values), freq='D')

        # Ensure the series has at least the required length
        required_length = 14
        if len(ts) < required_length:
            print(f"Padding time series for {filename} with {required_length - len(ts)} observations")
            # Pad with edge values to avoid NaNs
            ts = pd.Series(np.pad(ts.values, (0, required_length - len(ts)), mode='edge'),
                        index=pd.date_range(ts.index[0], periods=required_length, freq='D'))

        try:
            # Perform decomposition
            decomposition = seasonal_decompose(ts, model='additive', period=7, extrapolate_trend='freq')

            # Seasonal Features
            seasonal = decomposition.seasonal.dropna()
            if seasonal.std() > 0:
                features['seasonal_mean_DIV_seasonal_std'] = seasonal.mean() / seasonal.std()
            else:
                features['seasonal_mean_DIV_seasonal_std'] = 0.0


        except Exception as e:
            print(f"Skipping seasonal decomposition for {filename}: {e}")
            features['seasonal_mean'] = 0.0
            features['seasonal_std'] = 0.0



        ##########################################
        # ----- Rolling Window -----
        ##########################################

        # Define the list of rolling window sizes
        window_sizes = [2, 144] 

        for window in window_sizes:
            if len(values) >= window:
                rolling_series = pd.Series(values).rolling(window=window)

                # Maximum of the maxiumum of the windows 
                features[f'rolling_max_{window}'] = rolling_series.max().max()

                # Median of the medians of the windows
                features[f'rolling_median_{window}'] = rolling_series.median().median()

            else:
                # Default values if not enough data
                features.update({
                    f'rolling_max_{window}': 0.0,
                    f'rolling_median_{window}': 0.0,
                })



        del features['rolling_median_144']
        del features['rolling_median_2']


        ##########################################
        # ----- Fast Fourier Transform (FFT) -----
        ##########################################
        """
        Breaks down a signal into its constituent frequencies.
        """

        # Fast Fourier Transform (FFT) Features
        fft_coeffs = np.fft.fft(values)                       # FFT coefficients
        fft_phase = np.angle(fft_coeffs)                      # Phase (angle)
        features['fft_phase_mean'] = np.mean(fft_phase)       # Mean phase angle


        ##########################################
        # ----- Wavelet Transform Features -----
        ##########################################
        # Captures time-frequency patterns and detects localized features like spikes, bursts, or shifts.
        # Ideal for non-stationary signals where FFT might fail to capture transient changes.

        # ---- Haar Wavelet (Level 2) Decomposition ----
        coeffs_haar = pywt.wavedec(values, wavelet='haar', level=2)

        # Extract Features for Haar Wavelet Decomposition and add to features dictionary
        for i, coeff in enumerate(coeffs_haar):
            coeff = np.abs(coeff)  # Use magnitudes
            features[f'wavelet_mean_haar_{i}'] = np.mean(coeff)
            features[f'wavelet_max_haar_{i}'] = np.max(coeff)
            features[f'wavelet_min_haar_{i}'] = np.min(coeff)


        del features['wavelet_mean_haar_0']
        del features['wavelet_mean_haar_1']
        del features['wavelet_mean_haar_2']
        del features['wavelet_max_haar_0']
        del features['wavelet_max_haar_1']
        del features['wavelet_min_haar_0']


        ##########################################
        # ----- Lag Features -----
        ##########################################
        # Define the number of lags
        lags = [1, 36]

        for lag in lags:
            if len(values) > lag:
                try:
                    # Suppress division warnings
                    with np.errstate(divide='ignore', invalid='ignore'):

                        # Compute autocorrelation
                        autocorr = acf(values, nlags=lag, fft=True)
                        features[f'autocorr_lag_{lag}'] = autocorr[lag] if len(autocorr) > lag else 0.0

                except Exception as e:
                    # Catch other errors and assign default value
                    print(f"Warning: Error in ACF calculation for lag {lag}: {e}")
                    features[f'autocorr_lag_{lag}'] = 0.0

            else:
                features[f'autocorr_lag_{lag}'] = 0.0
        
        # ##########################################
        # # ----- Setpoint Features -----
        # ##########################################

        air_flow_setpoint_values = [
            0.0, 4.0, 20.0, 70.0, 80.0, 87.0, 100.0, 105.0, 110.0, 135.0, 140.0, 150.0,
            198.0, 200.0, 210.0, 220.0, 230.0, 250.0, 260.0, 275.0, 295.0, 300.0, 305.0,
            310.0, 328.0, 330.0, 340.0, 348.0, 355.0, 360.0, 365.0, 366.0, 390.0, 395.0,
            410.0, 415.0, 500.0, 505.0, 620.0, 975.0, 1000.0, 1152.0, 1265.0, 1300.0,
            1315.0, 1320.0, 1355.0, 1360.0, 1465.0, 2000.0, 2200.0
        ]
        features['is_Air_Flow_Setpoint'] = 1 if most_recurring_value in air_flow_setpoint_values else 0

        air_temperature_setpoint_values = [
            5.0, 10.0, 11.0, 12.0, 16.0, 18.0, 19.0, 19.5, 20.0, 21.0, 21.5, 22.0,
            22.3, 22.5, 23.0, 23.7, 23.9, 24.0, 24.2, 24.4, 24.5, 24.8, 25.0, 25.5,
            26.0, 26.5, 26.7, 27.0, 28.0, 30.0, 32.0
        ]
        # features['is_Air_Temperature_Setpoint'] = 1 if most_recurring_value in air_temperature_setpoint_values else 0
        #features['is_Cooling_Demand_Setpoint'] = 1 if most_recurring_value == 50.0 else 0
        features['is_Cooling_Supply_Air_Temp_deadband_Setpoint'] = 1 if most_recurring_value in [24.0, 24.5] else 0
        features['is_Cooling_Temperature_Setpoint'] = 1 if most_recurring_value in [22.0, 22.5, 23.0, 23.5, 24.0, 24.5] else 0
        #features['is_Damper_Position_Setpoint'] = 1 if most_recurring_value in [40.0, 100.0] and least_recurring_value == 0.0 else 0
        # features['is_Dewpoint_Setpoint'] = 1 if most_recurring_value in [7.77, 11.05, 100.0] else 0
        features['is_differential_pressure_Setpoint'] = 1 if most_recurring_value in [12.0, 21.0, 51.0, 76.0, 109.0, 136.0, 193.0, 300.0] else 0
        features['is_Discharge_Air_Temperature_Setpoint'] = 1 if most_recurring_value in [11.0] else 0

        flow_setpoint_values = [
            0.0, 4.0, 6.5, 20.0, 32.95, 70.0, 80.0, 87.0, 100.0, 105.0, 110.0, 135.0,
            140.0, 150.0, 198.0, 200.0, 210.0, 220.0, 230.0, 250.0, 260.0, 275.0,
            295.0, 300.0, 305.0, 310.0, 330.0, 340.0, 348.0, 355.0, 360.0, 365.0,
            366.0, 390.0, 395.0, 410.0, 415.0, 500.0, 505.0, 620.0, 1000.0, 1265.0,
            1300.0, 1315.0, 1320.0, 1355.0, 1360.0, 1465.0, 2000.0, 2200.0
        ]
        features['is_Flow_Setpoint'] = 1 if most_recurring_value in flow_setpoint_values else 0
        features['is_Heating_Demand_Setpoint'] = 1 if most_recurring_value in [30.0] else 0
        features['is_Heating_Supply_Air_Temperature_Deadband_Setpoint'] = 1 if most_recurring_value in [19.5, 20.0] else 0
        features['is_Heating_Temperature_Setpoint'] = 1 if most_recurring_value in [18.0, 19.5, 20.0, 21.0, 21.5, 22.0, 22.5] else 0
        features['is_Humidity_Setpoint'] = 1 if most_recurring_value in [40.0, 50.0, 70.0] else 0
        features['is_Low_Outside_Air_Temperature_Enable_Setpoint'] = 1 if most_recurring_value in [5.0] else 0
        features['is_Max_Air_Temperature_Setpoint'] = 1 if most_recurring_value in [23.0] else 0
        features['is_Min_Air_Temperature_Setpoint'] = 1 if most_recurring_value in [21.0] else 0
        features['is_Outside_Air_Lockout_Temperature_Setpoint'] = 1 if most_recurring_value in [12.0, 16.0, 18.0 ,21.0, 22.3, 24.0] else 0
        features['is_Outside_Air_Temperature_Setpoint'] = 1 if most_recurring_value in [5.0, 10.0, 12.0, 21.0, 22.3, 24.0] else 0
        # features['is_Reset_Setpoint'] = 1 if most_recurring_value in [1.0] else 0

        room_air_temperature_setpoint_values = [
            19.0, 21.0, 21.5, 22.0, 22.5, 23.0, 23.7, 23.9, 24.0, 24.2, 24.4, 24.5,
            24.8, 25.0, 25.5, 26.0, 26.5, 26.7, 28.0
        ]
        features['is_Room_Air_Temperature_Setpoint'] = 1 if most_recurring_value in room_air_temperature_setpoint_values else 0

        speed_setpoint_values = [
            0.0, 20.0, 25.0, 30.0, 40.0, 60.0, 70.0, 75.0, 76.0, 80.0, 82.0, 85.0,
            90.0, 94.0, 100.0, 105.0, 125.0, 150.0, 160.0, 182.0, 200.0, 210.0, 234.0,
            270.0, 300.0, 480.0, 525.0, 545.0
        ]
        features['is_Speed_Setpoint'] = 1 if most_recurring_value in speed_setpoint_values else 0

        static_pressure_setpoint_values = [
            100.0, 110.0, 145.0, 270.0, 275.0, 280.0, 285.0, 300.0, 315.0, 380.0,
            390.0, 395.0, 410.0, 450.0
        ]
        features['is_Static_Pressure_Setpoint'] = 1 if most_recurring_value in static_pressure_setpoint_values else 0

        supply_air_static_pressure_setpoint_values = [
            100.0, 270.0, 275.0, 280.0, 285.0, 300.0, 315.0
        ]
        #features['is_Supply_Air_Static_Pressure_Setpoint'] = 1 if most_recurring_value in supply_air_static_pressure_setpoint_values else 0

        supply_air_temperature_setpoint_values = [
            22.0, 24.0, 27.0, 30.0, 32.0
        ]
        #features['is_Supply_Air_Temperature_Setpoint'] = 1 if most_recurring_value in supply_air_temperature_setpoint_values else 0

        temperature_setpoint_values = [
            5.0, 6.5, 9.0, 10.0, 11.0, 12.0, 14.0, 16.0, 18.0, 19.0, 19.5, 20.0,
            21.0, 21.5, 22.0, 22.3, 22.5, 23.0, 23.5, 23.7, 23.9, 24.0, 24.2, 24.4,
            24.5, 24.8, 25.0, 25.5, 26.0, 26.5, 26.7, 27.0, 28.0, 30.0, 32.0, 38.0,
            68.0, 72.0
        ]
        features['is_Temperature_Setpoint'] = 1 if most_recurring_value in temperature_setpoint_values else 0

        time_setpoint_values = [
            0.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 5864.0, 9512.0, 14454.0, 15120.0,
            18720.0, 18822.0, 18830.0, 18848.0, 19248.0, 22830.0, 23688.0, 25910.0,
            26534.0, 28502.0, 28648.0, 36630.0, 36758.0
        ]

        features['is_Time_Setpoint'] = 1 if most_recurring_value in time_setpoint_values else 0


        water_temperature_setpoint_values = [
            6.5, 9.0, 12.0, 68.0, 72.0
        ]
        features['is_Water_Temperature_Setpoint'] = 1 if most_recurring_value in water_temperature_setpoint_values else 0
        features['is_Zone_Air_Humidity_Setpoint'] = 1 if most_recurring_value in [40.0, 50.0] else 0

    except Exception as e:
        print(f"Error extracting features from {filename}: {e}")
        return None
    
    return features