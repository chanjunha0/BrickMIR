import gc
import numpy as np
import os
import pandas as pd
import pickle

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict


# ===========================
# Utility Functions
# ===========================

def load_pickle(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load a dictionary of numpy arrays from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing arrays (e.g., {'t': ..., 'v': ...}).

    Raises:
        RuntimeError: If the file cannot be loaded or the pickle is malformed.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle file {file_path}: {e}")

def save_cleaned_data(cleaned_data: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned data to a pickle file.

    This function expects the DataFrame to have columns 't' and 'v', which
    will be converted into numpy arrays and pickled.

    Args:
        cleaned_data (pd.DataFrame): A dataframe with at least the columns 't' and 'v'.
        output_path (str): The path to the output pickle file.

    Raises:
        RuntimeError: If unable to save the data to the specified path.
    """
    try:
        cleaned_dict = {
            't': cleaned_data['t'].to_numpy(),
            'v': cleaned_data['v'].to_numpy()
        }
        with open(output_path, 'wb') as f:
            pickle.dump(cleaned_dict, f)
    except Exception as e:
        raise RuntimeError(f"Failed to save cleaned data to {output_path}: {e}")


# ===========================
# Outlier Detection Function
# ===========================

def detect_zscore_outliers(df: pd.DataFrame, column: str = 'v', threshold: float = 1.0) -> pd.DataFrame:
    """
    Detect and remove rows deemed outliers based on Z-score, but only remove
    rows where the detected outlier has a value of 0 in the specified column.

    Args:
        df (pd.DataFrame): The input dataframe containing at least the specified column.
        column (str, optional): The column to perform outlier detection on. Defaults to 'v'.
        threshold (float, optional): The Z-score threshold above which points are considered outliers. 
                                     Defaults to 1.0.

    Returns:
        pd.DataFrame: A DataFrame with outlier rows removed if they meet the conditions.
    """
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    
    if std == 0:
        return df

    # Calculate Z-scores
    z_scores = (df[column] - mean) / std

    # Identify outliers based on Z-score
    outliers = np.abs(z_scores) > threshold

    # Only drop rows where the value is 0 and they are detected as outliers
    df = df[~((outliers) & (df[column] == 0))]

    return df


# ===========================
# File Processing Function
# ===========================

def process_file(args):
    """
    Process a single pickle file to remove outliers (based on Z-score),
    then save the cleaned data to an output directory.

    Steps:
        1. Load data from pickle.
        2. Convert to DataFrame with columns ['t', 'v'].
        3. If the column 'v' has only two unique values, save data unchanged.
        4. Otherwise, perform Z-score outlier detection and remove rows where
           detected outliers have a value of 0.
        5. Save the cleaned data to a pickle file in the output directory.
        6. Perform garbage collection to free up memory.

    Args:
        args (tuple): A tuple containing:
            - file_path (str): Path to the input pickle file.
            - output_dir (str): Directory to save the cleaned pickle files.
            - zscore_threshold (float): Threshold used for Z-score outlier detection.

    Raises:
        RuntimeError: If an error occurs during file loading or saving.
    """
    file_path, output_dir, zscore_threshold = args
    try:
        # Load data
        data = load_pickle(file_path)

        # Convert to DataFrame
        df_original = pd.DataFrame(data, columns=['t', 'v'])

        # If only 2 unique values, we will not drop any rows
        if df_original['v'].nunique() == 2:
            output_file = os.path.join(output_dir, os.path.basename(file_path))
            save_cleaned_data(df_original, output_file)
            return
        
        # Z-score outlier detection
        df_cleaned = detect_zscore_outliers(df_original, column='v', threshold=zscore_threshold) 

        # Save Cleaned Data
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        save_cleaned_data(df_cleaned, output_file)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    finally:
        # Free memory
        del df_original, data
        gc.collect()


# ===========================
# Parallel Cleaning Function
# ===========================

def parallel_cleaning(input_dir: str, 
                      output_dir: str,
                      zscore_threshold: float,
                      num_workers: int = None) -> None:
    """
    Process all pickle files in a directory using multiple CPU cores to 
    detect and remove outliers, then save the cleaned data to an output directory.

    Args:
        input_dir (str): Directory containing the input pickle files.
        output_dir (str): Directory where cleaned pickle files will be saved.
        zscore_threshold (float): The threshold for Z-score outlier detection.
        num_workers (int, optional): Number of worker processes to use for parallel 
                                     processing. Defaults to all available CPU cores.

    Returns:
        None: This function does not return a value; it processes files and writes them to disk.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of pickle files
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.pkl')
    ]

    if not files:
        print(f"No pickle files found in {input_dir}.")
        return

    # Prepare arguments for each file
    args_list = [
        (file_path, output_dir, zscore_threshold)
        for file_path in files
    ]

    # Determine the number of worker processes
    if num_workers is None:
        num_workers = cpu_count()

    # Initialize the multiprocessing pool
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, args_list), total=len(args_list), desc="Processing Files"):
            pass


# ===========================
# Configuration and Constants
# ===========================

ZSCORE_THRESHOLD: float = 3

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    """
    Execute the cleaning pipeline for both training and test data.

    This script:
        1. Reads pickle files from the directories TRAIN_X_DIR and TEST_X_DIR.
        2. Applies outlier removal based on a Z-score threshold (ZSCORE_THRESHOLD).
        3. Saves the cleaned data in TRAIN_OUTPUT_DIRECTORY and TEST_OUTPUT_DIRECTORY.
        4. Utilizes multiprocessing to parallelize the workload across available CPU cores.
    """
    # Paths 
    TRAIN_Y_PATH = 'train_y_v0.1.0.csv'
    TRAIN_X_DIR = 'train_X/'
    TEST_X_DIR = 'test_X/'

    TRAIN_OUTPUT_DIRECTORY = 'train_X_cleaned/'
    TEST_OUTPUT_DIRECTORY = 'test_X_cleaned/'
    
    SAMPLE_SUBMISSION_PATH = 'sample_submission_v0.1.0.csv.gz'
    
    # Get the number of CPU threads available
    num_threads = cpu_count()
    print(f"Number of CPU threads available: {num_threads}")
    
    # Start the cleaning process for training data
    parallel_cleaning(
        input_dir=TRAIN_X_DIR,
        output_dir=TRAIN_OUTPUT_DIRECTORY,
        zscore_threshold=ZSCORE_THRESHOLD,
        num_workers=num_threads - 1
    )

    # Start the cleaning process for test data
    parallel_cleaning(
        input_dir=TEST_X_DIR,
        output_dir=TEST_OUTPUT_DIRECTORY,
        zscore_threshold=ZSCORE_THRESHOLD,
        num_workers=num_threads - 1
    )