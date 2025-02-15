import multiprocessing
import os
import pandas as pd
import time

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from feature_extraction import extract_features_from_dict

# Paths 
TRAIN_X_DIR_CLEAN = 'train_X_cleaned/'
TEST_X_DIR_CLEAN = 'test_X_cleaned/'
TRAIN_Y_PATH = 'train_y_v0.1.0.csv'
SAMPLE_SUBMISSION_PATH = 'sample_submission_v0.1.0.csv.gz'



# Functions
def process_file(filename, directory):
    """
    Loads a pickle file and extracts features from it.
    
    Args:
        filename (str): Name of the file (e.g., '12345.pkl' or similar).
        directory (str): Directory path containing the pickle files.

    Returns:
        dict or None: Feature dictionary with an added 'filename' key, or None if failed.
    """
    filepath = os.path.join(directory, filename)
    
    try:
        # Load pickle file
        data_dict = pd.read_pickle(filepath)
    except Exception as e:
        # If loading fails, return None
        print(f"Error loading {filepath}: {e}")
        return None

    # Extract features
    feats = extract_features_from_dict(data_dict, filename)
    if feats is not None:
        feats['filename'] = filename
    return feats


# Decide how many cores 
num_cores = multiprocessing.cpu_count()  
print(f"Using {num_cores} CPU cores for parallel processing.\n")


# record start time
start_time = time.time()

##########################################
# Train files 
##########################################

# Load training labels
train_y = pd.read_csv(TRAIN_Y_PATH)

# The first column is filename, the rest are labels
label_cols = train_y.columns[1:]

# Build Training Feature Matrix
train_data = []
train_files = train_y['filename'].values

# Run the processing function in parallel for each train file
results = Parallel(n_jobs=num_cores)(
    delayed(process_file)(f, TRAIN_X_DIR_CLEAN) for f in tqdm(train_files, desc="Processing Train Files")
)

# Filter out None results (in case of errors)
train_data = [res for res in results if res is not None]

train_X = pd.DataFrame(train_data).merge(train_y, on='filename', how='left')

# Handle any missing merges
if train_X.isnull().any().any():
    print("Warning: Missing values detected after merging.")

    # Identify columns with missing values
    missing_cols = train_X.columns[train_X.isnull().any()].tolist()

    # Print columns and count of missing values
    print("\nColumns with missing values and their counts:")
    for col in missing_cols:
        missing_count = train_X[col].isnull().sum()
        print(f" - {col}: {missing_count} missing values")

    # Fill missing feature values with 0
    print("\nFilling missing feature values with 0...")
    feature_cols = [col for col in train_X.columns if col not in ['filename'] + list(label_cols)]
    train_X[feature_cols] = train_X[feature_cols].fillna(0)
else:
    print("No missing values detected after merging.")

# Separate features and labels
X_train = train_X.drop(['filename'] + list(label_cols), axis=1)
Y_train = train_X[label_cols]

# save to csv
X_train.to_csv('X_train_processed.csv', index=False)
Y_train.to_csv('Y_train_processed.csv', index=False)

print("\nFeature extraction for training data completed.\n")


##########################################
# Test Data
##########################################

# Load the Sample Submission to Get the Test Filenames
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH, compression='gzip')
test_files = sample_submission['filename'].values

# Run the processing function in parallel for each test file
results = Parallel(n_jobs=num_cores)(
    delayed(process_file)(f, TEST_X_DIR_CLEAN) for f in tqdm(test_files, desc="Processing Test Files")
)

# Filter out None results (in case of errors)
test_data = [res for res in results if res is not None]

# Convert to DataFrame
test_X = pd.DataFrame(test_data)

# Handle any missing features
if test_X.isnull().any().any():
    print("Warning: Missing values detected in test features.")

    # Identify columns with missing values
    missing_cols = test_X.columns[test_X.isnull().any()].tolist()

    # Print columns and count of missing values
    print("\nColumns with missing values and their counts:")
    for col in missing_cols:
        missing_count = test_X[col].isnull().sum()
        print(f" - {col}: {missing_count} missing values")

    # Fill missing feature values with 0
    print("\nFilling missing feature values with 0...")
    feature_cols = [col for col in test_X.columns if col != 'filename']
    test_X[feature_cols] = test_X[feature_cols].fillna(0)
else:
    print("No missing values detected in test features.")

X_test = test_X
output_path = 'X_test_processed.csv'

# Check if file exists and delete it
if os.path.exists(output_path):
    os.remove(output_path)

# check if row length is 315720, if its not, print error message but continue 
if X_test.shape[0] != 315720:
    print(f"Error: X_test has {X_test.shape[1]} rows, expected 315720 rows.")

# Save the new file
X_test.to_csv(output_path, index=False)

print("\nFeature extraction for test data completed.\n")

# record end time
end_time = time.time()

# calculate elapsed time in miutes
elapsed_time = (end_time - start_time) / 60
print(f"Elapsed time: {elapsed_time:.2f} minutes")


