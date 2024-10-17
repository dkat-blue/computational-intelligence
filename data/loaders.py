import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_raw_data(filepath):
    """
    Load the raw dataset from the given filepath.
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File at {filepath} not found.")
    return pd.read_csv(filepath)


def split_train_val_test(df, val_ratio=0.15, test_ratio=0.15, random_state=None):
    """
    Split the dataset into training, validation, and test sets in one operation.
    
    Args:
        df (pd.DataFrame): Dataset to split.
        val_ratio (float): Ratio of the data to use for validation.
        test_ratio (float): Ratio of the data to use for test set.
        random_state (int): Seed for random number generator.
    
    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Training, validation, and test datasets.
    """
    if not (0 < val_ratio < 1) or not (0 < test_ratio < 1):
        raise ValueError("val_ratio and test_ratio must be between 0 and 1")
    
    # Calculate the proportion of validation and test relative to the total dataset
    val_test_ratio = val_ratio + test_ratio
    
    # First, split into train and temp (which contains both validation and test)
    train_df, temp_df = train_test_split(df, test_size=val_test_ratio, random_state=random_state)
    
    # Now split temp into validation and test sets
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / val_test_ratio, random_state=random_state)
    
    return train_df, val_df, test_df