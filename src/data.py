import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the wind power data.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        tuple: Preprocessed data, input column names, and output column name.
    """
    logger.info(f"Loading data from {file_path}")
    train_data = pd.read_csv(file_path)
    logger.info(f"Train data shape: {train_data.shape}")
    
    # Drop unnecessary columns
    train_data.drop(columns=['Unnamed: 0', 'Time', 'Location'], inplace=True, errors='ignore')
    
    # Define input and output columns
    input_columns = ['Temp_2m', 'RelHum_2m', 'DP_2m', 'WS_10m', 'WS_100m', 'WD_10m', 'WD_100m', 'WG_10m']
    output_column = 'Power'
    
    logger.info(f"Input columns: {input_columns}")
    logger.info(f"Output column: {output_column}")
    
    return train_data, input_columns, output_column

def split_and_scale_data(data, input_columns, output_column, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split the data into training, validation, and test sets, and scale the features.

    Args:
        data (pd.DataFrame): The preprocessed data.
        input_columns (list): List of input feature column names.
        output_column (str): Name of the target variable column.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of training data to use for validation.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Scaled train, validation, and test data, along with scalers.
    """
    # Split the data into training, validation, and test sets
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=val_size_adjusted, random_state=random_state)
    
    logger.info(f"Training set shape: {train_data.shape}")
    logger.info(f"Validation set shape: {val_data.shape}")
    logger.info(f"Test set shape: {test_data.shape}")
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Scale input features
    train_X = scaler_X.fit_transform(train_data[input_columns])
    val_X = scaler_X.transform(val_data[input_columns])
    test_X = scaler_X.transform(test_data[input_columns])
    
    # Scale output variable
    train_y = scaler_y.fit_transform(train_data[[output_column]])
    val_y = scaler_y.transform(val_data[[output_column]])
    test_y = scaler_y.transform(test_data[[output_column]])
    
    logger.info("Data scaling completed")
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y), scaler_X, scaler_y

def load_data(data_dir='../data/raw', file_name='Train.csv'):
    """
    Load the data from the specified directory and file name.

    Args:
        data_dir (str): Directory containing the data file.
        file_name (str): Name of the data file.

    Returns:
        tuple: Preprocessed and split data, along with scalers.
    """
    file_path = os.path.join(data_dir, file_name)
    train_data, input_columns, output_column = load_and_preprocess_data(file_path)
    return split_and_scale_data(train_data, input_columns, output_column)

if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'raw')
    (train_X, train_y), (val_X, val_y), (test_X, test_y), scaler_X, scaler_y = load_data(data_dir)
    
    logger.info(f"Loaded data shapes:")
    logger.info(f"Train X: {train_X.shape}, Train y: {train_y.shape}")
    logger.info(f"Validation X: {val_X.shape}, Validation y: {val_y.shape}")
    logger.info(f"Test X: {test_X.shape}, Test y: {test_y.shape}")