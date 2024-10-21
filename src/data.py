# src/data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

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

def split_and_scale_data(data, input_columns, output_column, test_size=0.15, val_size=0.15):
    """
    Split the data sequentially into training, validation, and test sets, and scale the features.

    Args:
        data (pd.DataFrame): The preprocessed data.
        input_columns (list): List of input feature column names.
        output_column (str): Name of the target variable column.

    Returns:
        tuple: Generators for train, validation, and test sets, along with scalers.
    """
    # Calculate the number of samples for each set
    num_samples = len(data)
    num_test = int(num_samples * test_size)
    num_val = int(num_samples * val_size)
    num_train = num_samples - num_test - num_val

    # Split the data sequentially
    train_data = data.iloc[:num_train]
    val_data = data.iloc[num_train:num_train + num_val]
    test_data = data.iloc[num_train + num_val:]

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

    # Create TimeseriesGenerator instances
    window_size = 12
    batch_size = 32  # You can adjust this as needed

    train_generator = TimeseriesGenerator(train_X, train_y, length=window_size, batch_size=batch_size)
    val_generator = TimeseriesGenerator(val_X, val_y, length=window_size, batch_size=batch_size)
    test_generator = TimeseriesGenerator(test_X, test_y, length=window_size, batch_size=batch_size)

    logger.info("Data scaling and sequence generation completed")

    return train_generator, val_generator, test_generator, scaler_X, scaler_y

def load_data(data_dir='../data/raw', file_name='Train.csv'):
    """
    Load the data from the specified directory and file name.

    Args:
        data_dir (str): Directory containing the data file.
        file_name (str): Name of the data file.

    Returns:
        tuple: Generators for train, validation, and test sets, along with scalers.
    """
    file_path = os.path.join(data_dir, file_name)
    train_data, input_columns, output_column = load_and_preprocess_data(file_path)
    return split_and_scale_data(train_data, input_columns, output_column)

if __name__ == "__main__":
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'raw')
    train_gen, val_gen, test_gen, scaler_X, scaler_y = load_data(data_dir)
    
    logger.info(f"Number of training sequences: {len(train_gen)}")
    logger.info(f"Number of validation sequences: {len(val_gen)}")
    logger.info(f"Number of test sequences: {len(test_gen)}")
