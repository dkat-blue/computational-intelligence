import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Configuration dictionary to centralize parameters
config = {
    'look_back': 15,
    'batch_size': 32,
    'val_split': 0.2,
    'test_split': 0.1
}

class DataProcessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self._load_data()
        
    def _load_data(self):
        # Load data from CSV file
        try:
            data = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.file_path}")
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        
        # Convert Time column to datetime with day first format
        data['Time'] = pd.to_datetime(data['Time'], format='%d-%m-%Y %H:%M')
        
        return data

    def _scale_data(self, data):
        # Create a copy of the data
        scaled_df = data.copy()
        
        # Drop non-numeric columns before scaling
        columns_to_scale = scaled_df.select_dtypes(include=[np.number]).columns
        
        # Scale only numeric columns
        scaled_values = self.scaler.fit_transform(scaled_df[columns_to_scale])
        
        # Replace scaled values in the dataframe
        for i, col in enumerate(columns_to_scale):
            scaled_df[col] = scaled_values[:, i]
            
        return scaled_df

    def _inverse_scale(self, data):
        # Inverse scaling for post-processing predictions or actual values
        return self.scaler.inverse_transform(data)

    def _create_sequences(self, data):
        # Select only numeric columns for sequence creation
        numeric_data = data.select_dtypes(include=[np.number]).values
        
        X, y = [], []
        for i in range(len(numeric_data) - config['look_back']):
            X.append(numeric_data[i:i + config['look_back']])
            y.append(numeric_data[i + config['look_back'], -1])  # Assuming target is the last numeric column
        
        return np.array(X), np.array(y)

    def prepare_data(self):
        # Scale data and split it into training, validation, and test sets
        scaled_data = self._scale_data(self.data)
        
        # Sort data by time to ensure temporal ordering
        scaled_data = scaled_data.sort_values('Time')
        
        # Create sequences maintaining temporal order
        X, y = self._create_sequences(scaled_data)
        
        # Calculate split indices - NO SHUFFLING to maintain temporal order
        train_size = int(len(X) * (1 - config['val_split'] - config['test_split']))
        val_size = int(len(X) * config['val_split'])
        
        # Split data temporally
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
        
        # Print shapes and ranges for debugging
        print("\nData Ranges (scaled):")
        print(f"Training   - X: {X_train.shape}, y: min={y_train.min():.4f}, max={y_train.max():.4f}")
        print(f"Validation - X: {X_val.shape}, y: min={y_val.min():.4f}, max={y_val.max():.4f}")
        print(f"Test      - X: {X_test.shape}, y: min={y_test.min():.4f}, max={y_test.max():.4f}")
        
        # Verify no data leakage between sets
        train_dates = scaled_data.iloc[:train_size]['Time']
        val_dates = scaled_data.iloc[train_size:train_size + val_size]['Time']
        print("\nTemporal Split Check:")
        print(f"Training period: {train_dates.min()} to {train_dates.max()}")
        print(f"Validation period: {val_dates.min()} to {val_dates.max()}")
        
        # Package data as tf.data.Dataset
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(config['batch_size'])
        val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(config['batch_size'])
        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(config['batch_size'])
        
        return train_data, val_data, test_data

    def check_scaling(self, data, scaled_data):
        # Validate scaling by checking min and max values in both directions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        assert np.min(scaled_data[numeric_cols].values) >= 0 and np.max(scaled_data[numeric_cols].values) <= 1, "Scaled data is out of range [0, 1]"
        
        # Check inverse scaling only for numeric columns
        inversed_data = self._inverse_scale(scaled_data[numeric_cols].values)
        np.testing.assert_almost_equal(data[numeric_cols].values, inversed_data, decimal=5, err_msg="Inverse scaling is inaccurate")