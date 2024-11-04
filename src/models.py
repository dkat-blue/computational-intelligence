# model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# Configuration dictionary for easy adjustment of hyperparameters
config = {
    'look_back': 15,
    'batch_size': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'l2_reg': 0.01,
    'early_stopping': {
        'patience': 5,
        'restore_best_weights': True
    },
    'reduce_lr': {
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-5
    },
    'epochs': 50,
    'seed': 42  # Random seed for reproducibility
}

class ModelWrapper:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        self.compile_model()

    def build_model(self):
        # Set random seed for reproducibility
        tf.random.set_seed(config['seed'])
        
        # Create weight initializer with random seed
        initializer = GlorotUniform(seed=config['seed'])
        
        model = Sequential([
            LSTM(64, 
                 return_sequences=True, 
                 input_shape=self.input_shape,
                 kernel_initializer=initializer),
            Dropout(config['dropout_rate']),
            LSTM(32, 
                 kernel_regularizer=l2(config['l2_reg']),
                 kernel_initializer=initializer),
            Dropout(config['dropout_rate']),
            Dense(1, kernel_initializer=initializer)
        ])
        return model

    def compile_model(self):
        # Ensure the optimizer uses gradient-based optimization
        optimizer = Adam(learning_rate=config['learning_rate'])
        self.model.compile(optimizer=optimizer, 
                         loss='mean_squared_error', 
                         metrics=['mae'])

    def reset_weights(self):
        """Reset model weights to random values"""
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                # LSTM layers have multiple weight matrices
                weight_initializer = GlorotUniform(seed=config['seed'])
                bias_initializer = tf.zeros_initializer()
                
                # Reset kernel weights (input)
                shape = layer.weights[0].shape
                layer.weights[0].assign(weight_initializer(shape=shape))
                
                # Reset recurrent weights
                shape = layer.weights[1].shape
                layer.weights[1].assign(weight_initializer(shape=shape))
                
                # Reset bias
                if layer.use_bias:
                    shape = layer.weights[2].shape
                    layer.weights[2].assign(bias_initializer(shape=shape))
                    
            elif isinstance(layer, tf.keras.layers.Dense):
                # Dense layers have simple weight matrices
                weight_initializer = GlorotUniform(seed=config['seed'])
                bias_initializer = tf.zeros_initializer()
                
                # Reset kernel weights
                shape = layer.weights[0].shape
                layer.weights[0].assign(weight_initializer(shape=shape))
                
                # Reset bias
                if layer.use_bias:
                    shape = layer.weights[1].shape
                    layer.weights[1].assign(bias_initializer(shape=shape))

    def fit(self, train_data, val_data):
        # Reset model weights before training
        self.reset_weights()
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping']['patience'],
            restore_best_weights=config['early_stopping']['restore_best_weights']
        )
        
        # Learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['reduce_lr']['factor'],
            patience=config['reduce_lr']['patience'],
            min_lr=config['reduce_lr']['min_lr']
        )
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=config['epochs'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return history

    def evaluate_unscaled(self, data_generator, scaler):
        """Evaluate the model and return unscaled metrics"""
        predictions = self.model.predict(data_generator)
        
        # Get actual values from the generator
        actual_values = np.concatenate([y for x, y in data_generator], axis=0)
        
        # Reshape for inverse transform if needed
        predictions = predictions.reshape(-1, 1)
        actual_values = actual_values.reshape(-1, 1)
        
        # Inverse transform the predictions and actual values
        predictions_unscaled = scaler.inverse_transform(predictions)
        actual_values_unscaled = scaler.inverse_transform(actual_values)
        
        # Calculate metrics in original scale
        mse = mean_squared_error(actual_values_unscaled, predictions_unscaled)
        mae = np.mean(np.abs(actual_values_unscaled - predictions_unscaled))
        rmse = np.sqrt(mse)
        
        print("\nUnscaled Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Range of actual values: [{np.min(actual_values_unscaled):.4f}, {np.max(actual_values_unscaled):.4f}]")
        print(f"Range of predicted values: [{np.min(predictions_unscaled):.4f}, {np.max(predictions_unscaled):.4f}]")
        
        return mse, rmse, mae

    def predict(self, data):
        # Run predictions
        return self.model.predict(data)

    def check_input_shape(self, input_data):
        # Check if the input data shape matches the expected model input shape
        if input_data.shape[1:] != self.model.input_shape[1:]:
            raise ValueError(
                f"Incorrect input shape. Expected: {self.model.input_shape[1:]}, "
                f"but got: {input_data.shape[1:]}"
            )