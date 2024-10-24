import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Dropout
import logging

from src.optimizers import Optimizer, GeneticAlgorithmOptimizer  # Import the optimizer classes
from src.utils import log_experiment_params

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelWrapper:
    """
    A class to encapsulate the neural network model.
    """

    def __init__(self, input_shape, layer_sizes):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Shape of the input data (excluding batch size).
            layer_sizes (list): List of integers representing the number of units in each hidden layer.
        """
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        self.model = self._build_model()
        self.optimizer = None
        self.hyperparameters = {
            "input_shape": self.input_shape,
            "layer_sizes": self.layer_sizes
        }

    def _build_model(self):
        """
        Build the Sequential model based on the input shape and layer sizes.

        Returns:
            tf.keras.Model: The constructed Sequential model.
        """
        print("Building a new model...")
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        model.add(Flatten())

        for units in self.layer_sizes:
            model.add(Dense(units=units, activation='relu'))
            model.add(Dropout(0.2))

        # Output layer with sigmoid activation to ensure non-negative outputs
        model.add(Dense(1, activation='sigmoid'))
        return model

    def compile(self, optimizer, loss='mse'):
        """
        Compile the model with the given optimizer and loss function.

        Args:
            optimizer: An instance of a Keras optimizer.
            loss (str): Loss function to use.
        """
        self.model.compile(optimizer=optimizer, loss=loss)
        logger.info("Model compiled with optimizer: %s and loss: %s", optimizer, loss)

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the model.

        Args:
            optimizer (Optimizer): An instance of an optimizer class.
        """
        self.optimizer = optimizer
        logger.info("Optimizer set to: %s", type(optimizer).__name__)

    def train(self, train_generator, val_generator, scaler_y=None, results_dir=None, **kwargs):
        """
        Train the model using the optimizer's optimize method.

        Args:
            train_generator: Training data generator.
            val_generator: Validation data generator.
            scaler_y: The scaler used for the target variable.
            results_dir (str): Directory to save results.
            **kwargs: Additional arguments for the optimizer's optimize method.

        Returns:
            History object if optimizer has 'history', else None.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use set_optimizer() to assign an optimizer.")

        # Optimize the model
        if isinstance(self.optimizer, GeneticAlgorithmOptimizer):
            # Pass scaler_y to optimize method
            self.optimizer.optimize(self.model, train_generator, val_generator, scaler_y, results_dir=results_dir, **kwargs)
        else:
            # For optimizers that don't require scaler_y or results_dir
            self.optimizer.optimize(self.model, train_generator, val_generator, **kwargs)

        # Save model and optimizer hyperparameters
        if results_dir:
            # Save model hyperparameters
            model_hyperparams = self.get_hyperparameters()
            model_hyperparams_path = os.path.join(results_dir, 'model_hyperparameters.txt')
            log_experiment_params(model_hyperparams, model_hyperparams_path)
            logger.info(f"Model hyperparameters saved to: {model_hyperparams_path}")

            # Save optimizer hyperparameters
            optimizer_hyperparams = self.optimizer.get_hyperparameters()
            optimizer_name = type(self.optimizer).__name__
            optimizer_hyperparams_path = os.path.join(results_dir, f'{optimizer_name}_hyperparameters.txt')
            log_experiment_params(optimizer_hyperparams, optimizer_hyperparams_path)
            logger.info(f"Optimizer hyperparameters saved to: {optimizer_hyperparams_path}")

            # Save training history if available
            if hasattr(self.optimizer, 'history') and self.optimizer.history is not None:
                history_path = os.path.join(results_dir, f'{optimizer_name}_history.txt')
                with open(history_path, 'w') as f:
                    for key in self.optimizer.history.history.keys():
                        f.write(f"{key}: {self.optimizer.history.history[key]}\n")
                logger.info(f"Training history saved to: {history_path}")

        # Return the optimizer's history if it exists
        if hasattr(self.optimizer, 'history'):
            return self.optimizer.history
        else:
            return None

    def evaluate(self, test_generator):
        """
        Evaluate the model on the test data.

        Args:
            test_generator: Test data generator.

        Returns:
            float: Test loss.
        """
        loss = self.model.evaluate(test_generator, verbose=0)
        logger.info("Test loss: %.4f", loss)
        return loss

    def predict(self, X):
        """
        Generate predictions for the input samples.

        Args:
            X: Input data.

        Returns:
            np.array: Predictions.
        """
        return self.model.predict(X)

    def get_hyperparameters(self):
        """
        Get the model's hyperparameters.

        Returns:
            dict: Dictionary of model hyperparameters.
        """
        return self.hyperparameters

    def get_weights(self):
        """
        Get the model's weights.

        Returns:
            List of arrays: The model's weights.
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """
        Set the model's weights.

        Args:
            weights: List of arrays representing the model's weights.
        """
        self.model.set_weights(weights)