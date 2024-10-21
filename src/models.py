import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
import logging

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
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        model.add(Flatten())

        for units in self.layer_sizes:
            model.add(Dense(units=units, activation='relu'))

        model.add(Dense(1))  # Output layer
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

    def train(self, train_generator, val_generator, results_dir=None, **kwargs):
        """
        Train the model using the optimizer's optimize method.

        Args:
            train_generator: Training data generator.
            val_generator: Validation data generator.
            results_dir (str): Directory to save results.
            **kwargs: Additional arguments for the optimizer's optimize method.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use set_optimizer() to assign an optimizer.")

        # Optimize the model
        self.optimizer.optimize(self.model, train_generator, val_generator, results_dir=results_dir, **kwargs)

        # Save model and optimizer hyperparameters
        if results_dir:
            # Save model hyperparameters
            model_hyperparams = self.get_hyperparameters()
            model_hyperparams_path = os.path.join(results_dir, 'model_hyperparameters.txt')
            log_experiment_params(model_hyperparams, model_hyperparams_path)
            logger.info(f"Model hyperparameters saved to: {model_hyperparams_path}")

            # Save optimizer hyperparameters
            optimizer_hyperparams = self.optimizer.get_hyperparameters()
            optimizer_hyperparams_path = os.path.join(results_dir, 'optimizer_hyperparameters.txt')
            log_experiment_params(optimizer_hyperparams, optimizer_hyperparams_path)
            logger.info(f"Optimizer hyperparameters saved to: {optimizer_hyperparams_path}")

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