import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(input_shape, layer_sizes=[256, 128, 64], dropout_rate=0.2, l2_reg=0.0001):
    """
    Create a neural network model with the specified architecture.

    Args:
        input_shape (tuple): Shape of the input data.
        layer_sizes (list): List of integers representing the number of units in each hidden layer.
        dropout_rate (float): Dropout rate to apply after each hidden layer.
        l2_reg (float): L2 regularization factor.

    Returns:
        tf.keras.Model: The created model.
    """
    model = Sequential([InputLayer(input_shape=input_shape)])

    for units in layer_sizes:
        model.add(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Output layer

    logger.info(f"Created model with architecture: {layer_sizes}")
    return model

def compile_model_sgd(model, learning_rate=0.001, momentum=0.9):
    """
    Compile the model with SGD optimizer.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Learning rate for the SGD optimizer.
        momentum (float): Momentum for the SGD optimizer.

    Returns:
        tf.keras.Model: The compiled model.
    """
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    logger.info(f"Compiled model with SGD optimizer (lr={learning_rate}, momentum={momentum})")
    return model

def create_and_compile_sgd_model(input_shape, layer_sizes=[256, 128, 64], dropout_rate=0.2, l2_reg=0.0001,
                                 learning_rate=0.001, momentum=0.9):
    """
    Create and compile a model for SGD training.

    Args:
        input_shape (tuple): Shape of the input data.
        layer_sizes (list): List of integers representing the number of units in each hidden layer.
        dropout_rate (float): Dropout rate to apply after each hidden layer.
        l2_reg (float): L2 regularization factor.
        learning_rate (float): Learning rate for the SGD optimizer.
        momentum (float): Momentum for the SGD optimizer.

    Returns:
        tf.keras.Model: The created and compiled model.
    """
    model = create_model(input_shape, layer_sizes, dropout_rate, l2_reg)
    return compile_model_sgd(model, learning_rate, momentum)

def create_ga_model(input_shape, layer_sizes=[256, 128, 64]):
    """
    Create a model for use with the Genetic Algorithm.

    Args:
        input_shape (tuple): Shape of the input data.
        layer_sizes (list): List of integers representing the number of units in each hidden layer.

    Returns:
        tf.keras.Model: The created model (uncompiled).
    """
    model = Sequential([InputLayer(input_shape=input_shape)])

    for units in layer_sizes:
        model.add(Dense(units, activation='relu'))

    model.add(Dense(1))  # Output layer

    logger.info(f"Created GA model with architecture: {layer_sizes}")
    return model

def get_total_params(model):
    """
    Get the total number of parameters in the model.

    Args:
        model (tf.keras.Model): The model to analyze.

    Returns:
        int: Total number of parameters in the model.
    """
    return sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

if __name__ == "__main__":
    # Example usage and testing
    input_shape = (8,)  # Assuming 8 input features
    layer_sizes = [128, 64, 32]

    # Create and compile SGD model
    sgd_model = create_and_compile_sgd_model(input_shape, layer_sizes)
    sgd_model.summary()
    logger.info(f"SGD Model total parameters: {get_total_params(sgd_model)}")

    # Create GA model
    ga_model = create_ga_model(input_shape, layer_sizes)
    ga_model.summary()
    logger.info(f"GA Model total parameters: {get_total_params(ga_model)}")

    # Verify that both models have the same number of parameters
    assert get_total_params(sgd_model) == get_total_params(ga_model), "Models have different number of parameters"
    logger.info("Both models have the same number of parameters")