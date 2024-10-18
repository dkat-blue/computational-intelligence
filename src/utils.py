import os
import random
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seeds set to {seed}")

def create_results_directory():
    """
    Create a timestamped directory for storing results.

    Returns:
        str: Path to the created results directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results directory created: {results_dir}")
    return results_dir

def save_model_summary(model, file_path):
    """
    Save a summary of the model architecture to a text file.

    Args:
        model (tf.keras.Model): The model to summarize.
        file_path (str): Path to save the summary file.
    """
    with open(file_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logger.info(f"Model summary saved to {file_path}")

def log_experiment_params(params, file_path):
    """
    Log experiment parameters to a file.

    Args:
        params (dict): Dictionary of parameters to log.
        file_path (str): Path to save the parameters file.
    """
    with open(file_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Experiment parameters logged to {file_path}")

if __name__ == "__main__":
    # Example usage
    set_seeds()
    results_dir = create_results_directory()
    print(f"Results will be saved in: {results_dir}")

    # Example of logging experiment parameters
    example_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "model_architecture": "Dense(64) -> Dense(32) -> Dense(1)"
    }
    log_experiment_params(example_params, os.path.join(results_dir, "example_params.txt"))