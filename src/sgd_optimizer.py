import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import logging
from src.visualization import plot_training_history, plot_predictions
from src.utils import save_model_summary, log_experiment_params

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model_sgd(model, train_data, val_data, epochs=1000, batch_size=32, patience=10, min_delta=1e-4, 
                    min_lr=1e-6, factor=0.1, results_dir=None):
    """
    Train the model using SGD optimizer with early stopping and learning rate reduction.

    Args:
        model (tf.keras.Model): The compiled model to train.
        train_data (tuple): Training data as (X, y).
        val_data (tuple): Validation data as (X, y).
        epochs (int): Maximum number of epochs to train.
        batch_size (int): Batch size for training.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        min_lr (float): Minimum learning rate.
        factor (float): Factor by which the learning rate will be reduced.
        results_dir (str): Directory to save results. If None, results won't be saved.

    Returns:
        history: Training history object.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience//2, min_lr=min_lr, verbose=1)
    ]

    logger.info("Starting model training with SGD")
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Model training completed")

    if results_dir:
        # Save training history plot
        plot_training_history(history.history, save_path=f"{results_dir}/sgd_training_history.png")
        
        # Save model summary
        save_model_summary(model, f"{results_dir}/sgd_model_summary.txt")
        
        # Log experiment parameters
        log_experiment_params({
            "optimizer": "SGD",
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "min_delta": min_delta,
            "min_lr": min_lr,
            "factor": factor
        }, f"{results_dir}/sgd_experiment_params.txt")

    return history

def evaluate_model_sgd(model, test_data, scaler_y, results_dir=None):
    """
    Evaluate the trained model on test data.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_data (tuple): Test data as (X, y).
        scaler_y (sklearn.preprocessing.StandardScaler): Scaler used for the target variable.
        results_dir (str): Directory to save results. If None, results won't be saved.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model on test data")
    
    test_X, test_y = test_data
    test_predictions_scaled = model.predict(test_X)
    
    # Inverse transform the predictions and actual values
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled)
    test_actual = scaler_y.inverse_transform(test_y)
    
    # Calculate metrics
    mse = mean_squared_error(test_actual, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_actual, test_predictions)
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test R2 Score: {r2:.4f}")
    
    if results_dir:
        # Plot predictions
        plot_predictions(test_actual, test_predictions, "SGD Model Predictions", 
                         save_path=f"{results_dir}/sgd_predictions.png")
        
        # Log evaluation metrics
        log_experiment_params({
            "test_mse": mse,
            "test_rmse": rmse,
            "test_r2": r2
        }, f"{results_dir}/sgd_evaluation_metrics.txt")
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": test_predictions,
        "actual": test_actual
    }

def sgd_pipeline(model, train_data, val_data, test_data, scaler_y, epochs=1000, batch_size=32, 
                 patience=10, min_delta=1e-4, min_lr=1e-6, factor=0.1, results_dir=None):
    """
    Run the complete SGD training and evaluation pipeline.

    Args:
        model (tf.keras.Model): The compiled model to train.
        train_data (tuple): Training data as (X, y).
        val_data (tuple): Validation data as (X, y).
        test_data (tuple): Test data as (X, y).
        scaler_y (sklearn.preprocessing.StandardScaler): Scaler used for the target variable.
        epochs (int): Maximum number of epochs to train.
        batch_size (int): Batch size for training.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        min_lr (float): Minimum learning rate.
        factor (float): Factor by which the learning rate will be reduced.
        results_dir (str): Directory to save results. If None, results won't be saved.

    Returns:
        tuple: Training history and evaluation results.
    """
    logger.info("Starting SGD pipeline")
    
    # Train the model
    history = train_model_sgd(model, train_data, val_data, epochs, batch_size, patience, 
                              min_delta, min_lr, factor, results_dir)
    
    # Evaluate the model
    eval_results = evaluate_model_sgd(model, test_data, scaler_y, results_dir)
    
    logger.info("SGD pipeline completed")
    return history, eval_results

if __name__ == "__main__":
    # Example usage
    from src.models import create_and_compile_sgd_model
    from src.data import load_data
    import os

    # Load data
    (train_X, train_y), (val_X, val_y), (test_X, test_y), scaler_X, scaler_y = load_data()

    # Create and compile model
    model = create_and_compile_sgd_model(input_shape=(train_X.shape[1],))

    # Set up results directory
    results_dir = "example_results"
    os.makedirs(results_dir, exist_ok=True)

    # Run SGD pipeline
    history, eval_results = sgd_pipeline(
        model, 
        (train_X, train_y), 
        (val_X, val_y), 
        (test_X, test_y), 
        scaler_y,
        epochs=100,  # Reduced for quicker example
        results_dir=results_dir
    )

    print("SGD training and evaluation complete. Check the 'example_results' directory for outputs.")