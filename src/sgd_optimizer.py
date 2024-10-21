import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
from src.visualization import plot_training_history, plot_predictions, plot_time_series_predictions
from src.utils import save_model_summary, log_experiment_params, save_model_weights, save_evaluation_metrics
from src.utils import set_seeds

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_data_from_generator(generator):
    """
    Extract all data from a TimeseriesGenerator for evaluation.

    Args:
        generator (TimeseriesGenerator): The data generator.

    Returns:
        tuple: Arrays of inputs and targets.
    """
    X = []
    y = []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        X.append(x_batch)
        y.append(y_batch)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

def train_model_sgd(model, train_generator, val_generator, epochs=1000, patience=10, min_delta=1e-4, 
                    min_lr=1e-6, factor=0.1, results_dir=None):
    """
    Train the model using SGD optimizer with early stopping and learning rate reduction.

    Args:
        model (tf.keras.Model): The compiled model to train.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        epochs (int): Maximum number of epochs to train.
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
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Model training completed")

    if results_dir:
        # Save model weights
        model_weights_path = os.path.join(results_dir, 'best_model.weights.h5')
        save_model_weights(model, model_weights_path)

        # Save training history plot
        training_history_path = os.path.join(results_dir, 'sgd_training_history.png')
        plot_training_history(history.history, save_path=training_history_path)

        # Save model summary
        model_summary_path = os.path.join(results_dir, 'sgd_model_summary.txt')
        save_model_summary(model, model_summary_path)

        # Log experiment parameters
        params = {
            "optimizer": "SGD",
            "epochs": epochs,
            "patience": patience,
            "min_delta": min_delta,
            "min_lr": min_lr,
            "factor": factor
        }
        params_path = os.path.join(results_dir, 'sgd_experiment_params.txt')
        log_experiment_params(params, params_path)

    return history

def evaluate_model_sgd(model, test_generator, scaler_y, results_dir=None):
    """
    Evaluate the trained model on test data.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        test_generator: Test data generator.
        scaler_y (sklearn.preprocessing.StandardScaler): Scaler used for the target variable.
        results_dir (str): Directory to save results. If None, results won't be saved.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logger.info("Evaluating model on test data")

    # Extract test data
    test_X, test_y = extract_data_from_generator(test_generator)
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
        predictions_plot_path = os.path.join(results_dir, 'sgd_predictions.png')
        plot_predictions(test_actual, test_predictions, "SGD Model Predictions", save_path=predictions_plot_path)

        # Plot time series predictions
        time_series_plot_path = os.path.join(results_dir, 'sgd_time_series_predictions.png')
        plot_time_series_predictions(test_actual, test_predictions, "SGD Model Time Series Predictions",
                                     save_path=time_series_plot_path)

        # Log evaluation metrics
        metrics = {
            "test_mse": f"{mse:.4f}",
            "test_rmse": f"{rmse:.4f}",
            "test_r2": f"{r2:.4f}"
        }
        metrics_path = os.path.join(results_dir, 'sgd_evaluation_metrics.txt')
        save_evaluation_metrics(metrics, metrics_path)

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": test_predictions,
        "actual": test_actual
    }

def sgd_pipeline(model, train_generator, val_generator, test_generator, scaler_y, epochs=1000, 
                 patience=10, min_delta=1e-4, min_lr=1e-6, factor=0.1, results_dir=None):
    """
    Run the complete SGD training and evaluation pipeline.

    Args:
        model (tf.keras.Model): The compiled model to train.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        test_generator: Test data generator.
        scaler_y (sklearn.preprocessing.StandardScaler): Scaler used for the target variable.
        epochs (int): Maximum number of epochs to train.
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
    history = train_model_sgd(model, train_generator, val_generator, epochs, patience, 
                              min_delta, min_lr, factor, results_dir)

    # Evaluate the model
    eval_results = evaluate_model_sgd(model, test_generator, scaler_y, results_dir)

    logger.info("SGD pipeline completed")
    return history, eval_results

if __name__ == "__main__":
    # Example usage
    from src.models import create_and_compile_sgd_model
    from src.data import load_data
    import os

    # Set random seeds for reproducibility
    set_seeds(42)

    # Load data
    train_generator, val_generator, test_generator, scaler_X, scaler_y = load_data()

    # Get input shape from one batch
    x_batch, y_batch = train_generator[0]
    input_shape = x_batch.shape[1:]  # Exclude batch size dimension
    logger.info(f"Input shape: {input_shape}")

    # Create and compile model
    model = create_and_compile_sgd_model(input_shape=input_shape)

    # Set up results directory
    results_dir = "example_results_sgd"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")

    # Run SGD pipeline
    history, eval_results = sgd_pipeline(
        model, 
        train_generator,
        val_generator,
        test_generator,
        scaler_y,
        epochs=100,  # Adjust as needed
        results_dir=results_dir
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Test MSE: {eval_results['mse']:.4f}")
    print(f"Test RMSE: {eval_results['rmse']:.4f}")
    print(f"Test R2 Score: {eval_results['r2']:.4f}")

    logger.info("SGD training and evaluation complete. Check the 'example_results_sgd' directory for outputs.")
