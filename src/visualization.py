import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_plot_style():
    """Set the style for all plots."""
    try:
        plt.style.use('seaborn')
    except OSError:
        logger.warning("Seaborn style not found. Using default style.")
        plt.style.use('default')
    
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    logger.info("Plot style set")

def plot_training_history(history, save_path=None):
    """
    Plot the training and validation loss over epochs.

    Args:
        history (dict): Training history dictionary.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    set_plot_style()
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()

def plot_fitness_history(fitness_history, save_path=None):
    """
    Plot the best fitness over generations for the genetic algorithm.

    Args:
        fitness_history (list): List of best fitness values per generation.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    set_plot_style()
    plt.figure()
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Fitness history plot saved to {save_path}")
    else:
        plt.show()

def plot_predictions(actual, predicted, title, save_path=None):
    """
    Create a scatter plot of actual vs predicted values and a histogram of residuals.

    Args:
        actual (np.array): Array of actual values.
        predicted (np.array): Array of predicted values.
        title (str): Title for the plots.
        save_path (str, optional): Path to save the plots. If None, the plots are displayed.
    """
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Scatter plot
    ax1.scatter(actual, predicted, alpha=0.5)
    ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{title}: Actual vs Predicted Values')
    ax1.grid(True)

    # Residuals histogram
    residuals = actual - predicted
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title}: Distribution of Residuals')
    ax2.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Predictions plot saved to {save_path}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for models that support feature importance.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    set_plot_style()
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
    else:
        logger.warning("The model does not have feature_importances_ attribute.")

def compare_models(actual1, predicted1, actual2, predicted2, model1_name, model2_name, save_path=None):
    """
    Compare two models by plotting their predictions and performance metrics.

    Args:
        actual1, actual2 (np.array): Arrays of actual values for each model.
        predicted1, predicted2 (np.array): Arrays of predicted values for each model.
        model1_name, model2_name (str): Names of the models being compared.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(actual1, predicted1, alpha=0.5, label=model1_name)
    ax.scatter(actual2, predicted2, alpha=0.5, label=model2_name)
    ax.plot([min(actual1.min(), actual2.min()), max(actual1.max(), actual2.max())],
            [min(actual1.min(), actual2.min()), max(actual1.max(), actual2.max())],
            'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Model Comparison: {model1_name} vs {model2_name}')
    ax.legend()
    ax.grid(True)

    # Calculate and display performance metrics
    rmse1 = np.sqrt(mean_squared_error(actual1, predicted1))
    rmse2 = np.sqrt(mean_squared_error(actual2, predicted2))
    r2_1 = r2_score(actual1, predicted1)
    r2_2 = r2_score(actual2, predicted2)

    plt.text(0.05, 0.95, f'{model1_name} - RMSE: {rmse1:.4f}, R2: {r2_1:.4f}\n'
                         f'{model2_name} - RMSE: {rmse2:.4f}, R2: {r2_2:.4f}',
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Model comparison plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Generate some example data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2 * x + 1 + np.random.normal(0, 1, 100)
    y_pred = 2.1 * x + 0.9 + np.random.normal(0, 0.5, 100)

    # Example of using the plotting functions
    plot_predictions(y_true, y_pred, "Example Model")
    
    # Example fitness history
    fitness_history = [0.5 + 0.1 * i for i in range(10)]
    plot_fitness_history(fitness_history)

    logger.info("Example plots generated")