import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import save_model_weights, save_evaluation_metrics

# Flattened weight arrays helper functions
def get_weights_flat(model):
    # Get model weights and flatten them into a 1D array
    weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in weights])
    return flat_weights

def set_weights_from_flat(model, flat_weights):
    # Set model weights from a flat 1D array
    weights = []
    idx = 0
    for layer in model.layers:
        for w in layer.get_weights():
            shape = w.shape
            size = np.prod(shape)
            new_w = flat_weights[idx:idx+size].reshape(shape)
            weights.append(new_w)
            idx += size
    model.set_weights(weights)

def extract_data_from_generator(generator):
    X = []
    y = []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        X.append(x_batch)
        y.append(y_batch)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

# Define the fitness function
def compute_fitness(E_Wi):
    # FI(W_i) = 1 / (E_Wi + epsilon)
    epsilon = 1e-8  # Small constant to prevent division by zero
    FI_Wi = 1 / (E_Wi + epsilon)
    return FI_Wi

def tournament_selection(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = np.argmax(tournament_fitnesses)
        winner = population[tournament_indices[winner_idx]]
        selected.append(winner)
    return selected

def genetic_algorithm(model, train_generator, val_generator, test_generator, scaler_y, population_size=50, generations=100, 
                      tournament_size=3, mutation_rate=0.005, mutation_scale=0.1, patience=10, results_dir=None):
    """
    Genetic Algorithm optimization with early stopping.

    Args:
        model: The neural network model to optimize.
        train_generator: Training data generator.
        val_generator: Validation data generator.
        test_generator: Test data generator.
        scaler_y: Scaler for the target variable.
        population_size (int): Size of the population.
        generations (int): Maximum number of generations.
        tournament_size (int): Size of the tournament for selection.
        mutation_rate (float): Probability of mutation for each gene.
        mutation_scale (float): Scale of mutation noise.
        patience (int): Number of generations to wait for improvement before stopping.
        results_dir (str): Directory to save results.

    Returns:
        Tuple containing the best model, fitness history, and evaluation results.
    """
    # Extract data from generators for evaluation
    train_X, train_y = extract_data_from_generator(train_generator)
    val_X, val_y = extract_data_from_generator(val_generator)
    test_X, test_y = extract_data_from_generator(test_generator)

    # Initialize population
    initial_weights_flat = get_weights_flat(model)
    num_weights = len(initial_weights_flat)
    population = [np.random.randn(num_weights) * 0.1 for _ in range(population_size)]

    best_fitness_overall = -np.inf
    best_individual_overall = None
    fitness_history = []
    generations_without_improvement = 0  # Counter for early stopping

    for generation in range(generations):
        print(f"Generation {generation}")
        # Evaluate fitness for each individual
        fitnesses = []
        for individual in population:
            set_weights_from_flat(model, individual)
            # Compute loss on validation data
            y_pred = model.predict(val_X, verbose=0)
            E_Wi = mean_squared_error(val_y, y_pred)
            FI_Wi = compute_fitness(E_Wi)
            fitnesses.append(FI_Wi)
        fitnesses = np.array(fitnesses)

        # Keep track of the best individual
        best_fitness_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_fitness_idx]
        best_individual = population[best_fitness_idx]
        print(f"Best fitness: {best_fitness:.6f}")
        fitness_history.append(best_fitness)

        # Early stopping check
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_individual_overall = best_individual.copy()
            generations_without_improvement = 0  # Reset counter
        else:
            generations_without_improvement += 1
            if generations_without_improvement >= patience:
                print(f"No improvement for {patience} generations. Early stopping at generation {generation}.")
                break

        selected_population = tournament_selection(population, fitnesses, tournament_size)

        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            # Select parents
            parent1, parent2 = random.sample(selected_population, 2)
            # Uniform crossover
            mask = np.random.rand(num_weights) < 0.5
            offspring1 = np.where(mask, parent1, parent2)
            offspring2 = np.where(mask, parent2, parent1)
            # Mutation
            mutation_mask1 = np.random.rand(num_weights) < mutation_rate
            mutation_values1 = np.random.randn(num_weights) * mutation_scale
            offspring1[mutation_mask1] += mutation_values1[mutation_mask1]

            mutation_mask2 = np.random.rand(num_weights) < mutation_rate
            mutation_values2 = np.random.randn(num_weights) * mutation_scale
            offspring2[mutation_mask2] += mutation_values2[mutation_mask2]

            new_population.extend([offspring1, offspring2])

        # Ensure population size does not exceed the limit
        population = new_population[:population_size]

    # After GA, set the best individual's weights and evaluate on the test set
    set_weights_from_flat(model, best_individual_overall)

    # Evaluate on test set
    y_pred_test = model.predict(test_X)
    test_loss = mean_squared_error(test_y, y_pred_test)
    print(f"Test Loss (MSE): {test_loss:.6f}")

    # Compute RMSE and R2 Score
    test_predictions = scaler_y.inverse_transform(y_pred_test)
    test_actual = scaler_y.inverse_transform(test_y)
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    r2 = r2_score(test_actual, test_predictions)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

    if results_dir:
        # Save model weights
        model_weights_path = os.path.join(results_dir, 'best_model.weights.h5')
        save_model_weights(model, model_weights_path)

        # Save evaluation metrics
        metrics = {
            "test_mse": f"{test_loss:.4f}",
            "test_rmse": f"{rmse:.4f}",
            "test_r2": f"{r2:.4f}"
        }
        metrics_path = os.path.join(results_dir, 'evaluation_metrics.txt')
        save_evaluation_metrics(metrics, metrics_path)

        # Removed automatic plotting as per your request
        # You can perform plotting separately in your notebook or script

    return model, fitness_history, {
        "mse": test_loss,
        "rmse": rmse,
        "r2": r2,
        "predictions": test_predictions,
        "actual": test_actual
    }

if __name__ == "__main__":
    # Example usage
    from src.models import create_ga_model
    from src.data import load_data
    import os

    # Load data
    train_gen, val_gen, test_gen, scaler_X, scaler_y = load_data()

    # Create model
    input_shape = (12, 8)  # window_size=12, num_features=8
    model = create_ga_model(input_shape=input_shape)

    # Set up results directory
    results_dir = "example_results_ga"
    os.makedirs(results_dir, exist_ok=True)

    # Run genetic algorithm
    best_model, fitness_history, eval_results = genetic_algorithm(
        model,
        train_gen,
        val_gen,
        test_gen,
        scaler_y,
        population_size=50,
        generations=50,  # Adjust as needed
        results_dir=results_dir
    )

    print("Genetic Algorithm optimization complete. Check the 'example_results_ga' directory for outputs.")