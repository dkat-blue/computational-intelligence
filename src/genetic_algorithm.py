import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

def genetic_algorithm(model, train_data, val_data, test_data, scaler_y, population_size=50, generations=100, 
                      tournament_size=3, mutation_rate=0.005, mutation_scale=0.1, results_dir=None):
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    # Initialize population
    initial_weights_flat = get_weights_flat(model)
    num_weights = len(initial_weights_flat)
    population = [np.random.randn(num_weights) * 0.1 for _ in range(population_size)]

    best_fitness_overall = -np.inf
    best_individual_overall = None
    fitness_history = []

    for generation in range(generations):
        print(f"Generation {generation}")
        # Evaluate fitness for each individual
        fitnesses = []
        for individual in population:
            set_weights_from_flat(model, individual)
            # Compute loss on training data
            y_pred = model.predict(train_X, verbose=0)
            E_Wi = mean_squared_error(train_y, y_pred)
            FI_Wi = compute_fitness(E_Wi)
            fitnesses.append(FI_Wi)
        fitnesses = np.array(fitnesses)

        # Keep track of the best individual
        best_fitness_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_fitness_idx]
        best_individual = population[best_fitness_idx]
        print(f"Best fitness: {best_fitness:.6f}")
        fitness_history.append(best_fitness)

        # Check for early stopping
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_individual_overall = best_individual.copy()

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

    # Compute RMSE
    test_predictions = scaler_y.inverse_transform(y_pred_test)
    test_actual = scaler_y.inverse_transform(test_y)
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    print(f"Test RMSE: {rmse:.4f}")

    if results_dir:
        # Plot fitness over generations
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Over Generations')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'ga_fitness_history.png'))
        plt.close()

    return model, fitness_history, {
        "mse": test_loss,
        "rmse": rmse,
        "predictions": test_predictions,
        "actual": test_actual
    }

if __name__ == "__main__":
    # Example usage
    from src.models import create_ga_model
    from src.data import load_data
    import os

    # Load data
    (train_X, train_y), (val_X, val_y), (test_X, test_y), scaler_X, scaler_y = load_data()

    # Create model
    model = create_ga_model(input_shape=(train_X.shape[1],))

    # Set up results directory
    results_dir = "example_results_ga"
    os.makedirs(results_dir, exist_ok=True)

    # Run genetic algorithm
    best_model, fitness_history, eval_results = genetic_algorithm(
        model,
        (train_X, train_y),
        (val_X, val_y),
        (test_X, test_y),
        scaler_y,
        population_size=50,
        generations=50,  # Reduced for quicker example
        results_dir=results_dir
    )

    print("Genetic Algorithm optimization complete. Check the 'example_results_ga' directory for outputs.")