import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import tensorflow as tf

from src.utils import extract_data_from_generator

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Optimizer:
    """
    Base class for optimizers.
    """
    def optimize(self, model, train_generator, val_generator, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_hyperparameters(self):
        raise NotImplementedError("Subclasses should implement this method.")

class SGDOptimizer(Optimizer):
    """
    Optimizer class for Stochastic Gradient Descent optimization with Early Stopping and ReduceLROnPlateau.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0, epochs=10, patience=5,
                 reduce_lr_patience=3, reduce_lr_factor=0.5, min_lr=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.patience = patience  # For EarlyStopping
        self.reduce_lr_patience = reduce_lr_patience  # For ReduceLROnPlateau
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.hyperparameters = {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "patience": self.patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "reduce_lr_factor": self.reduce_lr_factor,
            "min_lr": self.min_lr
        }
        self.history = None  # Store training history

    def get_hyperparameters(self):
        return self.hyperparameters

    def optimize(self, model, train_generator, val_generator, **kwargs):
        """
        Perform SGD optimization using Keras' fit method with Early Stopping and ReduceLROnPlateau.

        Args:
            model: The Keras model to optimize.
            train_generator: Training data generator.
            val_generator: Validation data generator.
            **kwargs: Additional arguments to pass to model.fit().
        """
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )
        model.compile(optimizer=optimizer, loss='mse')

        # EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )

        # ReduceLROnPlateau callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
            verbose=1
        )

        callbacks = [early_stopping, reduce_lr]

        self.history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            **kwargs
        )
        logger.info("SGD optimization completed")
        return self.history

class GeneticAlgorithmOptimizer(Optimizer):
    """
    Optimizer class for Genetic Algorithm optimization using MSE as the fitness metric (to be minimized).
    """
    def __init__(self, population_size=50, generations=100, tournament_size=3,
                 mutation_rate=0.005, mutation_scale=0.1, patience=10):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.patience = patience
        self.fitness_history = []
        self.hyperparameters = {
            "population_size": self.population_size,
            "generations": self.generations,
            "tournament_size": self.tournament_size,
            "mutation_rate": self.mutation_rate,
            "mutation_scale": self.mutation_scale,
            "patience": self.patience
        }

    def get_hyperparameters(self):
        return self.hyperparameters

    def optimize(self, model, train_generator, val_generator, scaler_y, results_dir=None, **kwargs):
        """
        Perform optimization using a Genetic Algorithm.

        Args:
            model: The Keras model to optimize.
            train_generator: Training data generator.
            val_generator: Validation data generator.
            scaler_y: The scaler used for the target variable.
            results_dir (str): Directory to save results.
            **kwargs: Additional arguments.
        """
        # Extract data from generators
        val_X, val_y = extract_data_from_generator(val_generator)
        val_y_unscaled = scaler_y.inverse_transform(val_y)

        initial_weights_flat = self.get_weights_flat(model)
        num_weights = len(initial_weights_flat)
        # Initialize population around current model weights
        population = [initial_weights_flat + np.random.randn(num_weights) * 0.1 for _ in range(self.population_size)]

        best_fitness_overall = np.inf  # Initialize to infinity for minimization
        best_individual_overall = None
        generations_without_improvement = 0

        for generation in range(self.generations):
            logger.info(f"Generation {generation}")
            fitnesses = []
            for individual in population:
                self.set_weights_from_flat(model, individual)
                y_pred = model.predict(val_X, verbose=0)
                y_pred_unscaled = scaler_y.inverse_transform(y_pred)
                val_y_flat = np.squeeze(val_y_unscaled)
                y_pred_flat = np.squeeze(y_pred_unscaled)
                mse = mean_squared_error(val_y_flat, y_pred_flat)
                fitness = mse  # Computed on unscaled data
                fitnesses.append(fitness)
            fitnesses = np.array(fitnesses)

            best_fitness_idx = np.argmin(fitnesses)  # Use np.argmin for minimization
            best_fitness = fitnesses[best_fitness_idx]
            best_individual = population[best_fitness_idx]
            logger.info(f"Best fitness (lowest MSE): {best_fitness:.6f}")

            self.fitness_history.append(best_fitness)

            if best_fitness < best_fitness_overall:
                best_fitness_overall = best_fitness
                best_individual_overall = best_individual.copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                if generations_without_improvement >= self.patience:
                    logger.info(f"No improvement for {self.patience} generations. Early stopping at generation {generation}.")
                    break

            selected_population = self.tournament_selection(population, fitnesses)

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            population = new_population[:self.population_size]

        # Set the best weights found
        self.set_weights_from_flat(model, best_individual_overall)
        logger.info("Genetic Algorithm optimization completed")

        # Save fitness history
        if results_dir:
            fitness_history_path = os.path.join(results_dir, 'fitness_history.txt')
            self.save_fitness_history(fitness_history_path)
            logger.info(f"Fitness history saved to: {fitness_history_path}")

    def save_fitness_history(self, file_path):
        """
        Save the fitness history to a text file.

        Args:
            file_path (str): Path to save the fitness history file.
        """
        with open(file_path, 'w') as f:
            for fitness in self.fitness_history:
                f.write(f"{fitness}\n")
        logger.info(f"Fitness history saved to {file_path}")

    def get_weights_flat(self, model):
        weights = model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        return flat_weights

    def set_weights_from_flat(self, model, flat_weights):
        weights = []
        idx = 0
        for layer in model.layers:
            layer_weights = layer.get_weights()
            new_layer_weights = []
            for w in layer_weights:
                shape = w.shape
                size = np.prod(shape)
                new_w = flat_weights[idx:idx+size].reshape(shape)
                new_layer_weights.append(new_w)
                idx += size
            layer.set_weights(new_layer_weights)

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]  # Use np.argmin
            selected.append(population[winner_idx])
        return selected

    def crossover(self, parent1, parent2):
        """
        Perform single-point crossover between two parents.

        Args:
            parent1 (np.array): First parent individual.
            parent2 (np.array): Second parent individual.

        Returns:
            tuple: Two offspring individuals resulting from the crossover.
        """
        num_weights = len(parent1)
        if num_weights <= 1:
            # No crossover possible if only one gene
            return parent1.copy(), parent2.copy()

        crossover_point = np.random.randint(1, num_weights)
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def mutate(self, individual):
        num_weights = len(individual)
        mutation_mask = np.random.rand(num_weights) < self.mutation_rate
        mutation_values = np.random.randn(num_weights) * self.mutation_scale
        individual[mutation_mask] += mutation_values[mutation_mask]
        return individual
