import os
import random
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

from src.utils import extract_data_from_generator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Optimizer:
    """
    Base class for optimizers.
    """
    def optimize(self, model, train_generator, val_generator, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_hyperparameters(self):
        raise NotImplementedError("Subclasses should implement this method.")

class GeneticAlgorithmOptimizer(Optimizer):
    """
    Optimizer class for Genetic Algorithm optimization.
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

    def optimize(self, model, train_generator, val_generator, results_dir=None, **kwargs):
        # Extract data from generators
        val_X, val_y = extract_data_from_generator(val_generator)

        initial_weights_flat = self.get_weights_flat(model)
        num_weights = len(initial_weights_flat)
        population = [np.random.randn(num_weights) * 0.1 for _ in range(self.population_size)]

        best_fitness_overall = -np.inf
        best_individual_overall = None
        generations_without_improvement = 0

        for generation in range(self.generations):
            logger.info(f"Generation {generation}")
            fitnesses = []
            for individual in population:
                self.set_weights_from_flat(model, individual)
                y_pred = model.predict(val_X, verbose=0)
                val_y_flat = np.squeeze(val_y)
                y_pred_flat = np.squeeze(y_pred)
                mse = mean_squared_error(val_y_flat, y_pred_flat)
                fitness = 1 / (mse + 1e-8)
                fitnesses.append(fitness)
            fitnesses = np.array(fitnesses)

            best_fitness_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_fitness_idx]
            best_individual = population[best_fitness_idx]
            logger.info(f"Best fitness: {best_fitness:.6f}")

            self.fitness_history.append(best_fitness)

            if best_fitness > best_fitness_overall:
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
            for w in layer.get_weights():
                shape = w.shape
                size = np.prod(shape)
                new_w = flat_weights[idx:idx+size].reshape(shape)
                weights.append(new_w)
                idx += size
        model.set_weights(weights)

    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx])
        return selected

    def crossover(self, parent1, parent2):
        num_weights = len(parent1)
        mask = np.random.rand(num_weights) < 0.5
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        return offspring1, offspring2

    def mutate(self, individual):
        num_weights = len(individual)
        mutation_mask = np.random.rand(num_weights) < self.mutation_rate
        mutation_values = np.random.randn(num_weights) * self.mutation_scale
        individual[mutation_mask] += mutation_values[mutation_mask]
        return individual
