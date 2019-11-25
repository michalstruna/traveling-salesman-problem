import numpy as np
from enum import Enum
from random import random

class FitnessType(Enum):
    LOWEST = (min, np.argmin)
    HIGHEST = (max, np.argmax)

class Crossover(Enum):

    @staticmethod
    def ORDER1(x, y):
        child = np.zeros(x.shape[0], dtype = np.int32) - 1
        points = np.random.choice(x.shape[0], 2)
        start, end = min(points), max(points)
        child[start:end] = x[start:end]
        Crossover._fill(child, y)
        return child     


    @staticmethod
    def ORDER2(x, y):
        child = np.zeros(x.shape[0], dtype = np.int32) - 1
        points = np.random.choice(x.shape[0], 2)
        start, end = min(points), max(points)
        child[0:end - start] = x[start:end]
        Crossover._fill(child, y)
        return child   

    @staticmethod
    def CYCLE(x, y):
        child = np.zeros(x.shape[0], dtype = np.int32) - 1
        i = np.random.randint(low = 0, high = x.shape[0] - 1)

        while True:
            child[i] = x[i]
            i = np.where(x == y[i])[0][0]

            if child[i] >= 0:
                break

        Crossover._fill(child, y)
        return child

    @staticmethod
    def PMX(x, y):
        child = np.zeros(x.shape[0], dtype = np.int32) - 1
        points = np.random.choice(x.shape[0], 2)
        start, end = min(points), max(points)
        child[start:end] = x[start:end]

        for i in range(start, end):
            if y[i] in child:
                continue

            j = i
            l = 0

            while True:
                l += 1

                if l >= x.shape[0] or j >= x.shape[0]:
                    break

                pos = np.where(y == x[j])[0][0]

                if start <= pos <= end:
                    j = pos
                    continue

                child[pos] = y[i]
                break

        Crossover._fill(child, y)
        return child

    @staticmethod
    def _fill(child, y):
        sx = set(child)
        choices = np.array([yi for yi in y if yi not in sx])
        child[child == -1] = choices

    @staticmethod
    def from_string(string):
        if string == "order1":
            return Crossover.ORDER1
        elif string == "order2":
            return Crossover.ORDER2
        elif string == "cycle":
            return Crossover.CYCLE
        elif string == "pmx":
            return Crossover.PMX


class Genetic:

    def __init__(self, distances, population, fitness_type, mutation_prob, crossover, generations, players, on_iteration = lambda current, max: None):
        self.distances = distances
        self.path_size = self.distances.shape[0]
        self.population = population
        self.get_best = fitness_type.value[0]
        self.get_best_index = fitness_type.value[1]
        self.mutation_prob = mutation_prob
        self.get_child = crossover
        self.on_iteration = on_iteration
        self.generations = generations
        self.players = min(players, population)

    def get_rand_population(self):
        individuals = self.get_empty_population()

        for i in range(self.population):
            individuals[i] = np.arange(0, self.path_size, dtype = np.int32)
            np.random.shuffle(individuals[i, 1:])

        fitness = np.zeros(self.population)

        return individuals, fitness

    def evaluate(self):
        for i, individual in enumerate(self.individuals):
            self.fitness[i] = self.get_fitness(individual)

    def get_fitness(self, individual):
        fitness = 0

        for i in range(len(individual) - 1):
            fitness += self.get_distance(individual[i], individual[i + 1])

        fitness += self.get_distance(individual[0], individual[-1])

        return fitness

    def get_distance(self, city1, city2):
        return self.distances[city1, city2] if city1 < city2 else self.distances[city2, city1]

    def get_empty_population(self):
        return np.empty((self.population, self.path_size), dtype = np.int32)

    def select(self):
        individuals = self.get_empty_population()

        for i in range(self.population):
            players_indexes = np.random.choice(self.population, self.players, replace = False)
            fitness = self.fitness[players_indexes]
            individuals[i] = self.individuals[players_indexes[self.get_best_index(fitness)]]

        self.individuals = individuals

    def mutate(self):
        for individual in self.individuals:
            if random() < self.mutation_prob:
                switch = np.random.choice(self.path_size - 1, 2) + 1
                individual[switch[0]], individual[switch[1]] = individual[switch[1]], individual[switch[0]]

    def crossover(self):
        new = self.get_empty_population()

        for i in range(0, self.population - 1, 2):
            x = self.individuals[i]
            y = self.individuals[i + 1]
            new[i] = self.get_child(x, y)
            new[i + 1] = self.get_child(y, x)

        self.individuals = new

    def get_path(self):
        self.individuals, self.fitness = self.get_rand_population()

        mean_fitness = []
        best_fitness = []
        worst_fitness = []

        for i in range(self.generations): ## TODO: Evaluate last?
            self.evaluate()
            best = self.individuals[self.get_best_index(self.fitness)]
            self.select()
            self.crossover()
            self.mutate()
            self.individuals[np.random.randint(low = 0, high = self.individuals.shape[0])] = best

            best_fitness.append(self.get_best(self.fitness))
            mean_fitness.append(np.mean(self.fitness))
            worst_fitness.append(max(self.fitness))
            self.on_iteration(i, self.generations)

        self.on_iteration(self.generations, self.generations)
        best = np.append(self.individuals[self.get_best_index(self.fitness)], 0)

        return best, best_fitness, mean_fitness, worst_fitness