import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

from genetic import Crossover

class Reader:

    def read_csv(self, path, size = sys.maxsize):
        rows = []

        with open(path) as file:
            reader = csv.reader(file)

            for row in reader:
                rows.append(row)

        data = np.array(rows)
        cities = data[1:, 0]
        distances = data[2:, 2:]

        for i in range(distances.shape[0]):
            distances[i, i] = 0

            for j in range(i):
                distances[i, j] = distances[j, i]
        
        distances = distances.astype(np.float)
        distances = distances[:size, :size]
                
        return cities, distances

    def read_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--fitness", help = "Show fitness history in plot.", action = "store_true")
        parser.add_argument("-i", "--input", help = "Source csv file.", default = "Distance.csv")
        parser.add_argument("-m", "--mutation", help = "Probability of mutation from 0 to 1.", type = float, default = 0.05)
        parser.add_argument("-g", "--generations", help = "Count of generations.", type = int, default = 250)
        parser.add_argument("-p", "--population", help = "Count of individuals in population.", type = int, default = 500)
        parser.add_argument("-c", "--crossover", help = "Crossover operator.", choices = ["order1", "order2", "cycle", "pmx"], default = "order1")
        parser.add_argument("-t", "--tournament", help = "Count of tournament player in selection.", type = int, default = 2)
        parser.add_argument("-a", "--astar", help = "Use A* instead of genetic algorithm.", action = "store_true")
        parser.add_argument("-s", "--size", help = "Select only first N cities from source file.", type = int, default = sys.maxsize)
        args = parser.parse_args()

        args.crossover = Crossover.from_string(args.crossover)

        return args

class Writer:

    def dynamic_write(self, text):
        print(text, end = "\r")

    def write_progress_bar(self, current, max, size = 50):
        progress = round(size * current / max)

        if progress >= size:
            self.dynamic_write(" " * (size + 2))
        else:
            self.dynamic_write("|" + ("█" * progress) + ("-" * (size - progress)) + "|")

    def write(self, text):
        print(text)

    def write_result(self, path, cities, length, time):
        print("")
        self.dynamic_write(f"Vzdálenost: {round(length, 2)}\n")
        print(" → ".join(map(lambda i: cities[i], path)))
        print(f"Doba trvání: {round(time, 2)} s")

    def show_fitness_history(self, best_fitness, mean_fitness, worst_fitness):
        plt.plot(best_fitness, label = 'Best')
        plt.plot(mean_fitness, label = 'Mean')
        plt.plot(worst_fitness, label = 'Worst')
        plt.xlabel('Generace')
        plt.ylabel('Fitness')
        plt.legend()
        plt.ylim(bottom = 0)
        plt.grid()
        plt.show()