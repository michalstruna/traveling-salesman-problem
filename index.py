#!/usr/bin/python3

import time
import numpy as np

from io_utils import Reader, Writer
from genetic import Genetic, FitnessType, Crossover
from astar import PathFinder

writer = Writer()
reader = Reader()
args = reader.read_args()
cities, distances = reader.read_excel(args.input)

start = time.time()

if args.astar:
    finder = PathFinder()
    path, distance = finder.get_path(distances)
else:
    genetic = Genetic(
        distances = distances,
        population = args.population,
        fitness_type = FitnessType.LOWEST,
        mutation_prob = args.mutation,
        crossover = Crossover.ORDER1,
        on_iteration = writer.write_progress_bar,
        generations = args.generations,
        players = args.tournament
    )

    path, best_fitness, mean_fitness, worst_fitness = genetic.get_path()
    distance = best_fitness[-1]


end = time.time()
writer.write_result(path, cities, distance, end - start)

if not args.astar and args.fitness:
    writer.show_fitness_history(best_fitness, mean_fitness, worst_fitness)