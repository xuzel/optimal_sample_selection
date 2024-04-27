import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import perf_counter
from tqdm import tqdm
import csv
import os

from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function, save_result_to_file

def fitness_func_with_param(set_info: SatInfo):
    def fitness_func(solution):
        if set_info.all_j_subsets_covered(solution):
            return sum(solution)
        else:
            return math.inf

    return fitness_func


# def greedy_set_cover(cover_graph):
#     covered = set()
#     selected_k_combs = []
#     while any(j_sub not in covered for j_subsets in cover_graph.values() for j_sub in j_subsets):
#         best_k_comb = max(cover_graph, key=lambda k: len(set(cover_graph[k]) - covered))
#         selected_k_combs.append(best_k_comb)
#         covered.update(cover_graph[best_k_comb])
#     return selected_k_combs


def main(iteration, param_version, output_folder):
    algorithm = 'GA'

    start_time = perf_counter()
    sat_info = SatInfo(45, 9, 6, 4, 4)

    #参数：种群大小，迭代次数，变异概率
    gene_size = sat_info.get_input_len()
    num_generations = 600
    prob_mut=0.01,
    weight = 20 
    size_pop = int(math.sqrt(gene_size) * weight)
    size_pop = size_pop if size_pop % 2 == 0 else size_pop + 1


    ga = GA(func=fitness_func_with_param(sat_info),
            n_dim=gene_size,
            size_pop=size_pop,
            max_iter=num_generations,
            prob_mut=prob_mut,  # 0.001
            lb=[0] * sat_info.get_input_len(),
            ub=[1] * sat_info.get_input_len(),
            precision=1)
    solution = ga.run()
    run_time = perf_counter() - start_time

    # Store the solution and time in a dictionary
    result = {
        'gene_size': gene_size, 
        'num_generations': num_generations, 
        'prob_mut': prob_mut, 
        'size_pop': size_pop,
        'solution': sum(solution[0]), 
        'time': run_time
        }
    save_result_to_file(algorithm, iteration, param_version, result, output_folder, ga)

# if __name__ == '__main__':
#     i = 5  # Number of iterations
#     param_version = 1  # Change this for different parameter versions
#     output_folder = 'results_GA'
#     for iteration in tqdm(range(i)):
#         main(iteration + 1, param_version, output_folder)