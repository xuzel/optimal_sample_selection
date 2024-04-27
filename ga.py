import numpy as np
from data_structure import SatInfo, fitness_func_with_param
from utils import TEST_SET, hash_function
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import perf_counter
from tqdm import tqdm
import csv
import os

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


def main(iteration):
    start_time = perf_counter()
    sat_info = SatInfo(45, 9, 6, 4, 4)
    gene_size = sat_info.get_input_len()
    print(gene_size)
    num_generations = 600
    weight = 20
    size_pop = int(math.sqrt(gene_size) * weight)
    size_pop = size_pop if size_pop % 2 == 0 else size_pop + 1
    print(f"pop size = {size_pop}")
    ga = GA(func=fitness_func_with_param(sat_info),
            n_dim=gene_size,
            size_pop=size_pop,
            max_iter=num_generations,
            prob_mut=0.01,  # 0.001
            lb=[0] * sat_info.get_input_len(),
            ub=[1] * sat_info.get_input_len(),
            precision=1)
    solution = ga.run()
    # print(f'the solution is:\n{sat_info.choose_list(solution[0])}\n{solution}\n')
    # print(f'the number of the solution is {sum(solution[0])}')
    # print(f'valid the solution is {sat_info.all_j_subsets_covered(sat_info.choose_list(solution[0]))}')
    # print(f'the time run is {perf_counter() - start_time} second')
    run_time = perf_counter() - start_time

    # Store the solution and time in a dictionary
    result = {'solution': sum(solution[0]), 'time': run_time}

    # Write the result to a CSV file
    with open('results.csv', 'a', newline='') as csvfile:
        fieldnames = ['solution', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only once
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(result)

    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    
    # Create a directory for the plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Save the figure to the 'plots' directory
    plt.savefig(f'plots/plot_{iteration}.png')
    iteration += 1
    
    # plt.show()


if __name__ == '__main__':
    i = 100
    for _ in tqdm(range(i)):
        main(i)
