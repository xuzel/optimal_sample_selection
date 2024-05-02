import sys
import numpy as np
import random
import itertools
import time
import matplotlib.pyplot as plt
import math
from .data_structure import SatInfo, fitness_func_with_param, Result
from .utils import TEST_SET, hash_function
import typing


def run_sa_greedy(sample_parm: typing.List[int],
                        **kwargs):
    start_time = time.perf_counter()
    sat_info = SatInfo(*sample_parm, **kwargs)
    cover_graph = sat_info.graph
    current_solution = simulated_annealing(cover_graph, T=10000, T_min=0.001, alpha=0.99, time_limit=8)
    solution = np.array(current_solution)
    # print(solution,[tuple(row) for row in solution],solution.shape[0])
    end_time = time.perf_counter()
    result = Result(
        solution=[tuple(row) for row in solution],
        solution_num=solution.shape[0],
        input_parm=sample_parm,
        algorithm='sa_greedy',
        encoder_solution=None,
        valid=True,
        run_time=end_time - start_time,
        y_history=None  # Assuming you have some way to track history in your SA implementation
    )

    return result

def all_j_subsets_covered(cover_graph, solution):
    all_j_subsets = set(itertools.chain(*cover_graph.values()))
    covered_j_subsets = set(itertools.chain(*[cover_graph[k] for k in solution]))
    return covered_j_subsets == all_j_subsets

def simulated_annealing(cover_graph, T=10000, T_min=0.001, alpha=0.99, time_limit=8):
    start_time = time.time()
    k_combinations = list(cover_graph.keys())
    random.shuffle(k_combinations)
    current_solution = k_combinations[:len(k_combinations) // 4]
    while not all_j_subsets_covered(cover_graph, current_solution):
        current_solution.append(random.choice([k for k in k_combinations if k not in current_solution]))
    current_energy = len(current_solution)
    while T > T_min:
        if time.time() - start_time > time_limit:
            print("Switching to greedy algorithm due to time limit.")
            return greedy_set_cover(cover_graph)
        for _ in range(50):
            new_solution = current_solution[:]
            if random.random() > 0.5 and len(new_solution) > 1:
                new_solution.remove(random.choice(new_solution))
            else:
                possible_additions = [k for k in k_combinations if k not in new_solution]
                if possible_additions:
                    new_solution.append(random.choice(possible_additions))
            if all_j_subsets_covered(cover_graph, new_solution):
                new_energy = len(new_solution)
                if new_energy < current_energy or math.exp((current_energy - new_energy) / T) > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
        T *= alpha
    return current_solution


def greedy_set_cover(cover_graph):
    covered = set()
    selected_k_combs = []
    while any(j_sub not in covered for j_subsets in cover_graph.values() for j_sub in j_subsets):
        best_k_comb = max(cover_graph, key=lambda k: len(set(cover_graph[k]) - covered))
        selected_k_combs.append(best_k_comb)
        covered.update(cover_graph[best_k_comb])
    return selected_k_combs

def main():
    pass

if __name__ == '__main__':
    run_sa_greedy(
        [45, 8, 6, 4, 4]
    ).print_result(False)