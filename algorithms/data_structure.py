import math
import numpy
import itertools
import typing
from string import ascii_uppercase
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


data_order = list
DEBUG = True


class SatInfo:
    def __init__(self, m: int, n: int, k: int, j: int, s: int):
        self.n_set = data_order(ascii_uppercase[:n])
        self.all_k_set = data_order(itertools.combinations(self.n_set, k))
        self.all_j_set = data_order(itertools.combinations(self.n_set, j))
        self.all_s_set = [data_order(itertools.combinations(each_j, s)) for each_j in self.all_j_set]
        self.graph = self.subset_cover_graph(k, j, s)

        if DEBUG:
            print(f"all j set: \n{self.all_j_set}\n")
            print(f"all k set: \n{self.all_k_set}\n")
            print(f"all n set: \n{self.n_set}\n")
            print(f"all s set: \n{self.all_s_set}\n")


    def get_input_len(self):
        return len(self.all_k_set)

    def choose_list(self, encoder_list: typing.List[int]):
        return [self.all_k_set[i] for i in range(self.get_input_len()) if encoder_list[i] == 1]

    def subset_cover_graph(self, k, j, s):
        j_subsets = data_order(itertools.combinations(self.n_set, j))
        k_combinations = data_order(itertools.combinations(self.n_set, k))
        cover_graph = defaultdict(list)
        for k_comb in k_combinations:
            for j_sub in j_subsets:
                if any(set(subset).issubset(k_comb) for subset in itertools.combinations(j_sub, s)):
                    cover_graph[tuple(k_comb)].append(tuple(j_sub))
        return cover_graph

    def all_j_subsets_covered(self, solution):
        # solution = self.choose_list(solution)
        if type(solution[0]) is numpy.float64 or int:
            solution = self.choose_list(solution)
        all_j_subsets = set(itertools.chain(*self.graph.values()))
        covered_j_subsets = set(itertools.chain(*[self.graph[k] for k in solution]))
        return covered_j_subsets == all_j_subsets

    def greedy_set_cover(self):
        covered = set()
        selected_k_combs = []
        while any(j_sub not in covered for j_subsets in self.graph.values() for j_sub in j_subsets):
            best_k_comb = max(self.graph, key=lambda k: len(set(self.graph[k]) - covered))
            selected_k_combs.append(best_k_comb)
            covered.update(self.graph[best_k_comb])
        return selected_k_combs

    def encoder_list(self, input_list):
        set2 = set(input_list)
        output_list = [1 if item in set2 else 0 for item in self.all_k_set]
        return output_list

    def encoder_greedy_solution(self):
        return self.encoder_list(self.greedy_set_cover())


class Result:
    def __init__(self,
                 solution: typing.List,
                 solution_num: int,
                 algorithm: str,
                 encoder_solution: typing.List,
                 valid: bool,
                 run_time: float,
                 y_history: typing.Union[typing.List, pd.DataFrame]):
        self.solution = solution
        self.solution_num = solution_num
        self.algorithm = algorithm
        self.encoder_solution = encoder_solution
        self.valid = valid
        self.run_time = run_time
        self.y_history = y_history

    def print_result(self, draw_pic: bool):
        print(f"the solution is: {self.solution}")
        print(f"the number of the solution is: {self.solution_num}")
        print(f"the solution is: {self.valid}")
        print(f"the run time is: {self.run_time}")
        if draw_pic:
            if self.algorithm == 'ga':
                fig, ax = plt.subplots(2, 1)
                ax[0].plot(self.y_history.index, self.y_history.values, '.', color='red')
                self.y_history.min(axis=1).cummin().plot(kind='line')
                plt.show()
            elif self.algorithm == 'sa':
                plt.plot(pd.DataFrame(self.y_history).cummin(axis=0))
                plt.show()
            else:
                plt.plot(self.y_history)
                plt.show()

    def save_fit_func_pic(self, file_name):
        if self.algorithm == 'ga':
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.y_history.index, self.y_history.values, '.', color='red')
            self.y_history.min(axis=1).cummin().plot(kind='line')
        elif self.algorithm == 'sa':
            plt.plot(pd.DataFrame(self.y_history).cummin(axis=0))
        else:
            plt.plot(self.y_history)
        plt.savefig(file_name)


def fitness_func_with_param(set_info: SatInfo):
    def fitness_func(solution):
        fix_solution = list()
        for x in fix_solution:
            if x < 0.5:
                fix_solution.append(0)
            else:
                fix_solution.append(1)
        if set_info.all_j_subsets_covered(solution):
            return float(sum(solution))
        else:
            return math.inf

    return fitness_func

