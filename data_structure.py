import math
import numpy
import itertools
import typing
from string import ascii_uppercase
from collections import defaultdict

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


def fitness_func_with_param(set_info: SatInfo):
    def fitness_func(solution):
        fix_solution = list()
        for x in fix_solution:
            if x < 0.5:
                fix_solution.append(0)
            else:
                fix_solution.append(1)
        if set_info.all_j_subsets_covered(solution):
            return sum(solution)
        else:
            return math.inf

    return fitness_func

TEST_SET = [SatInfo(45, 7, 6, 5, 5),
            SatInfo(45, 8, 6, 4, 4),
            SatInfo(45, 9, 6, 4, 4),
            SatInfo(45, 8, 6, 6, 5),
            SatInfo(45, 9, 6, 5, 4),
            SatInfo(45, 10, 6, 6, 4),
            # SatInfo()

            ]


def hash_function(x):
    return x - 1