import math
from time import perf_counter
import time
import numpy
import itertools
import typing
from string import ascii_uppercase
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
from upsetplot import plot
from upsetplot import from_contents

data_order = list
DEBUG = False


def n_set_intersections(sets_list, n):
    """
    计算给定列表中任意 n 个集合的交集，并将所有这些交集的结果累加到一个新的集合中。

    参数:
    sets_list (list): 包含多个集合的列表。
    n (int): 要求交集的集合数量。

    返回:
    set: 包含所有 n 个集合交集元素的集合。
    """
    # 创建一个空集合用于存储所有 n 个集合交集的元素
    all_intersections = set()

    # 遍历所有可能的 n 个集合的组合
    for combo in combinations(sets_list, n):
        # 计算当前组合的交集
        if combo:
            intersection = set.intersection(*combo)
            # 将交集的元素添加到 all_intersections 集合中
            all_intersections.update(intersection)

    return all_intersections


class SatInfo:
    def __init__(self, m: int, n: int, k: int, j: int, s: int, **kwargs):
        if 'custom_arr' in kwargs and kwargs['custom_arr']:
            assert len(
                kwargs['custom_arr']) == n, f"the len of input arr is {len(kwargs['custom_arr'])}, but we need {n}"
            self.n_set = kwargs['custom_arr']
        else:
            self.n_set = data_order(ascii_uppercase[:n])
        self.all_k_set = data_order(itertools.combinations(self.n_set, k))
        self.all_j_set = data_order(itertools.combinations(self.n_set, j))
        self.all_s_set = [data_order(itertools.combinations(each_j, s)) for each_j in self.all_j_set]
        self.graph, self.reverse_graph = self.subset_cover_graph(k, j, s)
        # print(len(self.graph))

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
        reverse_cover_graph = defaultdict(list)
        for k_comb in k_combinations:
            for j_sub in j_subsets:
                if any(set(subset).issubset(k_comb) for subset in itertools.combinations(j_sub, s)):
                    cover_graph[tuple(k_comb)].append(tuple(j_sub))
                    reverse_cover_graph[tuple(j_sub)].append(tuple(k_comb))
        return cover_graph, reverse_cover_graph

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

    def calculate_only_set(self):
        all_set = list()
        for value in self.reverse_graph.values():
            this_set = set()
            for each_sat in value:
                this_set.add(frozenset(each_sat))
            all_set.append(this_set)
        # print(all_set)
        union_set = set.union(*all_set)
        # print(len(all_set))
        # print(len(union_set))
        # print(all_set)
        # for k, v in self.reverse_graph.items():
        #     print(k)
        #     print(v)
        #     print()
        # all_intersections = n_set_intersections(all_set, 6)
        # print(len(all_intersections))
        # data = {}
        # for index, x in enumerate(all_set):
        #     data[f'set{index}'] = x
        # upset_data = from_contents(data)
        # plot(upset_data, show_counts=True)
        # plt.show()

        return union_set, all_set


class Result:
    def __init__(self,
                 solution: typing.List,
                 solution_num: int,
                 input_parm: typing.List,
                 algorithm: str,
                 encoder_solution: typing.List,
                 valid: bool,
                 run_time: float,
                 y_history: typing.Union[typing.List, pd.DataFrame]):
        self.solution = solution
        self.solution_num = int(solution_num)
        self.input_parm = input_parm
        self.algorithm = algorithm
        self.encoder_solution = encoder_solution
        self.valid = valid
        self.run_time = run_time
        self.y_history = y_history
        self.title = (f"m: {self.input_parm[0]}, "
                      f"n: {self.input_parm[1]}, "
                      f"k: {self.input_parm[2]}, "
                      f"j: {self.input_parm[3]}, "
                      f"s: {self.input_parm[4]}, "
                      f"alg: {self.algorithm}, "
                      f"num_sol: {self.solution_num}, "
                      f"valid: {self.valid}")

    def print_result(self, draw_pic: bool = False):
        print('*' * 100)
        print(f"using algorithm: {self.algorithm}")
        print(f"the solution is: {self.solution}")
        print(f"the number of the solution is: {self.solution_num}")
        print(f"the solution is: {self.valid}")
        print(f"the run time is: {self.run_time}")
        print('*' * 100, end='\n\n')
        if draw_pic:
            if self.algorithm == 'ga':
                fig, ax = plt.subplots(2, 1)
                ax[0].plot(self.y_history.index, self.y_history.values, '.', color='red')
                self.y_history.min(axis=1).cummin().plot(kind='line')
            elif self.algorithm == 'sa':
                plt.plot(pd.DataFrame(self.y_history).cummin(axis=0))
            elif self.algorithm == 'greedy':
                return self
            else:
                plt.plot(self.y_history)
            plt.suptitle(self.title, fontsize=12, color='purple', fontweight='bold')
            plt.show()
        return self

    def save_fit_func_pic(self, file_path, file_name):
        # plt.title(self.title)
        if self.algorithm == 'ga':
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.y_history.index, self.y_history.values, '.', color='red')
            self.y_history.min(axis=1).cummin().plot(kind='line')
        elif self.algorithm == 'sa':
            plt.plot(pd.DataFrame(self.y_history).cummin(axis=0))
        elif self.algorithm == 'greedy':
            return self
        else:
            plt.plot(self.y_history)
        plt.suptitle(self.title, fontsize=12, color='purple', fontweight='bold')
        file_name_format = self.title.replace(' ', '')
        file_name_format = file_name_format.replace(',', '-')
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        plt.savefig(os.path.join(file_path, f"{file_name}-{file_name_format}.jpg"))
        plt.close()
        return self


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


def run_greedy(sample_parm: typing.List[int], **kwargs):
    start_time = perf_counter()
    sat_info = SatInfo(*sample_parm, **kwargs)
    solution = sat_info.greedy_set_cover()
    end_time = perf_counter()
    result = Result(
        solution=solution,
        solution_num=len(solution),
        input_parm=sample_parm,
        algorithm='greedy',
        encoder_solution=sat_info.encoder_list(solution),
        valid=True,
        run_time=end_time - start_time,
        y_history=[]
    )
    return result


if __name__ == '__main__':
    data = SatInfo(45, 8, 6, 4, 4)
    data.calculate_only_set()
    # for value in data.graph.values():
    #     print(len(value))
