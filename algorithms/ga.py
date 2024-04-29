import numpy as np

from .data_structure import SatInfo, fitness_func_with_param, Result
from .utils import TEST_SET, hash_function, replace_rows_randomly

from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import math
from time import perf_counter

import typing
from copy import deepcopy


def run_ga(sample_parm: typing.List[int],
           size_pop: int = 50,
           max_iter: int = 50,
           prob_mut=0.01,
           auto_parm=True,
           greedy_init=True,
           greedy_replace_probability=0.01,
           **kwargs):
    start_time = perf_counter()
    sat_info = SatInfo(*sample_parm, **kwargs)
    n_dim = sat_info.get_input_len()
    if auto_parm:
        weight = 15
        size_pop = int(math.sqrt(n_dim) * weight)
        size_pop = size_pop if size_pop % 2 == 0 else size_pop + 1
        if size_pop > 200:
            size_pop = 200
        # print(size_pop)
    ga = GA(
        func=fitness_func_with_param(sat_info),
        n_dim=n_dim,
        size_pop=size_pop,
        max_iter=max_iter,
        prob_mut=prob_mut,
        lb=[0] * n_dim,
        ub=[1] * n_dim,
        precision=1
    )
    if greedy_init:
        greedy_solution = np.array(sat_info.encoder_greedy_solution())
        print('find greedy init')
        ga.Chrom = replace_rows_randomly(ga.Chrom, greedy_replace_probability, greedy_solution)
    solution = ga.run()[0]
    end_time = perf_counter()
    result = Result(
        solution=sat_info.choose_list(solution),
        solution_num=sum(solution),
        input_parm=sample_parm,
        algorithm='ga',
        encoder_solution=solution,
        valid=sat_info.all_j_subsets_covered(solution),
        run_time=end_time - start_time,
        y_history=pd.DataFrame(ga.all_history_Y)
    )
    return result


def main():
    start_time = perf_counter()
    sat_info = TEST_SET[hash_function(4)]
    gene_size = sat_info.get_input_len()
    print(gene_size)
    num_generations = 1000
    weight = 15
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
    print(f'the solution is:\n{sat_info.choose_list(solution[0])}\n{solution}\n')
    print(f'the number of the solution is {sum(solution[0])}')
    # print(f'valid the solution is {sat_info.all_j_subsets_covered(sat_info.choose_list(solution[0]))}')
    print(f'valid the solution is {sat_info.all_j_subsets_covered(solution[0])}')
    print(f'the time run is {perf_counter() - start_time} second')

    solution = solution[0]
    decoder_list = sat_info.choose_list(solution)
    print(decoder_list)
    encoder_list = sat_info.encoder_list(decoder_list)
    print(encoder_list)
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.show()


if __name__ == '__main__':
    run_ga(
        [45, 12, 6, 6, 4],
        auto_parm=False,
        max_iter=100
    ).print_result(True)
