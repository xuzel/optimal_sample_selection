import numpy as np
from sko.SA import SA, SACauchy, SABoltzmann
import matplotlib.pyplot as plt
import pandas as pd

from .data_structure import SatInfo, fitness_func_with_param, Result
from .utils import TEST_SET, hash_function
from time import perf_counter
import typing


def run_sa(sample_parm: typing.List[int],
           random_init=False,
           T_max=100,
           T_min=1e-9,
           L=300,
           max_stay_counter=350,
           **kwargs):
    start_time = perf_counter()
    sat_info = SatInfo(*sample_parm, **kwargs)
    n_dim = sat_info.get_input_len()
    if random_init:
        x0 = np.random.choice([0, 1], size=n_dim)
    else:
        x0 = sat_info.encoder_greedy_solution()
    sa = SA(
        func=fitness_func_with_param(sat_info),
        x0=x0,
        T_max=T_max,
        T_min=T_min,
        L=L,
        max_stay_counter=max_stay_counter
    )
    solution = sa.run()[0]
    if type(solution) is np.ndarray:
        solution = solution.tolist()
    solution = [round(x) for x in solution]
    end_time = perf_counter()
    result = Result(
        solution=sat_info.choose_list(solution),
        solution_num=sum(solution),
        input_parm=sample_parm,
        algorithm='sa',
        encoder_solution=solution,
        valid=sat_info.all_j_subsets_covered(solution),
        run_time=end_time - start_time,
        y_history=sa.generation_best_Y
    )
    return result


def main():
    # start_time = perf_counter()
    # sat_info = TEST_SET[hash_function(4)]
    # n_dim = sat_info.get_input_len()
    # sa = SA(func=fitness_func_with_param(sat_info),
    #         x0=np.random.choice([0, 1], size=n_dim),
    #         T_max=1,
    #         T_min=1e-9,
    #         L=300,
    #         max_stay_counter=150)
    # solution = sa.run()[0]
    # solution = [round(x) for x in solution]
    # print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
    # print(f'the number of the solution is {sum(solution)}')
    # print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')
    # print(f'the time run is {perf_counter() - start_time} second')
    # plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
    # plt.show()
    pass


if __name__ == '__main__':
    run_sa(
        [45, 8, 6, 4, 4]
    ).print_result(True)
