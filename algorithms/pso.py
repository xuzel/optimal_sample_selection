from sko.PSO import PSO
import matplotlib.pyplot as plt
import numpy as np
from .data_structure import SatInfo, fitness_func_with_param, Result
from .utils import TEST_SET, hash_function, replace_rows_randomly
from time import perf_counter
import typing


def run_pso(sample_parm: typing.List[int],
            greedy_init=True,
            greedy_replace_probability=0.01,
            pop=1000,
            max_iter=100,
            w=0.9,
            c1=5,
            c2=5,
            **kwargs):
    start_time = perf_counter()
    sat_info = SatInfo(*sample_parm, **kwargs)
    n_dim = sat_info.get_input_len()
    pso = PSO(
        func=fitness_func_with_param(sat_info),
        n_dim=n_dim,
        pop=pop,
        max_iter=max_iter,
        lb=[0] * n_dim,
        ub=[1] * n_dim,
        w=w,
        c1=c1,
        c2=c2
    )
    if greedy_init:
        # print(pso.X)
        # print(sum(sat_info.encoder_greedy_solution()))
        greedy_solution = np.array(sat_info.encoder_greedy_solution())
        pso.X = replace_rows_randomly(pso.X, greedy_replace_probability, greedy_solution)
        # print(pso.X)
    solution = pso.run()[0]
    if type(solution) is np.ndarray:
        if solution.ndim > 1:
            solution = [[round(x) for x in sublist] for sublist in solution]
        else:
            solution = [round(x) for x in solution]
    else:
        solution = [round(x) for x in solution]
    end_time = perf_counter()
    result = Result(
        solution=sat_info.choose_list(solution),
        solution_num=sum(solution),
        input_parm=sample_parm,
        algorithm='pso',
        encoder_solution=solution,
        valid=sat_info.all_j_subsets_covered(solution),
        run_time=end_time - start_time,
        y_history=pso.gbest_y_hist
    )
    return result


def main():
    start_time = perf_counter()
    sat_info = TEST_SET[hash_function(4)]
    n_dim = sat_info.get_input_len()

    pso = PSO(func=fitness_func_with_param(sat_info),
              n_dim=n_dim,
              pop=200,
              max_iter=200,
              lb=[0] * n_dim,
              ub=[1] * n_dim,
              w=0.9,
              c1=2,
              c2=2)

    solution = pso.run()
    end_time = perf_counter()
    print(solution[0])
    # solution = [round(x) for x in solution[0]]
    solution = [[round(x) for x in sublist] for sublist in solution]
    # print(solution)
    print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
    print(f'the number of the solution is {sum(solution)}')
    print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')
    print(f"run time: {end_time - start_time} seconds")

    plt.plot(pso.gbest_y_hist)
    plt.show()


if __name__ == '__main__':
    run_pso(
        [45, 9, 6, 4, 4]
    ).print_result(True)
