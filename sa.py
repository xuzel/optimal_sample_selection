import numpy as np
from sko.SA import SA
import matplotlib.pyplot as plt
import pandas as pd
from data_structure import SatInfo, fitness_func_with_param, TEST_SET, hash_function
from time import perf_counter


def main():
    start_time = perf_counter()
    sat_info = TEST_SET[hash_function(4)]
    n_dim = sat_info.get_input_len()
    sa = SA(func=fitness_func_with_param(sat_info),
            x0=np.random.choice([0, 1], size=n_dim),
            T_max=1,
            T_min=1e-9,
            L=300,
            max_stay_counter=150)
    solution = sa.run()[0]
    solution = [round(x) for x in solution]
    print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
    print(f'the number of the solution is {sum(solution)}')
    print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')
    print(f'the time run is {perf_counter() - start_time} second')
    plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
    plt.show()


if __name__ == '__main__':
    main()
