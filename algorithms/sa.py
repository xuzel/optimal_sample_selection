import numpy as np
from sko.SA import SA
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter
from tqdm import tqdm

from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function, save_result_to_file

def main(iteration, param_version, output_folder):
    Debug = False
    algorithm = 'SA'

    start_time = perf_counter()

    for i in tqdm(range(1, 7)):
        sat_info = TEST_SET[hash_function(i)]
            
        n_dim = sat_info.get_input_len()
        x0=np.random.choice([0, 1], size=n_dim)
        T_max = 1
        T_min = 1e-9
        L = 300
        max_stay_counter = 150

        
        sa = SA(func=fitness_func_with_param(sat_info),
                x0=x0,
                T_max=T_max,
                T_min=T_min,
                L=L,
                max_stay_counter=max_stay_counter)

        solution = sa.run()[0]
        solution = [round(x) for x in solution]

        run_time = perf_counter() - start_time
        result = {
            'qusetion_num': i,
            'T_max': T_max,
            'T_min': T_min,
            'L': L,
            'max_stay_counter': max_stay_counter,
            'solution': sum(solution),
            'time': run_time
        }
        save_result_to_file(algorithm, iteration, param_version, result, output_folder, sa, qusetion_num=i)
        
        if Debug:

            print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
            print(f'the number of the solution is {sum(solution)}')
            print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')
            print(f'the time run is {perf_counter() - start_time} second')
            plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
            plt.show()


# if __name__ == '__main__':
#     main()
