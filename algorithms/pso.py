from sko.PSO import PSO
import matplotlib.pyplot as plt
from time import perf_counter
from tqdm import tqdm

from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function,save_result_to_file

def main(iteration, param_version, output_folder):
    Debug = False
    algorithm = 'PSO'

    start_time = perf_counter()

    for i in tqdm(range(1, 7)):
        sat_info = TEST_SET[hash_function(i)]

        n_dim = sat_info.get_input_len()
        pop = 200
        max_iter = 200
        w = 0.9
        c1 = 2
        c2 = 2

        pso = PSO(func=fitness_func_with_param(sat_info),
                n_dim=n_dim,
                pop=pop,
                max_iter=max_iter,
                lb=[0] * n_dim,
                ub=[1] * n_dim,
                w=w,
                c1=c1,
                c2=c2)

        solution = pso.run()
        end_time = perf_counter()
        run_time = perf_counter() - start_time
        solution = [round(x) for x in solution[0]]

        result = {
            'n_dim': n_dim,
            'pop': pop,
            'max_iter': max_iter,
            'w': w,
            'c1': c1,
            'c2': c2,
            'solution': sum(solution),
            'time': run_time
        }
        save_result_to_file(algorithm, iteration, param_version, result, output_folder, pso, qusetion_num=i)

        if Debug:
            # print(solution)
            print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
            print(f'the number of the solution is {sum(solution)}')
            print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')

            print(f"run time: {end_time - start_time} seconds")

            plt.plot(pso.gbest_y_hist)
            plt.show()


# if __name__ == '__main__':
#     main()
