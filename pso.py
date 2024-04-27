from sko.PSO import PSO
import matplotlib.pyplot as plt
from data_structure import SatInfo, fitness_func_with_param
from utils import TEST_SET, hash_function
from time import perf_counter


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
    solution = [round(x) for x in solution[0]]
    # print(solution)
    print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
    print(f'the number of the solution is {sum(solution)}')
    print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')

    print(f"run time: {end_time - start_time} seconds")

    plt.plot(pso.gbest_y_hist)
    plt.show()


if __name__ == '__main__':
    main()
