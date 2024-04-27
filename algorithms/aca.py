from copy import deepcopy

import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function, save_result_to_file

class ACABinary:
    def __init__(self, func, n_dim, size_pop=10, max_iter=20, alpha=1, beta=2, rho=0.1):
        self.generation_best_Y = list()
        self.best_path = None
        self.best_cost = None
        self.func = func  # 目标函数
        self.n_dim = n_dim  # 解的维度
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 最大迭代次数
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 适应度的重要程度
        self.rho = rho  # 信息素挥发速度

        self.Tau = np.ones((n_dim, 2)) * 0.5  # 初始化信息素矩阵，每个维度两个选择（0和1）

    def run(self):
        for iter in range(self.max_iter):
            paths = np.zeros((self.size_pop, self.n_dim), dtype=int)
            path_costs = np.zeros(self.size_pop)

            for ant in range(self.size_pop):
                for dim in range(self.n_dim):
                    probabilities = self.Tau[dim, :] ** self.alpha
                    probabilities /= probabilities.sum()
                    choice = np.random.choice([0, 1], p=probabilities)
                    paths[ant, dim] = choice

                path_costs[ant] = self.func(paths[ant])

            # 更新信息素
            best_cost_index = path_costs.argmin()
            best_path = paths[best_cost_index]
            best_cost = path_costs[best_cost_index]
            self.generation_best_Y.append(deepcopy(best_cost))

            self.Tau *= (1 - self.rho)  # 信息素挥发
            for dim in range(self.n_dim):
                self.Tau[dim, best_path[dim]] += 1 / best_cost  # 信息素增强

            if iter == 0 or best_cost < self.best_cost:
                self.best_cost = best_cost
                self.best_path = best_path

        return self.best_path, self.best_cost


def main(iteration, param_version, output_folder):
    Debug = False
    algorithm = 'ACA'

    start_time = perf_counter()
    sat_info = TEST_SET[hash_function(2)]

    n_dim = sat_info.get_input_len()
    size_pop = 50
    max_iter = 200
    alpha = 1
    beta = 2
    rho = 0.1

    aca = ACABinary(
        func=fitness_func_with_param(sat_info),
        n_dim=n_dim,
        size_pop=size_pop,
        max_iter=max_iter,
        alpha=alpha,
        beta=beta,
        rho=rho
    )
    solution = aca.run()[0]
    end_time = perf_counter()
    run_time = perf_counter() - start_time

    result = {
        'n_dim': n_dim,
        'size_pop': size_pop,
        'max_iter': max_iter,
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'solution': sum(solution),
        'time': run_time
    }
    save_result_to_file(algorithm, iteration, param_version, result, output_folder, aca)

    if Debug:
        print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
        print(f'the number of the solution is {sum(solution)}')
        print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')
        print(f"run time: {end_time - start_time} seconds")
        print(len(aca.generation_best_Y))
        plt.figure(figsize=(10, 5))  # 设置图像大小
        plt.plot(aca.generation_best_Y, marker='o', linestyle='-', color='b')  # 折线图，标记数据点
        plt.title('Data Variation')  # 图像标题
        plt.xlabel('ite')  # x轴标签
        plt.ylabel('fit')  # y轴标签

        # 显示图像
        plt.show()


# if __name__ == '__main__':
#     main()
