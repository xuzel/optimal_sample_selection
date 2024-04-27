import typing
from copy import deepcopy
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from .data_structure import SatInfo, fitness_func_with_param, Result
from .utils import TEST_SET, hash_function



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



def run_aca(sample_parm: typing.List[int],
            size_pop: int = 50,
            max_iter: int = 200,
            alpha: int = 1,
            beta: int = 2,
            rho: float = 0.1):
    start_time = perf_counter()
    sat_info = SatInfo(*sample_parm)
    n_dim = sat_info.get_input_len()
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
    result = Result(
        solution=sat_info.choose_list(solution),
        solution_num=sum(solution),
        algorithm='aca',
        encoder_solution=solution,
        valid=sat_info.all_j_subsets_covered(solution),
        run_time=end_time - start_time,
        y_history=aca.generation_best_Y
    )
    return result


def main():
    start_time = perf_counter()
    sat_info = TEST_SET[hash_function(2)]
    n_dim = sat_info.get_input_len()
    aca = ACABinary(
        func=fitness_func_with_param(sat_info),
        n_dim=n_dim,
        size_pop=50,
        max_iter=200,
        alpha=1,
        beta=2,
        rho=0.1
    )
    solution = aca.run()[0]
    end_time = perf_counter()
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


if __name__ == '__main__':
    # main()
    run_aca(
        [45, 8, 6, 4, 4]
    ).print_result(True)

