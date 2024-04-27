from copy import deepcopy
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from tqdm import tqdm

from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function, save_result_to_file

class AFSA:
    def __init__(self, func, n_dim, size_pop=50, max_iter=300,
                 max_try_num=100, step=1, visual=3,
                 q=0.98, delta=0.5):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num
        self.step = step  # Not used in binary version
        self.visual = visual  # Max number of bits to flip
        self.q = q  # Visual range decay factor
        self.delta = delta  # Crowding factor threshold

        # Initialize population with binary random values
        self.X = np.random.randint(0, 2, size=(self.size_pop, self.n_dim))
        self.Y = np.array([self.func(x) for x in self.X])
        best_idx = self.Y.argmin()
        self.best_x, self.best_y = self.X[best_idx, :], self.Y[best_idx]
        self.history = []

    def move_to_target(self, idx_individual, x_target):
        x = self.X[idx_individual, :]
        x_new = x_target.copy()
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_y:
            self.best_x = x_new.copy()
            self.best_y = self.Y[idx_individual].copy()

    def move(self, idx_individual):
        x_new = self.X[idx_individual, :].copy()
        flip_indices = np.random.choice(self.n_dim, self.visual, replace=False)
        x_new[flip_indices] = 1 - x_new[flip_indices]

        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_y:
            self.best_x = x_new.copy()
            self.best_y = self.Y[idx_individual].copy()

    def prey(self, idx_individual):
        for try_num in range(self.max_try_num):
            x_target = self.X[idx_individual, :].copy()
            flip_indices = np.random.choice(self.n_dim, self.visual, replace=False)
            x_target[flip_indices] = 1 - x_target[flip_indices]
            if self.func(x_target) < self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        distances = np.array([np.sum(np.abs(self.X[idx_individual, :] - x)) for x in self.X])
        idx_in_vision = np.where((distances > 0) & (distances <= self.visual))[0]
        return idx_in_vision

    def swarm(self, idx_individual):
        idx_in_vision = self.find_individual_in_vision(idx_individual)
        if len(idx_in_vision) > 0:
            center = np.mean(self.X[idx_in_vision], axis=0)
            center = np.round(center).astype(int)  # Round and convert to 0 or 1
            center_y = self.func(center)
            if center_y * len(idx_in_vision) < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center)
                return
        self.prey(idx_individual)

    def follow(self, idx_individual):
        idx_in_vision = self.find_individual_in_vision(idx_individual)
        if len(idx_in_vision) > 0:
            best_y = np.min(self.Y[idx_in_vision])
            best_idx = idx_in_vision[np.argmin(self.Y[idx_in_vision])]
            if best_y * len(idx_in_vision) < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, self.X[best_idx])
                return
        self.prey(idx_individual)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for epoch in range(self.max_iter):
            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.follow(idx_individual)
            self.visual = max(1, int(self.visual * self.q))  # Decay visual, at least 1
            self.history.append(deepcopy(self.best_y))
        return self.best_x, self.best_y


def main(iteration, param_version, output_folder):
    Debug = False
    algorithm = 'AFSA'

    start_time = perf_counter()

    for i in tqdm(range(1, 7)):
        sat_info = TEST_SET[hash_function(i)]

        n_dim = sat_info.get_input_len()
        size_pop = 50
        max_iter = 100
        max_try_num = 100
        step = 1
        visual = 3
        q = 0.98
        delta = 0.5

        afsa = AFSA(fitness_func_with_param(sat_info),
                    n_dim=n_dim,
                    size_pop=size_pop,
                    max_iter=max_iter,
                    max_try_num=max_try_num,
                    step=step,
                    visual=visual,
                    q=q,
                    delta=delta)
        solution = afsa.run()
        end_time = perf_counter()
        run_time = perf_counter() - start_time
        solution = [round(x) for x in solution[0]]

        result = {
            'qusetion_num': i,
            'n_dim': n_dim,
            'size_pop': size_pop,
            'max_iter': max_iter,
            'max_try_num': max_try_num,
            'step': step,
            'visual': visual,
            'q': q,
            'delta': delta,
            'solution': sum(solution),
            'time': run_time
        }
        save_result_to_file(algorithm, iteration, param_version, result, output_folder, afsa, qusetion_num=i)

        if Debug:
            # print(solution)
            print(solution[0])
            print(f'the solution is:\n{sat_info.choose_list(solution)}\n{solution}\n')
            print(f'the number of the solution is {sum(solution)}')
            print(f'valid the solution is {sat_info.all_j_subsets_covered(solution)}')

            print(f"run time: {end_time - start_time} seconds")
            plt.plot(afsa.history)
            plt.show()


if __name__ == '__main__':
    main()
