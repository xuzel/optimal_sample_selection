import os
import csv
from algorithms import *

class AlgorithmTester:
    def __init__(self, algorithms, param_versions, output_folder='results'):
        self.algorithms = algorithms
        self.param_versions = param_versions
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def test_algorithms(self, num_iterations):
        for algorithm in self.algorithms:
            for param_version in self.param_versions:
                for iteration in range(num_iterations):
                    algorithm(iteration + 1, param_version, self.output_folder)

def test_algorithm(algorithm_main, iteration, param_version, output_folder):
    algorithm_main(iteration, param_version)

if __name__ == '__main__':
    # algorithms = [ga_main, pso_main, sa_main, aca_main, afsa_main]
    algorithms = [ga_main]
    param_versions = [1]  # Change this for different parameter versions
    tester = AlgorithmTester(algorithms, param_versions)
    num_iterations = 5  # Number of iterations
    tester.test_algorithms(num_iterations)
