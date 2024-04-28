import typing

from algorithms import *

test_sat = [
    {'parm': [45, 7, 6, 5, 5], 'solution': 6},
    {'parm': [45, 8, 6, 4, 4], 'solution': 8},
    {'parm': [45, 9, 6, 4, 4], 'solution': 12},
    {'parm': [45, 8, 6, 6, 5], 'solution': 4},
    {'parm': [45, 10, 6, 6, 4], 'solution': 4},
    {'parm': [45, 12, 6, 6, 4], 'solution': 8},
    {'parm': [45, 10, 6, 5, 5], 'solution': 50},
    {'parm': [45, 12, 6, 5, 5], 'solution': 132},
    {'parm': [45, 13, 6, 5, 5], 'solution': 245},
    {'parm': [45, 10, 6, 4, 4], 'solution': 37},
    {'parm': [45, 12, 6, 4, 4], 'solution': 42},
    {'parm': [45, 16, 6, 4, 4], 'solution': 162},
    {'parm': [45, 11, 6, 6, 5], 'solution': 26},
    {'parm': [45, 12, 6, 6, 5], 'solution': 42},
    {'parm': [45, 16, 6, 6, 5], 'solution': 280},
    {'parm': [45, 12, 6, 5, 4], 'solution': 18},
    {'parm': [45, 13, 6, 5, 4], 'solution': 28},
    {'parm': [45, 16, 6, 5, 4], 'solution': 65},
    {'parm': [45, 12, 6, 6, 4], 'solution': 6},
    {'parm': [45, 13, 6, 6, 4], 'solution': 12},
    {'parm': [45, 16, 6, 6, 4], 'solution': 38},
]


def test(test_set: typing.List, iter_num: int, save_file_path: str):
    algorithm = [run_ga, run_aca, run_asfa, run_sa, run_pso]



if __name__ == '__main__':
    test(test_sat, 1, 'result')
