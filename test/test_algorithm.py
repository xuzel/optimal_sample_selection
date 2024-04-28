import json
import os.path
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


def test(test_set: typing.List[typing.Dict], iter_num: int, save_file_path: str):
    result_dict = {
        'ga': list(),
        'aca': list(),
        'afsa': list(),
        'sa': list(),
        'pso': list()
    }
    algorithm = [run_ga, run_aca, run_afsa, run_sa, run_pso]
    # algorithm = [run_aca, run_afsa]
    for i in range(iter_num):
        for each_data in test_set:
            for each_alg in algorithm:
                result = each_alg(each_data['parm'])
                print(f"iter: {i}")
                result.print_result(True)
                result.save_fit_func_pic(file_path=os.path.join(save_file_path, result.algorithm),
                                         file_name=f'iter_{i}')
                result_dict[result.algorithm].append({
                    'iter': i,
                    # 'solution': result.solution,
                    'run_time': result.run_time,
                    'alg_solution_num': result.solution_num,
                    'parm': each_data['parm'],
                    'solution_num': each_data['solution']
                })

    return result_dict


if __name__ == '__main__':
    out_put = test(test_sat, 5, './test/result')
    json_output = json.dumps(out_put, indent=4)
    with open('result.json', 'w') as file:
        file.write(json_output)
