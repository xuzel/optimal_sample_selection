import typing
import time
import pulp
from data_structure import SatInfo, Result


def run_lp(sample_parm: typing.List[int], **kwargs):
    sat_info = SatInfo(*sample_parm, **kwargs)
    union_set, all_set = sat_info.calculate_only_set()
    print(len(all_set))
    subsets = dict()
    for index, element in enumerate(all_set):
        subsets[f'K{index}'] = element
    prob = pulp.LpProblem("SetCover", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", union_set, cat=pulp.LpBinary)
    prob += pulp.lpSum(x[i] for i in union_set)
    for k, elements in subsets.items():
        prob += pulp.lpSum(x[i] for i in elements) >= 1, f"Cover_{k}"
    prob.solve()
    print("选择的元素集合S为：", {i for i in union_set if x[i].varValue == 1})


if __name__ == '__main__':
    run_lp([45, 12, 6, 6, 4])
