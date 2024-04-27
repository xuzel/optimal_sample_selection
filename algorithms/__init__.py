# __init__.py

# 导入你的算法
from .ga import run_ga
from .aca import run_aca
from .pso import run_pso
from .sa import run_sa
from .afsa import run_asfa
from .data_structure import SatInfo, Result, fitness_func_with_param
# from utils import TEST_SET, hash_function

__author__ = '17A'
__version__ = '1.0.0'

__all__ = ['run_ga', 'run_aca', 'run_pso', 'run_sa', 'run_asfa', 'SatInfo', 'fitness_func_with_param']