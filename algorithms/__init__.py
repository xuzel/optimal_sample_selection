# __init__.py

# 导入你的算法
from .ga import main as ga_main
from .aca import main as aca_main
from .pso import main as pso_main
from .sa import main as sa_main
from .afsa import main as afsa_main
from .data_structure import SatInfo, fitness_func_with_param
from .utils import TEST_SET, hash_function

__author__ = '17A'
__version__ = '1.0.0'

__all__ = ['ga_main', 'aca_main', 'pso_main', 'sa_main', 'afsa_main', 'SatInfo', 'fitness_func_with_param', 'TEST_SET', 'hash_function']