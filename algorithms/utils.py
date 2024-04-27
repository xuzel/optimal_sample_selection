from .data_structure import SatInfo


TEST_SET = [
    SatInfo(45, 7, 6, 5, 5),
    SatInfo(45, 8, 6, 4, 4),
    SatInfo(45, 9, 6, 4, 4),
    SatInfo(45, 8, 6, 6, 5),
    SatInfo(45, 9, 6, 5, 4),
    SatInfo(45, 10, 6, 6, 4),

]


def hash_function(x):
    return x - 1
