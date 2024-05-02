from data_structure import SatInfo
import numpy as np


def replace_rows_randomly(original_array, replace_probability, replacement_array):
    num_rows = original_array.shape[0]
    rows_to_replace = int(np.round(num_rows * replace_probability))
    rows_to_replace = max(1, rows_to_replace)
    rows_indices = np.random.choice(num_rows, size=rows_to_replace, replace=False)
    for row_index in rows_indices:
        original_array[row_index, :] = replacement_array
    return original_array


TEST_SET = [
    # SatInfo(45, 7, 6, 5, 5),
    # SatInfo(45, 8, 6, 4, 4),
    # SatInfo(45, 9, 6, 4, 4),
    # SatInfo(45, 8, 6, 6, 5),
    # SatInfo(45, 9, 6, 5, 4),
    # SatInfo(45, 10, 6, 6, 4),

]


def hash_function(x):
    return x - 1
