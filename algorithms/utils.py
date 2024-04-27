from .data_structure import SatInfo
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


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


def save_result_to_file(algorithms_name, iteration, param_version, result, output_folder, algorithm):
    csv_file = os.path.join(output_folder, f'results_{algorithms_name}_{param_version}', f'results.csv')
    print(csv_file)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    fieldnames_dict = {
        'GA': ['gene_size', 'num_generations', 'prob_mut', 'size_pop', 'solution', 'time'],
        'PSO': ['n_dim', 'pop', 'max_iter', 'w', 'c1', 'c2', 'solution', 'time'],
        # 添加更多的算法和对应的fieldnames
    }

    fieldnames = fieldnames_dict[algorithms_name]

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['gene_size', 'num_generations', 'prob_mut', 'size_pop', 'solution', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        # Write the header only once
        if csvfile.tell() == 0:
            writer.writeheader()
    
        writer.writerow(result)

    Y_history = pd.DataFrame(algorithm.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    
    # Create a directory for the plots if it doesn't exist
    plot_dir = os.path.join(output_folder, f'results_{algorithms_name}_{param_version}', 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # Save the figure to the 'plots' directory
    plt.savefig(os.path.join(plot_dir, f'plot_{iteration}.png'))
    plt.close()