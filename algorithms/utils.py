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


def save_result_to_file(algorithms_name, iteration, param_version, result, output_folder, algorithm, qusetion_num):
    csv_file = os.path.join(output_folder, f'results_{algorithms_name}_{param_version}', f'results_q{qusetion_num}.csv')
    print(csv_file)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    fieldnames_dict = {
        'GA': ['qusetion_num', 'gene_size', 'num_generations', 'prob_mut', 'size_pop', 'solution', 'time'],
        'PSO': ['qusetion_num', 'n_dim', 'pop', 'max_iter', 'w', 'c1', 'c2', 'solution', 'time'],
        'SA': ['qusetion_num', 'T_max', 'T_min', 'L', 'max_stay_counter', 'solution', 'time'],
        'ACA': ['qusetion_num', 'n_dim', 'size_pop', 'max_iter', 'alpha', 'beta', 'rho', 'solution', 'time'],
        'AFSA': ['qusetion_num','n_dim', 'size_pop','max_iter', 'max_try_num','step','visual','q','delta','solution','time']
    }

    fieldnames = fieldnames_dict[algorithms_name]

    with open(csv_file, 'a', newline='') as csvfile:
        # fieldnames = ['gene_size', 'num_generations', 'prob_mut', 'size_pop', 'solution', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        # Write the header only once
        if csvfile.tell() == 0:
            writer.writeheader()
    
        writer.writerow(result)

    plot_save(algorithms_name, output_folder, iteration, param_version, algorithm, qusetion_num)

def plot_save(algorithms_name, output_folder, iteration, param_version, algorithm, qusetion_num):

    if algorithms_name == 'GA':
        Y_history = pd.DataFrame(algorithm.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        # plt.show()
        # plt.close()
    elif algorithms_name == 'PSO':
        plt.plot(algorithm.gbest_y_hist)
        # plt.show()
        # plt.close()
    elif algorithms_name == 'SA':
        plt.plot(pd.DataFrame(algorithm.best_y_history).cummin(axis=0))
        # plt.show()
        # plt.close()
    elif algorithms_name == 'ACA':
        plt.figure(figsize=(10, 5))  # 设置图像大小
        plt.plot(algorithm.generation_best_Y, marker='o', linestyle='-', color='b')  # 折线图，标记数据点
        plt.title('Data Variation')  # 图像标题
        plt.xlabel('ite')  # x轴标签
        plt.ylabel('fit')  # y轴标签
        # plt.show()
        # plt.close()
        
    # Create a directory for the plots if it doesn't exist
    plot_dir = os.path.join(output_folder, f'results_{algorithms_name}_{param_version}', f'plots_q{qusetion_num}')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # Save the figure to the 'plots' directory
    plt.savefig(os.path.join(plot_dir, f'plot_{iteration}.png'))