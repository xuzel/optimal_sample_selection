import pygad
from data_structure import SatInfo

last_fitness = 0


def fitness_func_with_param(set_info: SatInfo):
    def fitness_func(ga_instance, solution, solution_idx):
        if set_info.all_j_subsets_covered(solution):
            return 1.0 / sum(solution)
        else:
            return 0

    return fitness_func


def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}\n")
    last_fitness = ga_instance.best_solution()[1]


def main():
    sat_info = SatInfo(45, 10, 6, 6, 4)
    num_generations = 2500
    num_genes = sat_info.get_input_len()
    sol_per_pop = int(num_genes * 2)
    num_parents_mating = int(sol_per_pop * 0.1)

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func_with_param(sat_info),
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           on_generation=callback_generation,
                           gene_type=int,
                           gene_space=[0, 1],
                           # parallel_processing=['thread', 100],
                           parent_selection_type='rws',
                           # fitness_batch_size=fitness_batch_size,
                           crossover_type='single_point',
                           mutation_percent_genes=0.2
                           )
    ga_instance.run()

    # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
    ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    print(f"the number of k set of the best solution : {sum(solution)}")

    prediction = sat_info.choose_list(solution)
    print(f"Predicted output based on the best solution : {prediction}")

    if ga_instance.best_solution_generation != -1:
        print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

    # Saving the GA instance.
    filename = 'genetic'  # The filename to which the instance is saved. The name is without extension.
    ga_instance.save(filename=filename)

    # # Loading the saved GA instance.
    # loaded_ga_instance = pygad.load(filename=filename)
    # loaded_ga_instance.plot_fitness()


if __name__ == '__main__':
    main()
