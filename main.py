from algorithms import *
#
# run_ga(
#     [45, 8, 6, 4, 4],
# ).print_result(True)



# run_afsa(
#     [45, 8, 6, 4, 4]
# ).print_result(True)
#
# run_pso(
#     [45, 9, 6, 4, 4]
# ).print_result(True)
#
# run_sa(
#     [45, 8, 6, 4, 4]
# ).print_result(True)

# result = run_ga([45, 8, 6, 4, 4], auto_parm=True, greedy_init=True, max_iter=200, greedy_replace_probability=0.005)
# result.print_result(True)
# result.save_fit_func_pic('./test_result', 'test')


result = run_ga(
    [45, 8, 6, 4, 4], custom_arr=range(0, 8)
)
print(result.solution)
output = f"""
the time is {result.run_time}
the number of the solution is {result.solution_num}
the output is:
"""
for x in result.solution:
    output += f"{x}\n"
print(output)

