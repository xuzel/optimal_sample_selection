from algorithms import *
#
# run_ga(
#     [45, 8, 6, 4, 4],
# ).print_result(True)
#
# run_aca(
#     [45, 8, 6, 4, 4]
# ).print_result(True)
#
# run_asfa(
#     [45, 8, 6, 4, 4]
# ).print_result(True)
#
# run_pso(
#     [45, 8, 6, 4, 4]
# ).print_result(True)
#
# run_sa(
#     [45, 8, 6, 4, 4]
# ).print_result(True)

result = run_ga([45, 8, 6, 4, 4])
result.print_result()
print(result.solution)
