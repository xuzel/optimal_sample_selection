from algorithms import *

run_ga(
    [45, 8, 6, 4, 4],
    auto_parm=False,
    max_iter=100
).print_result(True)

run_aca(
    [45, 8, 6, 4, 4]
).print_result(True)

run_asfa(
    [45, 8, 6, 4, 4]
).print_result(True)

run_pso(
    [45, 8, 6, 4, 4]
).print_result(True)

run_sa(
    [45, 8, 6, 4, 4]
).print_result(True)
