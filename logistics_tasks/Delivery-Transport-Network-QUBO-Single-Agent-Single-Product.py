from sympy import Poly
from sympy import symbols, Symbol, expand, Expr
from math import log2, ceil, floor
from collections import defaultdict
import neal

INF = 1e9

def log_variables_to_number_expression(single_action_variables: tuple[Symbol, ...], max_num: int, max_power: int):
    return sum([single_action_variables[i] * 2 ** i for i in range(max_power)]) + (max_num + 1 - 2 ** max_power) * single_action_variables[-1]

def QUBO_from_hamiltonian_expr(H: Expr) -> defaultdict:
    Q = defaultdict(int)
    H = H.expand()
    for monom, coef in H.as_coefficients_dict().items():
        monom_variables = []
        for var in monom.free_symbols:
            monom_variables.append(str(var))
        monom_variables.sort()
        if len(monom.free_symbols) == 2:
            Q[tuple(monom_variables)] += coef
        elif len(monom.free_symbols) == 1:
            monom_variables.append(monom_variables[0])
            Q[tuple(monom_variables)] += coef
        elif len(monom.free_symbols) == 0:
            continue
        else:
            raise ValueError
    return Q

def path_from_path_variables_values(variables: dict[str, int], q: int, vertex_count: int):
    path = dict()
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "x":
            i, j = map(int, variable_parts[1:])
            if i in path:
                raise ValueError
            path[i] = j
        else:
            continue
    return list([vertex for _, vertex in sorted(path.items())])


def actions_from_actions_variables_values(variables: dict[str, int], q: int, max_power: int, max_num: int):
    actions = [0] * q
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "y":
            i, j = map(int, variable_parts[1:])
            actions[i] += 2 ** j if j < max_power else max_num + 1 - 2 ** max_power
        else:
            continue
    return actions




vertex_count = int(input())
edge_count = int(input())
edges = [[INF]*vertex_count for i in range(vertex_count)]
for i in range(edge_count):
    a, b, w = map(int, input().split())
    edges[a][b] = w
    edges[b][a] = w

vertex_type = list(map(int, input().split()))
requirements = list(map(int, input().split()))
capacity = int(input())
start, finish = map(int, input().split())

#TODO q determination

q = 10
log_capacity = floor(log2(capacity))

'''OUTDATED'''
#
# #variables
# path_variables = [tuple([symbols("x_{}_{}".format(i, j)) for j in range(vertex_count)]) for i in range(q)]
# action_variables = [tuple([symbols("y_{}_{}".format(i, j)) for j in range(log_capacity + 1)]) for i in range(q)]
# auxiliary_load_variables = [tuple([symbols("z_{}_{}".format(i, j)) for j in range(log_capacity + 1)]) for i in range(q)]
#
#
#
# action_num = [log_variables_to_number_expression(action_variables[i], capacity, log_capacity) for i in range(q)]
# load_num = [log_variables_to_number_expression(auxiliary_load_variables[i], capacity, log_capacity) for i in range(q)]
#
# path_const = 1000
# start_finish_const = 1
# req_const = 1
# load_const = 1
# end_no_load_const = 1
#
#
# #hamiltonians
# H_path = path_const * sum([(1 - sum([path_variables[i][j] for j in range(vertex_count)]))**2 for i in range(q)])
# H_start_finish = start_finish_const * ((1 - path_variables[0][start]) ** 2) + (1 - path_variables[-1][finish]) ** 2
# H_load = load_const * sum([(load_num[i] - sum([sum([action_num[j] * vertex_type[k] * path_variables[j][k] for k in range(vertex_count)]) for j in range(i + 1)])) ** 2 for i in range(q)])
# H_req = req_const * sum([(sum([action_num[i] * path_variables[i][j] for i in range(q)]) - requirements[j]) ** 2 for j in range(vertex_count)])
# H_end_no_load = end_no_load_const * sum([sum([action_num[j] * vertex_type[k] * path_variables[j][k] for k in range(vertex_count)]) for j in range(q)])



# a = QUBO_from_hamiltonian_expr(H)

# sampler = neal.SimulatedAnnealingSampler()
# res = sampler.sample_qubo(a).samples()[0]
# print(res)
# print(path_from_path_variables_values(res, q, vertex_count))
# print(actions_from_actions_variables_values(res, q, log_capacity, capacity))

