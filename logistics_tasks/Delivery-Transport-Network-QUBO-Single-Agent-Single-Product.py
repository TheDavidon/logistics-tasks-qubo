import random

from sympy import symbols, Symbol, expand, Expr
from math import log2, ceil, floor
from collections import defaultdict
import neal

INF = 1e9


#TODO Refactor code with classes usage

def random_network_generator(consumers_count: int, storage_count: int):
    n = consumers_count + storage_count
    coords = [(random.randint(-100, 100), random.randint(-100, 100)) for i in range(n)]
    vertex_types = [-1 if i < consumers_count else 1 for i in range(n)]
    dist = [[((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** (0.5) for j in range(n)] for i in range(n)]
    requirements = [random.randint(1, 100) for i in range(n)]
    cons_req = sum(requirements[:consumers_count])
    stor_req = sum(requirements[consumers_count:n])
    while cons_req > stor_req:
        t = random.randint(consumers_count, n - 1)
        a = random.randint(1, 50)
        requirements[t] += a
        stor_req += a
    capacity = random.randint(5, 50)
    start, finish = (random.randint(0, n - 1), random.randint(0, n - 1))
    return (n, coords, dist, vertex_types, requirements, capacity, start, finish)

def encoding_variables_to_number_expression(encoding_variables: tuple[Symbol, ...], max_num: int, max_power: int):
    return sum([encoding_variables[i] * 2 ** i for i in range(max_power)]) + (max_num + 1 - 2 ** max_power) * encoding_variables[-1]

def QUBO_from_hamiltonian_expr(H: Expr) -> tuple[defaultdict, int]:
    Q = defaultdict(int)
    H = H.expand()
    offset = 0
    for monom, coef in H.as_coefficients_dict().items():
        if monom == 1:
            offset = coef
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
    return (Q, offset)

def get_energy_offset(H: Expr):
    return H.as_coefficients_dict()['1']

def path_from_path_variables_values(variables: dict[str, int], q: int, vertex_count: int):
    path = [-1] * q
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "x":
            i, j = map(int, variable_parts[1:])
            if path[i] != -1:
                print("INCORRECT PATH, TWO VERTICES AT TURN: {} and {}".format(path[i], j))
                continue
            path[i] = j
        else:
            continue
    for i in range(vertex_count):
        if path[i] == -1:
            print("INCORRECT PATH, NO VERTICES AT TURN: {}".format(i))
    return path


def actions_from_actions_variables_values(variables: dict[str, int], q: int, max_power: int, max_num: int, path: list[int]):
    actions = [0] * q
    vertex_of_action = path
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "y":
            i, j, k = map(int, variable_parts[1:])
            if vertex_of_action[i] != j:
                print("INCORRECT ACTION AT TURN {}, PATH VERTEX AND ACTION VERTEX MISMATCH: {} and {}".format(i, vertex_of_action[i],  j))
                continue
            actions[i] += 2 ** k if k < max_power else max_num + 1 - 2 ** max_power
        else:
            continue
    return actions

def loads_from_loads_variables_values(variables: dict[str, int], q: int, max_power: int, max_num: int):
    loads = [0] * q
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "z":
            i, j = map(int, variable_parts[1:])
            loads[i] += 2 ** j if j < max_power else max_num + 1 - 2 ** max_power
        else:
            continue
    return loads

def vertex_action_from_vertex_action_variables_values(variables: dict[str, int], n: int, max_power: list[int], max_num: list[int]):
    vertex_actions = [0] * n
    for variable, value in variables.items():
        if value == 0:
            continue
        variable_parts = variable.split("_")
        if variable_parts[0] == "w":
            i, j = map(int, variable_parts[1:])
            vertex_actions[i] += 2 ** j if j < max_power[i] else max_num[i] + 1 - 2 ** max_power[i]
        else:
            continue
    return vertex_actions


def validate_loads_actions(loads: list[int], actions: list[int], path: list[int], vertex_types: list[int]) -> None:
    net_action = 0
    for i in range(len(loads)):
        net_action += actions[i] * vertex_types[path[i]]
        if net_action != loads[i]:
            print("MISMATCH OF LOAD AND NET ACTION AT TURN {}, LOAD: {}, NET ACTION: {}".format(i, loads[i], net_action))


def validate_actions_at_vertex(vertex_actions: list[int], actions: list[int], path: list[int], n: int, vertex_types: list[int]) -> None:
    net_vertex_action = [0] * n
    for i in range(len(actions)):
        net_vertex_action[path[i]] += actions[i] * vertex_types[path[i]]
    for i in range(n):
        if vertex_types[i] == 1 and net_vertex_action[i] != vertex_actions[i]:
            print("MISMATCH OF VERTEX ACTION AND NET VERTEX ACTION AT VERTEX {}, VERTEX ACTION: {}, NET ACTION: {}".format(i, vertex_actions[i], net_vertex_action[i]))

        if vertex_types[i] == -1:
            vertex_actions[i] = net_vertex_action[i]

def output_data(path, actions, loads):
    pass


max_d = 0

#random input

# vertex_count, coords, dist, vertex_type, requirements, capacity, start, finish = random_network_generator(2, 2)
# print(requirements)
# print(coords)
# print(start, finish)
# print(capacity)
# max_d = max([max(dist[i]) for i in range(vertex_count)])



#user input

vertex_count = int(input())
edge_count = int(input())
dist = [[0]*vertex_count for i in range(vertex_count)]
for i in range(edge_count):
    a, b, w = map(int, input().split())
    max_d = max(max_d, w)
    dist[a][b] = w
    dist[b][a] = w

vertex_type = list(map(int, input().split()))
requirements = list(map(int, input().split()))
capacity = int(input())
start, finish = map(int, input().split())



#TODO q determination

q = 15
M = floor(log2(capacity))
U = [floor(log2(req)) for req in requirements]

#main variables
path_variables = [tuple([symbols("x_{}_{}".format(i, j)) for j in range(vertex_count)]) for i in range(q)]
action_variables = [[tuple([symbols("y_{}_{}_{}".format(i, j, k)) for k in range(M + 1)]) for j in range(vertex_count)] for i in range(q)]
#auxiliary variables
auxiliary_load_variables = [tuple([symbols("z_{}_{}".format(i, j)) for j in range(M + 1)]) for i in range(q)]
auxiliary_vertex_action_variables = [tuple([symbols("w_{}_{}".format(i, j)) for j in range(U[i] + 1)]) for i in range(vertex_count)]


action_num = [[encoding_variables_to_number_expression(action_variables[i][j], capacity, M) for j in range(vertex_count)] for i in range(q)]
load_num = [encoding_variables_to_number_expression(auxiliary_load_variables[i], capacity, M) for i in range(q)]
vertex_action_num = [encoding_variables_to_number_expression(auxiliary_vertex_action_variables[i], requirements[i], U[i]) for i in range(vertex_count)]

#TODO hamiltonians const determination
const_correct_actions = ceil(2 * capacity * (q + 1) ** 3 / 3)
const_correct_end_points = ceil(const_correct_actions * capacity / 3) + 1
const_correct_path = max(max_d, 3 * const_correct_end_points + 1)
const_correct_loads = 200 * q
const_correct_storage_actions = 10 * q
const_satisfied_consumers = 10 * q
const_path_length = ceil((q * capacity) ** 2 / max_d)

#correctness hamiltonians
H_correct_path = const_correct_path * sum([(1 - sum([path_variables[i][j] for j in range(vertex_count)]))**2 for i in range(q)])
H_correct_end_points = const_correct_end_points * (2 - path_variables[0][start] - path_variables[-1][finish]) ** 2
H_correct_actions = const_correct_actions * sum([sum([action_num[i][j] * (1 - path_variables[i][j]) for j in range(vertex_count)]) for i in range(q)])
H_correct_loads = const_correct_loads * (sum([sum([load_num[i] - sum([sum([action_num[k][j] * vertex_type[j] for j in range(vertex_count)]) for k in range(i + 1)])]) ** 2 for i in range(q - 1)]) +
                   sum([sum([action_num[k][j] * vertex_type[j] for j in range(vertex_count)]) for k in range(q)]) ** 2)
H_correct_storage_actions = const_correct_storage_actions * sum([(vertex_action_num[j] - sum([action_num[i][j] for i in range(q)])) ** 2 for j in range(vertex_count) if vertex_type[j] == 1])

#requirements satisfaction hamiltonians
H_satisfied_consumers = const_satisfied_consumers * sum([(requirements[j] - sum([action_num[i][j] for i in range(q)])) ** 2 for j in range(vertex_count) if vertex_type[j] == -1])

#objective function hamiltonians
H_path_length = const_path_length * sum([sum([sum([path_variables[i][u] * path_variables[i + 1][v] * dist[u][v] for i in range(q - 1)]) for v in range(vertex_count)]) for u in range(vertex_count)])

H = H_correct_path + H_correct_end_points + H_correct_actions + H_correct_loads + H_satisfied_consumers + H_path_length + H_correct_storage_actions
a, offset = QUBO_from_hamiltonian_expr(H)


sampler = neal.SimulatedAnnealingSampler()
res = sampler.sample_qubo(a, num_reads = 1, num_sweeps=1000000).samples()[0]
print(res)
path = path_from_path_variables_values(res, q, vertex_count)
print(path)
actions = actions_from_actions_variables_values(res, q, M, capacity, path)
print(actions)
loads = loads_from_loads_variables_values(res, q, M, capacity)
print(loads)

vertex_actions = vertex_action_from_vertex_action_variables_values(res, vertex_count, U, requirements)

validate_loads_actions(loads, actions, path, vertex_type)

validate_actions_at_vertex(vertex_actions, actions, path, vertex_count, vertex_type)

print(*vertex_actions)


print("NET ACTION:")
for i in range(vertex_count):
    print("Vertex {}, net action = {}".format(i, vertex_actions[i]))



