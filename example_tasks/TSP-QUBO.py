import neal
import numpy as np
import itertools
from sympy import Poly
from sympy import symbols
from collections import defaultdict

def bin_var_to_cycle_vertices(bin_values: list[int], vertex_count: int) -> list[int]:
    cycle = []
    for i in range(vertex_count):
        cycle.append(bin_values[i * n: (i + 1) * n].index(1))
    return cycle

def cycle_weight(edges: list[list[int]], cycle: list[int]):
    return sum([edges[cycle[i]][cycle[(i + 1) % len(edges)]] for i in range(len(edges))])

n = int(input())
edge_count = int(input())

edges = [[0] * n for i in range(n)]
for i in range(edge_count):
    a, b, w = map(int, input().split())
    edges[a][b] = w

max_weight = max([max(edges[i]) for i in range(n)])

B = 1
A = B * max_weight + 1


list_of_symbols = [[symbols("x_{}_{}".format(i, j)) for i in range(n)] for j in range(n)]
variables_count = n * n


'''Hamiltonian cycle Hamiltonians'''
H_1 = Poly(A * sum([(1 - sum([list_of_symbols[i][j] for j in range(n)]))**2 for i in range(n)]))
H_2 = Poly(A * sum([(1 - sum([list_of_symbols[i][j] for i in range(n)]))**2 for j in range(n)]))
H_A = H_1 + H_2

for u in range(n):
    for v in range(n):
        if edges[u][v] == 0:
            H_A += sum([list_of_symbols[u][i] * list_of_symbols[v][(i + 1) % n] for i in range(n)])



'''Cycle weight Hamiltonian'''
H_B = Poly(sum([list_of_symbols[i][j] for i in range(n) for j in range(n)])) * 0
for u in range(n):
    for v in range(n):
        H_B += edges[u][v] * sum([list_of_symbols[u][i] * list_of_symbols[v][(i + 1) % n] for i in range(n)])

H = H_A + H_B

I = defaultdict(int)
flag = 0
for powers, k in H.as_dict().items():
    flag = 0
    for i in range(n * n):
        if powers[i] == 2:
            I[(i, i)] += k
            break
        elif powers[i] == 1:
            for j in range(i + 1, n * n):
                if powers[j] == 1:
                    I[(i, j)] += k
                    flag = 1
                    break
            if flag == 0:
                I[(i, i)] += k
            break


sampler = neal.SimulatedAnnealingSampler()
samp = sampler.sample_qubo(I)
ans = list(samp.first.sample.values())
print("Cycle: " + str(bin_var_to_cycle_vertices(ans, n)))
print("Weight: {}".format(cycle_weight(edges, bin_var_to_cycle_vertices(ans, n))))


'''brutefoce solution'''

# M = np.zeros((variables_count, variables_count))
# for pair, val in I.items():
#     M[pair[0]][pair[1]] = val
#
# it = itertools.product([0, 1], repeat=variables_count)
# min_energy = 1e9
# min_energy_values = ()
#
# for values in it:
#     values_vector = np.array(values).reshape(1, variables_count)
#     energy = (values_vector @ M @ values_vector.T)[0][0]
#     if min_energy >= energy:
#         min_energy = energy
#         min_energy_values = values
# 
# print(min_energy)
# print(min_energy_values)
# print(bin_var_to_cycle_vertices(min_energy_values, n))