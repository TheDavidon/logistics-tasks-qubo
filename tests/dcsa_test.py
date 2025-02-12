import random
import math
import time
import matplotlib.pyplot as plt
import timeit
from logistics_tasks.discrete_cargo_single_agent_QUBO import *



class RandomInstanceGenerator:
    def __init__(self, vertex_count, cargo_count, max_weight, capacity):
        self.vertex_count = vertex_count
        self.cargo_count = cargo_count
        self.max_weight = max_weight
        self.capacity = capacity
        self.scale = 100
    def generate_instance(self) -> (FormattedData, list):
        points = [(random.randint(0, self.scale), random.randint(0, self.scale)) for i in range(self.vertex_count)]
        dist = tuple(tuple(math.dist(p_1, p_2) for p_2 in points) for p_1 in points)
        cargo_list = []
        sum_weight = 0
        for i in range(self.cargo_count):
            a = random.randint(0, self.vertex_count - 1)
            b = random.randint(0, self.vertex_count - 1)
            a, b = min(a, b), max(a, b)
            if a == b:
                if b == self.vertex_count - 1:
                    a = b - 1
                else:
                    b = a + 1
            cargo_list.append(Cargo(random.randint(1, self.max_weight), a, b))
            sum_weight += cargo_list[-1].weight
        M = floor(log2(self.capacity))
        q = self.vertex_count * (sum_weight // self.capacity + 1) # ??? just an approximation
        start_vertex, finish_vertex = random.randint(0, self.vertex_count - 1), random.randint(0, self.vertex_count - 1)
        return FormattedData(self.vertex_count, self.capacity, start_vertex, finish_vertex, cargo_list, self.cargo_count, dist, q, M), points



data, points = RandomInstanceGenerator(10, 10, 10, 21).generate_instance()
print(data)
I = HamiltonianInitializer(data)


t_3 = time.time()
h = I.process()
t_4 = time.time()
print("hamiltonian creation: ", t_4 - t_3)
t_5 = time.time()
qubo_data = QUBOInitializer(h).get_QUBO()
t_6 = time.time()
print("matrix creation: ", t_6 - t_5)
t_7 = time.time()
sampler = neal.SimulatedAnnealingSampler()
res = sampler.sample_qubo(qubo_data.QUBO_dict, num_reads = 10, num_sweeps=100000)
t_8 = time.time()
print("annealing: ", t_8 - t_7)
print(res.samples()[0])
print(res.lowest())
print(qubo_data.offset)

solutionQUBO = res.samples()[0]

t_9 = time.time()
a = QUBOSolutionInterpreter(solutionQUBO, data).get_solution()
t_10 = time.time()
print("solution interpretation: ", t_10 - t_9)
print(*data.cargo_list)
print(a.path)

print(a.cargo_actions)



x = [p[0] for p in points]
y = [p[1] for p in points]

path_indices = a.path

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', marker='o', label='Точки')

plt.plot([x[i] for i in path_indices], [y[i] for i in path_indices],
         linestyle='-', color='gray', linewidth=1, alpha=0.7, label='Путь')

for i, (xi, yi) in enumerate(zip(x, y), start=0):
    plt.text(xi, yi, f'{i}', fontsize=12, ha='right', va='bottom', color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
