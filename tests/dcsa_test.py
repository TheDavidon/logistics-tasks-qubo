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

def QUBO_solution(data: FormattedData):
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
    res = sampler.sample_qubo(qubo_data.QUBO_dict, num_reads=10, num_sweeps=100000)
    t_8 = time.time()
    print("annealing: ", t_8 - t_7)

    solutionQUBO = res.samples()[0]
    t_9 = time.time()
    sol = QUBOSolutionInterpreter(solutionQUBO, data).get_solution()
    t_10 = time.time()
    print("solution interpretation: ", t_10 - t_9)

    print("-----------------------------------")

    print("cargos info (weight, start, finish):", *[(cargo.weight, cargo.start_vertex, cargo.finish_vertex) for cargo in data.cargo_list])
    print("path: ", sol.path)
    print("cargo actions (turn taken, turn given): ", sol.cargo_actions)
    print("path length: ", sum([data.dist[sol.path[i]][sol.path[i + 1]] for i in range(len(sol.path) - 1)]))
    print("-----------------------------------")

    return sol


def greedy_solution(data: FormattedData):
    t_1 = time.time()

    cargo_taken = [-1] * data.cargo_count
    cargo_given = [-1] * data.cargo_count

    greedy_path = [data.start_vertex]
    cargos_left = data.cargo_list[:]
    cur_cargos = []
    cur_weight = 0
    while cargos_left or cur_cargos:
        for cargo in cargos_left:
            if cargo.start_vertex == greedy_path[-1] and cargo.weight + cur_weight <= data.capacity:
                cur_cargos.append(cargo)
                cargo_taken[data.cargo_list.index(cargo)] = len(greedy_path) - 1
                cur_weight += cargo.weight
        for cargo in cur_cargos:
            if cargo in cargos_left:
                cargos_left.remove(cargo)
        closest_vertex = 0
        min_dist = 1e9
        if not cur_cargos:
            for cargo in cargos_left:
                if data.dist[greedy_path[-1]][cargo.start_vertex] < min_dist:
                    min_dist = data.dist[greedy_path[-1]][cargo.start_vertex]
                    closest_vertex = cargo.start_vertex
        else:
            for cargo in cur_cargos:
                if data.dist[greedy_path[-1]][cargo.finish_vertex] < min_dist:
                    min_dist = data.dist[greedy_path[-1]][cargo.finish_vertex]
                    closest_vertex = cargo.finish_vertex
        greedy_path.append(closest_vertex)
        delivered_cargos = []
        for cargo in cur_cargos:
            if cargo.finish_vertex == greedy_path[-1]:
                delivered_cargos.append(cargo)
                cargo_given[data.cargo_list.index(cargo)] = len(greedy_path) - 1
                cur_weight -= cargo.weight
        for cargo in delivered_cargos:
            cur_cargos.remove(cargo)
    if greedy_path[-1] != data.finish_vertex:
        greedy_path.append(data.finish_vertex)

    sol = SolutionData(path=greedy_path, cargo_actions=[(cargo_taken[i], cargo_given[i]) for i in range(data.cargo_count)])
    t_2 = time.time()
    print("solution time: ", t_2 - t_1)

    print("-----------------------------------")


    print("cargos info (weight, start, finish):", *[(cargo.weight, cargo.start_vertex, cargo.finish_vertex) for cargo in data.cargo_list])
    print("path: ", sol.path)
    print("cargo actions (turn taken, turn given): ", sol.cargo_actions)
    print("path length: ", sum([data.dist[sol.path[i]][sol.path[i + 1]] for i in range(len(sol.path) - 1)]))
    print("-----------------------------------")

    return sol

def visualize_path(path, points):

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    path_indices = path

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


data, points = RandomInstanceGenerator(5, 8, 15, 21).generate_instance()

sol_qubo = QUBO_solution(data)
visualize_path(sol_qubo.path, points)

sol_greedy = greedy_solution(data)
visualize_path(sol_greedy.path, points)

