import dataclasses

import neal
from sympy import symbols, Symbol, expand, Expr
from math import log2, ceil, floor
from collections import defaultdict


@dataclasses.dataclass
class Cargo:
    weight: int
    start_vertex: int
    finish_vertex: int
@dataclasses.dataclass
class FormattedData:
    vertex_count: int
    capacity: int
    start_vertex: int
    finish_vertex: int
    cargo_list: list[Cargo]
    cargo_count: int
    dist: tuple[tuple[float, ...], ...]
    q: int
    M: int

@dataclasses.dataclass
class HamiltoniansData:
    objective_hamiltonians: list[Expr]
    constraints_hamiltonians: list[Expr]
    sum_hamiltonian: Expr
@dataclasses.dataclass
class QUBOData:
    QUBO_dict: defaultdict[tuple[str, str], int]
    offset: int

@dataclasses.dataclass
class SolutionData:
    path: list[int]
    cargo_actions: list[tuple[int, int]]

class HamiltonianInitializer:
    def __init__(self, formatted_data):
        self.data: FormattedData = formatted_data
    def process(self):
        self.initialize_variables()
        self.initialize_consts()
        self.initialize_hamiltonians()
        return HamiltoniansData(self.objective_hamiltonians, self.constraints_hamiltonians, sum(self.objective_hamiltonians) + sum(self.constraints_hamiltonians))

    def initialize_variables(self):
        # main variables
        self.path_variables = [tuple([symbols("x_{}_{}".format(i, j)) for j in range(self.data.vertex_count)]) for i in range(self.data.q)]
        self.cargo_take_variables = [tuple([symbols("a_{}_{}".format(i, j)) for j in range(self.data.q)]) for i in range(self.data.cargo_count)]
        self.cargo_give_variables = [tuple([symbols("b_{}_{}".format(i, j)) for j in range(self.data.q)]) for i in range(self.data.cargo_count)]

        #auxiliary variables
        self.auxiliary_load_variables = [tuple([symbols("y_{}_{}".format(i, j)) for j in range(self.data.M + 1)]) for i in range(self.data.q)]

        #auxiliary expressions
        self.load_num = [self.auxiliary_variables_to_number_expression(i) for i in range(self.data.q)]
        self.cargo_status = [[sum([self.cargo_take_variables[j][k] - self.cargo_give_variables[j][k] for k in range(i + 1)]) for j in range(self.data.cargo_count)] for i in range(self.data.q)]
    def initialize_consts(self):
        self.norm_const_correct_end_points = 1 / 2
        self.norm_const_correct_path = 1
        self.norm_const_correct_loads = 1 / self.data.q
        self.norm_const_correct_take_give_action = 1
        self.norm_const_correct_take_vertices = 1
        self.norm_const_correct_give_vertices = 1
        self.norm_const_correct_order = 1 / 2
        self.norm_const_path_length = 1 / max([max(d) for d in self.data.dist])

        self.const_correct_end_points = 10 * self.norm_const_correct_end_points
        self.const_correct_path = 100 * self.norm_const_correct_path
        self.const_correct_loads = 1 * self.norm_const_correct_loads
        self.const_correct_take_give_action = 50 * self.norm_const_correct_take_give_action
        self.const_correct_take_vertices = 100 * self.norm_const_correct_take_vertices
        self.const_correct_give_vertices = 100 * self.norm_const_correct_give_vertices
        self.const_correct_order = 200 * self.norm_const_correct_order
        self.const_path_length = 1 * self.norm_const_path_length

    def initialize_hamiltonians(self):
        H_correct_path = self.const_correct_path * sum(
            [(1 - sum([self.path_variables[i][j] for j in range(self.data.vertex_count)])) ** 2 for i in range(self.data.q)])
        H_correct_end_points = self.const_correct_end_points * (
                    2 - self.path_variables[0][self.data.start_vertex] - self.path_variables[-1][self.data.finish_vertex]) ** 2
        H_correct_take_action = self.const_correct_take_give_action * sum([(1 - sum([self.cargo_take_variables[i][j] for j in range(self.data.q)])) ** 2 for i in range(self.data.cargo_count)])
        H_correct_give_action = self.const_correct_take_give_action * sum([(1 - sum([self.cargo_give_variables[i][j] for j in range(self.data.q)])) ** 2 for i in range(self.data.cargo_count)])
        H_correct_take_vertices = self.const_correct_take_vertices * sum([sum([(1 - self.path_variables[j][self.data.cargo_list[i].start_vertex]) * self.cargo_take_variables[i][j] for j in range(self.data.q)]) for i in range(self.data.cargo_count)])
        H_correct_give_vertices = self.const_correct_give_vertices * sum([sum([(1 - self.path_variables[j][self.data.cargo_list[i].finish_vertex]) * self.cargo_give_variables[i][j] for j in range(self.data.q)]) for i in range(self.data.cargo_count)])
        H_correct_order = self.const_correct_order * sum([sum([self.cargo_status[j][i] * (self.cargo_status[j][i] - 1) for j in range(self.data.q)]) for i in range(self.data.cargo_count)])
        H_correct_load = [(self.load_num[i] - sum([self.data.cargo_list[j].weight * self.cargo_status[i][j] for j in range(self.data.cargo_count)])) ** 2 for i in range(self.data.q)]
        H_correct_loads = self.const_correct_loads * sum(H_correct_load)

        H_path_length = self.const_path_length * sum([sum([sum([self.path_variables[i][u] * self.path_variables[i + 1][v] * self.data.dist[u][v]
                                                           for i in range(self.data.q - 1)]) for v in range(self.data.vertex_count)]) for u
                                                 in range(self.data.vertex_count)])

        self.constraints_hamiltonians = [H_correct_path, H_correct_end_points, H_correct_take_action, H_correct_give_action, H_correct_take_vertices, H_correct_give_vertices, H_correct_order, H_correct_loads]
        self.objective_hamiltonians = [H_path_length]
    def auxiliary_variables_to_number_expression(self, k):
        return sum([self.auxiliary_load_variables[k][i] * 2 ** i for i in range(self.data.M)]) + (self.data.capacity + 1 - 2 ** self.data.M) * self.auxiliary_load_variables[k][-1]

class QUBOInitializer:
    def __init__(self, hamiltonians_data):
        self.hamiltonians_data: HamiltoniansData = hamiltonians_data
    def get_QUBO(self) -> QUBOData:
        Q = defaultdict(int)
        H = self.hamiltonians_data.sum_hamiltonian.expand()
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
        return QUBOData(Q, offset)


class QUBOSolutionInterpreter:
    def __init__(self, variable_value, formatted_data):
        self.variable_value = variable_value
        self.formatted_data = formatted_data
    def get_solution(self):
        self.validate_solution()
        return SolutionData(path=self.path, cargo_actions=self.cargo_actions)
    def validate_solution(self):
        self.validate_path()
        self.validate_cargo()
        self.validate_load()

    def validate_path(self):
        self.path = []
        for i in range(self.formatted_data.q):
            visited_vertices = []
            for j in range(self.formatted_data.vertex_count):
                if self.variable_value[f"x_{i}_{j}"]:
                    visited_vertices.append(j)
            if len(visited_vertices) != 1:
                print(f"Not single vertex at {i} turn : ", *visited_vertices)
                self.path.append(tuple(visited_vertices))
            else:
                self.path.append(visited_vertices[0])
    def validate_cargo(self):
        self.cargo_actions = []
        for i in range(self.formatted_data.cargo_count):
            start_turn, finish_turn = [], []
            for j in range(self.formatted_data.q):
                if self.variable_value[f"a_{i}_{j}"]:
                    start_turn.append(j)
                if self.variable_value[f"b_{i}_{j}"]:
                    finish_turn.append(j)
            if len(start_turn) != 1:
                print(f"Not single turn for {i} cargo take: ", *start_turn)
                start_turn = tuple(start_turn)
            else:
                start_turn = start_turn[0]
                if self.path[start_turn] != self.formatted_data.cargo_list[i].start_vertex:
                    print(f"Cargo {i} taken at wrong vertex {self.path[start_turn]} instead of {self.formatted_data.cargo_list[i].start_vertex}")
            if len(finish_turn) != 1:
                print(f"Not single turn for {i} cargo give: ", *finish_turn)
                finish_turn = tuple(finish_turn)
            else:
                finish_turn = finish_turn[0]
                if self.path[finish_turn] != self.formatted_data.cargo_list[i].finish_vertex:
                    print(f"Cargo {i} given at wrong vertex {self.path[finish_turn]} instead of {self.formatted_data.cargo_list[i].finish_vertex}")
            self.cargo_actions.append((start_turn, finish_turn))
            
            
    def validate_load(self):
        actual_load = 0
        for i in range(self.formatted_data.q):
            for j in range(self.formatted_data.cargo_count):
                if self.variable_value[f"a_{j}_{i}"]:
                    actual_load += self.formatted_data.cargo_list[j].weight
                if self.variable_value[f"b_{j}_{i}"]:
                    actual_load -= self.formatted_data.cargo_list[j].weight
            auxiliary_variable_load = sum([self.variable_value[f"y_{i}_{k}"] * 2 ** k for k in range(self.formatted_data.M)]) + (self.formatted_data.capacity + 1 - 2 ** self.formatted_data.M) * self.variable_value[f"y_{i}_{self.formatted_data.M}"]
            if actual_load != auxiliary_variable_load:
                print(f"At turn {i} actual load not equal to auxiliary load: {actual_load} != {auxiliary_variable_load}")


