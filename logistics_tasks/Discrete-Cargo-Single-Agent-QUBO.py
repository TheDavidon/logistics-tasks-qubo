import dataclasses
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
    dist: tuple[tuple[int, ...], ...]
    q: int
    M: int

@dataclasses.dataclass
class HamiltoniansData:
    objective_hamiltonians: list[Expr]
    constraints_hamiltonians: list[Expr]
    sum_hamiltonian: Expr


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
        self.cargo_take_variables = [tuple([symbols("a_{}_{}".format(i, j)) for j in range(q)]) for i in range(self.data.cargo_count)]
        self.cargo_give_variables = [tuple([symbols("b_{}_{}".format(i, j)) for i in range(self.data.cargo_count)]) for j in range(self.data.q)]

        #auxiliary variables
        self.auxiliary_load_variables = [tuple([symbols("y_{}_{}".format(i, j)) for j in range(M + 1)]) for i in range(q)]

        #auxiliary expressions
        self.load_num = [self.auxiliary_variables_to_number_expression() for i in range(q)]
        self.cargo_status = [[sum([self.cargo_take_variables[j][k] - self.cargo_give_variables[j][k] for k in range(i + 1)]) for j in range(self.data.cargo_count)] for i in range(self.data.q)]
    def initialize_consts(self):
        self.const_correct_end_points = 1
        self.const_correct_path = 1
        self.const_correct_loads = 1
        self.const_correct_take_vertices = 1
        self.const_correct_give_vertices = 1
        self.const_correct_order = 1
        self.const_path_length = 1

    def initialize_hamiltonians(self):
        H_correct_path = self.const_correct_path * sum(
            [(1 - sum([self.path_variables[i][j] for j in range(self.data.vertex_count)])) ** 2 for i in range(q)])
        H_correct_end_points = self.const_correct_end_points * (
                    2 - self.path_variables[0][self.data.start_vertex] - self.path_variables[-1][self.data.finish_vertex]) ** 2
        H_correct_take_vertices = self.const_correct_take_vertices * sum([sum([1 - self.path_variables[j][self.data.cargo_list[i].start_vertex] * self.cargo_take_variables[i][j] for j in range(self.data.q)]) for i in range(self.data.cargo_count)])
        H_correct_give_vertices = self.const_correct_give_vertices * sum([sum([1 - self.path_variables[j][self.data.cargo_list[i].finish_vertex] * self.cargo_give_variables[i][j] for j in range(self.data.q)]) for i in range(self.data.cargo_count)])
        H_correct_order = self.const_correct_order * sum([[self.cargo_status[j][i] * (self.cargo_status[j][i] - 1) for j in range(self.data.q)]for i in range(self.data.cargo_count)])
        H_correct_load = [[[self.data.cargo_list[j].weight * self.cargo_status[i][j] for j in range(self.data.cargo_count)] for i in range(k + 1)] for k in range(q)]
        H_correct_loads = self.const_correct_loads * sum(H_correct_load)

        H_path_length = self.const_path_length * sum([sum([sum([self.path_variables[i][u] * self.path_variables[i + 1][v] * self.data.dist[u][v]
                                                           for i in range(self.data.q - 1)]) for v in range(self.data.vertex_count)]) for u
                                                 in range(self.data.vertex_count)])

        self.constraints_hamiltonians = [H_correct_path, H_correct_end_points, H_correct_take_vertices, H_correct_give_vertices, H_correct_order, H_correct_loads]
        self.objective_hamiltonians = [H_path_length]
    def auxiliary_variables_to_number_expression(self):
        return sum([self.auxiliary_load_variables[i] * 2 ** i for i in range(self.data.M)]) + (self.data.capacity + 1 - 2 ** self.data.M) * self.auxiliary_load_variables[-1]

