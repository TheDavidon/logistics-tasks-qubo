import manim.utils.rate_functions
import dataclasses
from manim import *
import numpy as np

@dataclasses.dataclass
class RenderData:
    points: dict[int, tuple[float, float]]
    cargos: dict[int, int]
    capacity: int
    path: list[int]
    baggage_states: list[tuple[int, ...]]

render_data = None
class CargoPathScene(Scene):
    def add_fixed_in_frame_mobject(self, mobject):
        # Method to fix mobject relative to the screen
        mobject.fixed_in_frame = True
        self.add(mobject)

    def construct(self):
        # Set a light background
        self.camera.background_color = WHITE

        # ------------------------------------------------------------------------------
        # Input data (example)
        # 10 points with arbitrary coordinates.
        points_data = render_data.points
        # Set of cargos: index -> weight
        cargos = render_data.cargos
        capacity = render_data.capacity
        # The route is a sequence of point indices (15 points)
        path = render_data.path
        # For each route vertex, the baggage state is specified (a set of cargo indices)
        baggage_states = render_data.baggage_states
        # Checking the consistency of input data.
        if not (len(path) == len(baggage_states)):
            raise ValueError

        # ------------------------------------------------------------------------------
        # Preparing the area for point visualization (world area)
        all_coords = np.array(list(points_data.values()))
        min_x, max_x = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        min_y, max_y = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        points_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])
        # The area in which the points will be embedded (taking into account the right indentation)
        world_width = 12
        world_height = 7.2
        world_center = np.array([-0.9, 0, 0])
        scale_x = world_width / (max_x - min_x + 1e-5) * 0.8
        scale_y = world_height / (max_y - min_y + 1e-5) * 0.8
        scale_factor = min(scale_x, scale_y)

        # Transform the original coordinates into visualization coordinates.
        def transform_point(pt):
            pt = np.array([pt[0], pt[1], 0])
            return (pt - points_center) * scale_factor + world_center

        # ------------------------------------------------------------------------------
        # Drawing points and their indices
        points_mobs = VGroup()
        for idx, coord in points_data.items():
            transformed_coord = transform_point(coord)
            point_dot = Dot(point=transformed_coord, radius=0.15, color=BLUE)
            point_label = Text(str(idx), font_size=24, color=BLACK).next_to(point_dot, UP, buff=0.2)
            points_mobs.add(point_dot, point_label)
        self.add(points_mobs)

        # Displaying the boundary of the visualization area
        boundary = Rectangle(width=world_width, height=world_height, color=GRAY, stroke_width=2)
        boundary.move_to(world_center)
        self.add(boundary)

        # ------------------------------------------------------------------------------
        # Function to get dynamic textual strings for the baggage state.
        def get_dynamic_texts(baggage_list):
            cargos_list = sorted(baggage_list)
            cargo_str = ", ".join(map(str, cargos_list)) if cargos_list else "Empty"
            total_weight = sum(cargos[c] for c in cargos_list) if cargos_list else 0
            return cargo_str, str(total_weight) + f"/{capacity}"

        # ------------------------------------------------------------------------------
        # Creating static labels for the side panel (fixed relative to the screen)
        # Labels will not change during animation
        cargo_label = Text("Baggage", font_size=24, color=BLACK)
        weight_label = Text("Weight", font_size=24, color=BLACK)
        # Place labels on the right
        cargo_label.to_edge(RIGHT, buff=0.5).shift(UP*1)
        weight_label.next_to(cargo_label, DOWN, buff=1)
        self.add_fixed_in_frame_mobject(cargo_label)
        self.add_fixed_in_frame_mobject(weight_label)

        # Creating dynamic values for the baggage state.
        init_cargo_str, init_weight_str = get_dynamic_texts(baggage_states[0])
        cargo_value = Text(init_cargo_str, font_size=24, color=BLACK)
        weight_value = Text(init_weight_str, font_size=24, color=BLACK)
        # Place dynamic texts under corresponding static labels.
        cargo_value.next_to(cargo_label, DOWN, buff=0.2)
        weight_value.next_to(weight_label, DOWN, buff=0.2)
        cargo_value.fixed_in_frame = True
        weight_value.fixed_in_frame = True
        self.add(cargo_value, weight_value)

        # ------------------------------------------------------------------------------
        # Create a "car" object in the form of a small red circle.
        start_point = transform_point(points_data[path[0]])
        car = Dot(point=start_point, radius=0.20, color=RED)
        self.add(car)

        # Add a tracing line behind the car.
        traced_path = TracedPath(car.get_center, stroke_color=RED, stroke_width=3)
        self.add(traced_path)

        # ------------------------------------------------------------------------------
        # Animation of the car moving along the route and updating the dynamic baggage values.
        for i in range(1, len(path)):
            # Determine the start and end points for the route segment.
            start_idx = path[i - 1]
            end_idx = path[i]
            start_pos = transform_point(points_data[start_idx])
            end_pos = transform_point(points_data[end_idx])
            move_path = Line(start_pos, end_pos)
            # Animation of the car movement along the line.
            self.play(MoveAlongPath(car, move_path), run_time=2, rate_func=manim.utils.rate_functions.smooth)
            # Get the new state of the baggage.
            new_baggage = baggage_states[i]
            new_cargo_str, new_weight_str = get_dynamic_texts(new_baggage)
            # Create new objects for the dynamic values and position them in the same places.
            new_cargo_value = Text(new_cargo_str, font_size=24, color=BLACK)
            new_cargo_value.move_to(cargo_value.get_center())
            new_cargo_value.fixed_in_frame = True

            new_weight_value = Text(new_weight_str, font_size=24, color=BLACK)
            new_weight_value.move_to(weight_value.get_center())
            new_weight_value.fixed_in_frame = True

            # Animate the update of the dynamic values.
            self.play(
                Transform(cargo_value, new_cargo_value),
                Transform(weight_value, new_weight_value),
                run_time=1
            )
            self.wait(0.5)

        self.wait(1)