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
        # Метод для фиксации mobject относительно экрана
        mobject.fixed_in_frame = True
        self.add(mobject)

    def construct(self):
        # Устанавливаем светлый фон
        self.camera.background_color = WHITE

        # ------------------------------------------------------------------------------
        # Входные данные (пример)
        # 10 точек с произвольными координатами.
        points_data = render_data.points
        # Набор грузов: индекс -> вес
        cargos = render_data.cargos
        capacity = render_data.capacity
        # Маршрут – последовательность индексов точек (15 пунктов)
        path = render_data.path
        # Для каждой вершины маршрута задано состояние багажа (множество индексов грузов)
        baggage_states = render_data.baggage_states
        # Проверка согласованности входных данных.
        if not (len(path) == len(baggage_states)):
            raise ValueError

        # ------------------------------------------------------------------------------
        # Подготовка области для визуализации точек (мировая область)
        all_coords = np.array(list(points_data.values()))
        min_x, max_x = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        min_y, max_y = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        points_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])
        # Область, в которую будут вписаны точки (учитываем отступ справа)
        world_width = 12
        world_height = 7.2
        world_center = np.array([-0.9, 0, 0])
        scale_x = world_width / (max_x - min_x + 1e-5) * 0.8
        scale_y = world_height / (max_y - min_y + 1e-5) * 0.8
        scale_factor = min(scale_x, scale_y)

        # Преобразование исходных координат в координаты для визуализации.
        def transform_point(pt):
            pt = np.array([pt[0], pt[1], 0])
            return (pt - points_center) * scale_factor + world_center

        # ------------------------------------------------------------------------------
        # Отрисовка точек и их индексов
        points_mobs = VGroup()
        for idx, coord in points_data.items():
            transformed_coord = transform_point(coord)
            point_dot = Dot(point=transformed_coord, radius=0.15, color=BLUE)
            point_label = Text(str(idx), font_size=24, color=BLACK).next_to(point_dot, UP, buff=0.2)
            points_mobs.add(point_dot, point_label)
        self.add(points_mobs)

        # Отображение границы области визуализации
        boundary = Rectangle(width=world_width, height=world_height, color=GRAY, stroke_width=2)
        boundary.move_to(world_center)
        self.add(boundary)

        # ------------------------------------------------------------------------------
        # Функция для получения динамических текстовых строк для состояния багажа.
        def get_dynamic_texts(baggage_list):
            cargos_list = sorted(baggage_list)
            cargo_str = ", ".join(map(str, cargos_list)) if cargos_list else "Пусто"
            total_weight = sum(cargos[c] for c in cargos_list) if cargos_list else 0
            return cargo_str, str(total_weight) + f"/{capacity}"

        # ------------------------------------------------------------------------------
        # Создание статических надписей для боковой панели (фиксация относительно экрана)
        # Надписи не будут меняться в ходе анимации.
        cargo_label = Text("Багаж", font_size=24, color=BLACK)
        weight_label = Text("Вес", font_size=24, color=BLACK)
        # Размещаем надписи справа
        cargo_label.to_edge(RIGHT, buff=0.5).shift(UP*1)
        weight_label.next_to(cargo_label, DOWN, buff=1)
        self.add_fixed_in_frame_mobject(cargo_label)
        self.add_fixed_in_frame_mobject(weight_label)

        # Создание динамических значений для состояния багажа.
        init_cargo_str, init_weight_str = get_dynamic_texts(baggage_states[0])
        cargo_value = Text(init_cargo_str, font_size=24, color=BLACK)
        weight_value = Text(init_weight_str, font_size=24, color=BLACK)
        # Располагаем динамичные тексты под соответствующими статическими надписями.
        cargo_value.next_to(cargo_label, DOWN, buff=0.2)
        weight_value.next_to(weight_label, DOWN, buff=0.2)
        cargo_value.fixed_in_frame = True
        weight_value.fixed_in_frame = True
        self.add(cargo_value, weight_value)

        # ------------------------------------------------------------------------------
        # Создаем объект "машина" в виде небольшого красного круга.
        start_point = transform_point(points_data[path[0]])
        car = Dot(point=start_point, radius=0.20, color=RED)
        self.add(car)

        # Добавляем трассирующую линию за машиной.
        traced_path = TracedPath(car.get_center, stroke_color=RED, stroke_width=3)
        self.add(traced_path)

        # ------------------------------------------------------------------------------
        # Анимация перемещения машины по маршруту и обновление динамических значений багажа.
        for i in range(1, len(path)):
            # Определяем начальные и конечные точки для сегмента маршрута.
            start_idx = path[i - 1]
            end_idx = path[i]
            start_pos = transform_point(points_data[start_idx])
            end_pos = transform_point(points_data[end_idx])
            move_path = Line(start_pos, end_pos)
            # Анимация движения машины вдоль линии.
            self.play(MoveAlongPath(car, move_path), run_time=2, rate_func=manim.utils.rate_functions.smooth)
            # Получаем новое состояние багажа.
            new_baggage = baggage_states[i]
            new_cargo_str, new_weight_str = get_dynamic_texts(new_baggage)
            # Создаем новые объекты для динамических значений и позиционируем их в тех же местах.
            new_cargo_value = Text(new_cargo_str, font_size=24, color=BLACK)
            new_cargo_value.move_to(cargo_value.get_center())
            new_cargo_value.fixed_in_frame = True

            new_weight_value = Text(new_weight_str, font_size=24, color=BLACK)
            new_weight_value.move_to(weight_value.get_center())
            new_weight_value.fixed_in_frame = True

            # Анимируем обновление динамических значений.
            self.play(
                Transform(cargo_value, new_cargo_value),
                Transform(weight_value, new_weight_value),
                run_time=1
            )
            self.wait(0.5)

        self.wait(1)