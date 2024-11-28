import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Self
from multiprocessing import Process, Manager, Value, RawArray, Pool
from itertools import count


def clamp(val: float, val_min: float, val_max: float) -> float:
    return min(val_max, max(val_min, val))


@dataclass
class float3:
    x: float = 0
    y: float = 0
    z: float = 0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def cross(v1: Self, v2: Self) -> Self:
        return float3(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )


@dataclass
class float2:
    x: float = 0
    y: float = 0

    @staticmethod
    def distance(p1: Self, p2: Self) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return np.sqrt(dx * dx + dy * dy)

    def angle(self, other: Self) -> float:
        if self.x == other.x:
            if self.y == other.y:
                # Same position! Undefined?
                return 0.0
            elif self.y > other.y:
                return np.pi / 2
            else:
                return 3 * np.pi / 2

        v2 = float2.norm(other - self)
        angle = np.acos(v2.x)
        return angle

    @staticmethod
    def dot(v1: Self, v2: Self) -> float:
        return v1.x * v2.x + v1.y * v2.y

    @staticmethod
    def norm(v: Self) -> Self:
        return float2(v.x, v.y) / np.sqrt(v.x * v.x + v.y * v.y)

    @staticmethod
    def length(v: Self) -> float:
        return np.sqrt(v.x * v.x + v.y * v.y)

    def __sub__(self, other: Self) -> Self:
        return float2(self.x - other.x, self.y - other.y)

    def __truediv__(self, factor: float) -> Self:
        return float2(self.x / factor, self.y / factor)


class Edge:
    id: int
    angle = 0.0
    pos: float2
    _ids = count(0)

    def __init__(self, pos: float2):
        rand = np.random.default_rng()
        self.angle = rand.random() * 2 * np.pi
        self.pos = pos
        self.id = next(self._ids)


class Perlin:
    edges: np.ndarray[Edge]
    num_edges_per_row: int = 0
    num_edges_per_col: int = 0
    area_size: int = 0
    px_to_angle: any
    px_to_adjusted_distance: any
    show_borders: bool = False

    def __init__(self):
        pass

    def smooth(self, t: float) -> float:
        return ((6 * t - 15) * t + 10) * t * t * t

    def prepare(self, img_width, img_height, area_size, show_borders):
        self.show_borders = show_borders
        self.area_size = area_size
        self.num_edges_per_row = img_width // self.area_size + 1
        self.num_edges_per_col = img_height // self.area_size + 1
        print(f"Edges per row/col: {self.num_edges_per_row}/{self.num_edges_per_col}")
        self.edges = np.empty(
            (self.num_edges_per_col, self.num_edges_per_row), dtype=object
        )
        for col_edge in range(0, self.num_edges_per_row):
            cur_pos_x = col_edge * self.area_size
            for row_edge in range(0, self.num_edges_per_col):
                cur_pos_y = row_edge * self.area_size
                self.edges[row_edge, col_edge] = Edge(float2(cur_pos_x, cur_pos_y))
        self.px_to_angle = [
            [dict() for _ in range(img_width)] for _ in range(img_height)
        ]
        self.px_to_adjusted_distance = [
            [dict() for _ in range(img_width)] for _ in range(img_height)
        ]
        for col_edge in range(0, self.num_edges_per_row):
            for row_edge in range(0, self.num_edges_per_col):
                edge = self.edges[row_edge, col_edge]
                col_start = max(0, edge.pos.x - self.area_size)
                col_end = min(img_width, edge.pos.x + self.area_size)
                for col in range(col_start, col_end):
                    row_start = max(0, edge.pos.y - self.area_size)
                    row_end = min(img_height, edge.pos.y + self.area_size)
                    for row in range(row_start, row_end):
                        self.px_to_angle[row][col][edge.id] = np.atan2(
                            row - edge.pos.y, col - edge.pos.x
                        )
                        self.px_to_adjusted_distance[row][col][edge.id] = (
                            float2.distance(edge.pos, float2(col, row)) / self.area_size
                        )

    def edge_contribution(self, pos: float2, edge: Edge):
        angle_from_edge = self.px_to_angle[pos.y][pos.x][edge.id]
        angle_diff = abs(angle_from_edge - edge.angle)
        distance = self.px_to_adjusted_distance[pos.y][pos.x][edge.id]
        return np.cos(angle_diff) * distance

    def render_within_area(
        self, img, edge_dl: Edge, edge_ul: Edge, edge_ur: Edge, edge_dr: Edge
    ):
        for col in range(edge_dl.pos.x, edge_dr.pos.x):
            for row in range(edge_dl.pos.y, edge_ul.pos.y):
                pos = float2(col, row)
                contribution_dl = clamp(self.edge_contribution(pos, edge_dl), -1.0, 1.0)
                contribution_ul = clamp(self.edge_contribution(pos, edge_ul), -1.0, 1.0)
                contribution_ur = clamp(self.edge_contribution(pos, edge_ur), -1.0, 1.0)
                contribution_dr = clamp(self.edge_contribution(pos, edge_dr), -1.0, 1.0)

                tx = (pos.x - edge_dl.pos.x) / self.area_size
                ty = (pos.y - edge_dl.pos.y) / self.area_size

                if self.show_borders and (tx < 0.01 or ty < 0.01):
                    img[row, col, 1] = 1.0
                    continue

                left_down = contribution_dl + self.smooth(tx) * (
                    contribution_dr - contribution_dl
                )
                left_up = contribution_ul + self.smooth(tx) * (
                    contribution_ur - contribution_ul
                )
                noise = left_down + self.smooth(ty) * (left_up - left_down)
                noise = clamp(noise, 0, 1)
                if noise >= 0:
                    img[row, col, 0] = noise
                else:
                    img[row, col, 2] = noise

    def render(self, img):
        for col_edge in range(0, self.num_edges_per_row - 1):
            for row_edge in range(0, self.num_edges_per_col - 1):
                edge_dl = self.edges[row_edge, col_edge]
                edge_ul = self.edges[row_edge + 1, col_edge]
                edge_ur = self.edges[row_edge + 1, col_edge + 1]
                edge_dr = self.edges[row_edge, col_edge + 1]
                self.render_within_area(img, edge_dl, edge_ul, edge_ur, edge_dr)

    def move(self):
        for col_edge in range(0, self.num_edges_per_row):
            for row_edge in range(0, self.num_edges_per_col):
                self.edges[row_edge, col_edge].angle += 0.5


def run():
    img_width = 400
    img_height = 300

    perlin = Perlin()
    perlin.prepare(img_width, img_height, 50, False)

    img = np.zeros((img_height, img_width, 3), np.float32)
    window_name = "Some title goes here"
    done = False
    frame_count = 0
    t_fps = time.time()
    while not done:
        perlin.move()
        perlin.render(img)
        cv2.imshow(window_name, img)
        frame_count += 1

        t_now = time.time()
        t_diff = t_now - t_fps
        if t_diff > 1:
            frame_time_ms = 1000 * t_diff / frame_count
            print(f"Frame time (ms): {frame_time_ms:.0f}")
            frame_count = 0
            t_fps = t_now

        kc = cv2.waitKey(1)
        if kc == 27:
            done = True


if __name__ == "__main__":
    run()
