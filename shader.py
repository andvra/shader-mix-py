import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Self
from multiprocessing import Process, Manager, Value, RawArray, Pool


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

        # auto dist_norm = distance(edge->pos, pos) / edge_spacing_px;
        # auto right = float2(1, 0);
        # auto v = pos - edge->pos;
        # auto cross_prod = cross(float3(v, 0), float3(right, 1));
        # auto is_inward = cross_prod.z < 0.0f;
        # auto cos_angle = dot(v, right) / length(v);
        # auto angle_from_edge = std::acosf(cos_angle);
        # if (!is_inward) {
        # 	angle_from_edge = 2.0f * pi_f - angle_from_edge;
        # }

        v2 = float2.norm(other - self)
        angle: float = 0
        if self.y > other.y:
            angle = np.acos(v2.x)
        else:
            angle = np.acos(v2.x)  # np.pi + np.acos(-v2.x)
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
    angle = 0.0
    pos: float2

    def __init__(self, pos: float2):
        rand = np.random.default_rng()
        self.angle = rand.random() * 2 * np.pi
        self.pos = pos


class Perlin:
    edges: np.ndarray[Edge]
    num_edges_per_row: int = 0
    num_edges_per_col: int = 0
    area_size: int = 0

    def __init__(self):
        pass

    def smooth(self, t: float) -> float:
        return ((6 * t - 15) * t + 10) * t * t * t

    def prepare(self, img_width, img_height):
        self.area_size = 100
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

    def edge_contribution(self, pos: float2, edge: Edge):
        # TODO: Improve

        # auto dist_norm = distance(edge->pos, pos) / edge_spacing_px;
        # auto right = float2(1, 0);
        # auto v = pos - edge->pos;
        # auto cross_prod = cross(float3(v, 0), float3(right, 1));
        # auto is_inward = cross_prod.z < 0.0f;
        # auto cos_angle = dot(v, right) / length(v);
        # auto angle_from_edge = std::acosf(cos_angle);
        # if (!is_inward) {
        # 	angle_from_edge = 2.0f * pi_f - angle_from_edge;
        # }

        v = pos - edge.pos
        if float2.length(v) == 0:
            return 0

        right = float2(1, 0)
        cross_prod = float3.cross(float3(v.x, v.y, 0), float3(right.x, right.y, 0))
        is_inward = cross_prod.z < 0.0
        cos_angle = float2.dot(v, right) / float2.length(v)
        angle_from_edge = np.acos(cos_angle)
        if not is_inward:
            angle_from_edge = 2.0 * np.pi - angle_from_edge

        angle_from_edge = np.atan2(pos.y - edge.pos.y, pos.x - edge.pos.x)
        # angle_pos = pos.angle(edge.pos)
        angle_diff = abs(angle_from_edge - edge.angle)
        return np.cos(angle_diff) * float2.distance(pos, edge.pos) / self.area_size

    def render_within_area(
        self, img, edge_dl: Edge, edge_ul: Edge, edge_ur: Edge, edge_dr: Edge
    ):
        for col in range(edge_dl.pos.x, edge_dr.pos.x):
            for row in range(edge_dl.pos.y, edge_ul.pos.y):
                # TODO: Remove limit
                if row > 200 or col > 600:
                    continue
                pos = float2(col, row)
                contribution_dl = clamp(self.edge_contribution(pos, edge_dl), -1.0, 1.0)
                contribution_ul = clamp(self.edge_contribution(pos, edge_ul), -1.0, 1.0)
                contribution_ur = clamp(self.edge_contribution(pos, edge_ur), -1.0, 1.0)
                contribution_dr = clamp(self.edge_contribution(pos, edge_dr), -1.0, 1.0)

                tx = (pos.x - edge_dl.pos.x) / self.area_size
                ty = (pos.y - edge_dl.pos.y) / self.area_size

                if tx < 0.01 or ty < 0.01:
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
                self.edges[row_edge, col_edge].angle += 0.1


def run():
    img_width = 800
    img_height = 600

    perlin = Perlin()
    perlin.prepare(img_width, img_height)

    img = np.zeros((img_height, img_width, 3), np.float32)
    window_name = "Some title goes here"
    done = False
    frame_count = 0
    t_fps = time.time()
    while not done:
        # perlin.move()
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


def run_test_inner(some_var, idx):
    some_var[idx] = idx


def run_test():
    pass
    # the_array = np.ndarray(10, np.float32)
    # some_var = RawArray("d", 10)
    # some_var_np = np.frombuffer(some_var)
    # np.copyto(some_var_np, the_array)
    # with Pool(processes=4) as pool:
    #     res = pool.map(run_test_inner, 10)
    #     print(res)
    # processes = []
    # for idx in range(10):
    #     processes.append(Process(target=run_test_inner, args=(some_var, idx)).start())

    # for p in processes:
    #     p.join()
    # print(the_array)


if __name__ == "__main__":
    # run_test()
    run()