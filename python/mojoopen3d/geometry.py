"""Point clouds, voxel grids, and KD-tree search."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._lib import addr, f64, i64, lib, parallel_ready, transform_points
from .utility import Vector3dVector


def _points(value) -> np.ndarray:
    array = f64(value)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("expected an array with shape (n, 3)")
    return array


def get_rotation_matrix_from_xyz(rotation):
    x, y, z = np.asarray(rotation, dtype=np.float64)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


def get_rotation_matrix_from_axis_angle(rotation):
    vector = np.asarray(rotation, dtype=np.float64).reshape(3)
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def get_rotation_matrix_from_quaternion(rotation):
    w, x, y, z = np.asarray(rotation, dtype=np.float64).reshape(4)
    norm = w * w + x * x + y * y + z * z
    if norm == 0:
        return np.eye(3)
    scale = 2.0 / norm
    return np.array(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ]
    )


class KDTreeSearchParam:
    pass


@dataclass
class KDTreeSearchParamKNN(KDTreeSearchParam):
    knn: int = 30


@dataclass
class KDTreeSearchParamRadius(KDTreeSearchParam):
    radius: float = 0.0


@dataclass
class KDTreeSearchParamHybrid(KDTreeSearchParam):
    radius: float = 0.0
    max_nn: int = 0


class KDTreeFlann:
    """A balanced median-split 3D KD-tree backed by Mojo."""

    def __init__(self, geometry=None):
        self._points = np.empty((0, 3), dtype=np.float64)
        self._nodes = np.empty(0, dtype=np.int64)
        self._axes = np.empty(0, dtype=np.int64)
        if geometry is not None:
            if isinstance(geometry, PointCloud):
                self.set_geometry(geometry)
            else:
                self.set_matrix_data(geometry)

    def set_geometry(self, geometry: "PointCloud") -> bool:
        return self._set_points(geometry.points)

    def set_matrix_data(self, data) -> bool:
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("KDTreeFlann data must be two-dimensional")
        if array.shape[0] == 3:
            array = array.T
        return self._set_points(array)

    def _set_points(self, points) -> bool:
        self._points = _points(points).copy()
        n = len(self._points)
        capacity = max(1, 2 * n + 1)
        self._nodes = np.empty(capacity, dtype=np.int64)
        self._axes = np.empty(capacity, dtype=np.int64)
        indices = np.empty(n, dtype=np.int64)
        if n:
            lib().m3d_kdtree_build(
                addr(self._points),
                n,
                addr(indices),
                addr(self._nodes),
                addr(self._axes),
                capacity,
                int(parallel_ready()),
            )
        return n > 0

    def _search_batch(
        self, queries, limit: int, max_distance: float = math.inf
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = _points(np.asarray(queries, dtype=np.float64).reshape((-1, 3)))
        max_distance = float(max_distance)
        if math.isnan(max_distance) or max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        limit = max(0, min(int(limit), len(self._points)))
        if limit == 0:
            return (
                np.empty((len(queries), 0), dtype=np.int64),
                np.empty((len(queries), 0), dtype=np.float64),
                np.zeros(len(queries), dtype=np.int64),
            )
        indices = np.empty((len(queries), limit), dtype=np.int64)
        distances = np.empty((len(queries), limit), dtype=np.float64)
        counts = np.empty(len(queries), dtype=np.int64)
        max_distance2 = (
            np.finfo(np.float64).max if math.isinf(max_distance) else max_distance**2
        )
        if len(queries):
            lib().m3d_kdtree_search(
                addr(self._points),
                addr(self._nodes),
                addr(self._axes),
                len(self._nodes),
                addr(queries),
                len(queries),
                limit,
                max_distance2,
                addr(indices),
                addr(distances),
                addr(counts),
                int(parallel_ready()),
            )
        return indices, distances, counts

    def search_knn_vector_3d(self, query, knn: int):
        indices, distances, counts = self._search_batch([query], knn)
        count = int(counts[0])
        return count, indices[0, :count].astype(np.int32), distances[0, :count].copy()

    def search_radius_vector_3d(self, query, radius: float):
        indices, distances, counts = self._search_batch(
            [query], len(self._points), float(radius)
        )
        count = int(counts[0])
        return count, indices[0, :count].astype(np.int32), distances[0, :count].copy()

    def search_hybrid_vector_3d(self, query, radius: float, max_nn: int):
        indices, distances, counts = self._search_batch([query], max_nn, float(radius))
        count = int(counts[0])
        return count, indices[0, :count].astype(np.int32), distances[0, :count].copy()

    def search_vector_3d(self, query, search_param: KDTreeSearchParam):
        if isinstance(search_param, KDTreeSearchParamKNN):
            return self.search_knn_vector_3d(query, search_param.knn)
        if isinstance(search_param, KDTreeSearchParamRadius):
            return self.search_radius_vector_3d(query, search_param.radius)
        if isinstance(search_param, KDTreeSearchParamHybrid):
            return self.search_hybrid_vector_3d(
                query, search_param.radius, search_param.max_nn
            )
        raise TypeError("unsupported KDTreeSearchParam")


def _voxel_aggregate(
    cloud: "PointCloud",
    voxel_size: float,
    origin: np.ndarray,
    *,
    bounded: bool = False,
    max_index=(0, 0, 0),
):
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")
    points = _points(cloud.points)
    if not np.all(np.isfinite(points)):
        raise ValueError("voxel operations require finite points")
    n = len(points)
    if n == 0:
        return PointCloud(), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    colors = _points(cloud.colors) if cloud.has_colors() else np.empty((1, 3))
    normals = _points(cloud.normals) if cloud.has_normals() else np.empty((1, 3))
    capacity = 1
    while capacity < n * 2:
        capacity *= 2
    slot_x = np.empty(capacity, dtype=np.int64)
    slot_y = np.empty(capacity, dtype=np.int64)
    slot_z = np.empty(capacity, dtype=np.int64)
    slot_ids = np.full(capacity, -1, dtype=np.int64)
    result_points = np.empty((n, 3), dtype=np.float64)
    result_colors = np.empty((n, 3), dtype=np.float64)
    result_normals = np.empty((n, 3), dtype=np.float64)
    result_counts = np.empty(n, dtype=np.int64)
    trace_ids = np.empty(n, dtype=np.int64)
    count = lib().m3d_voxel_aggregate(
        addr(points),
        addr(colors),
        addr(normals),
        n,
        float(voxel_size),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        int(bounded),
        int(max_index[0]),
        int(max_index[1]),
        int(max_index[2]),
        int(cloud.has_colors()),
        int(cloud.has_normals()),
        addr(slot_x),
        addr(slot_y),
        addr(slot_z),
        addr(slot_ids),
        capacity,
        addr(result_points),
        addr(result_colors),
        addr(result_normals),
        addr(result_counts),
        addr(trace_ids),
    )
    if not 0 <= count <= n:
        raise RuntimeError(f"voxel kernel returned invalid result count {count}")
    if np.any((trace_ids < 0) | (trace_ids >= count)):
        raise RuntimeError("voxel kernel returned an invalid trace ID")
    result = PointCloud(Vector3dVector(result_points[:count]))
    if cloud.has_colors():
        result.colors = Vector3dVector(result_colors[:count])
    if cloud.has_normals():
        result.normals = Vector3dVector(result_normals[:count])
    return result, trace_ids, result_counts[:count].copy()


class PointCloud:
    def __init__(self, points=None):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.normals = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.float64)
        if isinstance(points, PointCloud):
            self.points = points.points.copy()
            self.normals = points.normals.copy()
            self.colors = points.colors.copy()
        elif points is not None:
            self.points = Vector3dVector(points)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"PointCloud with {len(self.points)} points."

    def __iadd__(self, other):
        if not isinstance(other, PointCloud):
            return NotImplemented
        old_count = len(self.points)
        keep_normals = (old_count == 0 or self.has_normals()) and other.has_normals()
        keep_colors = (old_count == 0 or self.has_colors()) and other.has_colors()
        self.points = np.ascontiguousarray(
            np.concatenate((self.points, other.points), axis=0)
        )
        self.normals = (
            np.ascontiguousarray(np.concatenate((self.normals, other.normals), axis=0))
            if keep_normals
            else np.empty((0, 3), dtype=np.float64)
        )
        self.colors = (
            np.ascontiguousarray(np.concatenate((self.colors, other.colors), axis=0))
            if keep_colors
            else np.empty((0, 3), dtype=np.float64)
        )
        return self

    def __add__(self, other):
        result = PointCloud(self)
        result += other
        return result

    def has_points(self) -> bool:
        return len(self.points) > 0

    def has_normals(self) -> bool:
        return len(self.normals) == len(self.points) and len(self.points) > 0

    def has_colors(self) -> bool:
        return len(self.colors) == len(self.points) and len(self.points) > 0

    def is_empty(self) -> bool:
        return not self.has_points()

    def clear(self):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.normals = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.float64)
        return self

    def get_min_bound(self):
        return np.min(self.points, axis=0) if len(self.points) else np.zeros(3)

    def get_max_bound(self):
        return np.max(self.points, axis=0) if len(self.points) else np.zeros(3)

    def get_center(self):
        return np.mean(self.points, axis=0) if len(self.points) else np.zeros(3)

    def transform(self, transformation):
        matrix = f64(transformation)
        if matrix.shape != (4, 4):
            raise ValueError("transformation must have shape (4, 4)")
        self.points = transform_points(self.points, matrix)
        if self.has_normals():
            self.normals = np.ascontiguousarray(self.normals @ matrix[:3, :3].T)
        return self

    def translate(self, translation, relative: bool = True):
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        if relative:
            self.points = np.ascontiguousarray(self.points + translation)
        else:
            self.points = np.ascontiguousarray(
                self.points + translation - self.get_center()
            )
        return self

    def scale(self, scale: float, center):
        center = np.asarray(center, dtype=np.float64).reshape(3)
        self.points = np.ascontiguousarray(
            (self.points - center) * float(scale) + center
        )
        return self

    def rotate(self, rotation, center=np.zeros(3)):
        rotation = np.asarray(rotation, dtype=np.float64).reshape((3, 3))
        center = np.asarray(center, dtype=np.float64).reshape(3)
        self.points = np.ascontiguousarray(
            (self.points - center) @ rotation.T + center
        )
        if self.has_normals():
            self.normals = np.ascontiguousarray(self.normals @ rotation.T)
        return self

    def paint_uniform_color(self, color):
        color = np.asarray(color, dtype=np.float64).reshape(3)
        self.colors = np.ascontiguousarray(
            np.broadcast_to(color, (len(self.points), 3)).copy()
        )
        return self

    def normalize_normals(self):
        if self.has_normals():
            lengths = np.linalg.norm(self.normals, axis=1)
            valid = lengths > 0
            self.normals[valid] /= lengths[valid, None]
        return self

    def estimate_normals(
        self,
        search_param=KDTreeSearchParamKNN(),
        fast_normal_computation: bool = True,
    ):
        tree = KDTreeFlann(self)
        if isinstance(search_param, KDTreeSearchParamKNN):
            indices, _, counts = tree._search_batch(self.points, search_param.knn)
        elif isinstance(search_param, KDTreeSearchParamHybrid):
            indices, _, counts = tree._search_batch(
                self.points, search_param.max_nn, search_param.radius
            )
        elif isinstance(search_param, KDTreeSearchParamRadius):
            rows = [
                tree.search_radius_vector_3d(point, search_param.radius)[1]
                for point in self.points
            ]
            width = max((len(row) for row in rows), default=0)
            indices = np.full((len(rows), width), -1, dtype=np.int64)
            counts = np.array([len(row) for row in rows], dtype=np.int64)
            for row_index, row in enumerate(rows):
                indices[row_index, : len(row)] = row
        else:
            raise TypeError("unsupported KDTreeSearchParam")
        normals = np.zeros_like(self.points)
        for point_index, count in enumerate(counts):
            neighbors = self.points[indices[point_index, : int(count)]]
            if len(neighbors) < 3:
                normals[point_index] = (0, 0, 1)
                continue
            centered = neighbors - neighbors.mean(axis=0)
            covariance = centered.T @ centered / len(neighbors)
            _, vectors = np.linalg.eigh(covariance)
            normal = vectors[:, 0]
            if self.has_normals() and np.dot(normal, self.normals[point_index]) < 0:
                normal = -normal
            normals[point_index] = normal
        self.normals = np.ascontiguousarray(normals)
        return self

    def orient_normals_to_align_with_direction(
        self, orientation_reference=np.array([0.0, 0.0, 1.0])
    ):
        if not self.has_normals():
            raise ValueError("No normals in the PointCloud")
        reference = np.asarray(orientation_reference, dtype=np.float64).reshape(3)
        flip = self.normals @ reference < 0
        self.normals[flip] *= -1
        return self

    def orient_normals_towards_camera_location(
        self, camera_location=np.zeros(3)
    ):
        if not self.has_normals():
            raise ValueError("No normals in the PointCloud")
        camera = np.asarray(camera_location, dtype=np.float64).reshape(3)
        flip = np.sum(self.normals * (camera - self.points), axis=1) < 0
        self.normals[flip] *= -1
        return self

    def select_by_index(self, indices, invert: bool = False):
        indices = np.asarray(indices, dtype=np.int64)
        mask = np.full(len(self.points), bool(invert))
        mask[indices] = not invert
        result = PointCloud(self.points[mask])
        if self.has_normals():
            result.normals = Vector3dVector(self.normals[mask])
        if self.has_colors():
            result.colors = Vector3dVector(self.colors[mask])
        return result

    def uniform_down_sample(self, every_k_points: int):
        if every_k_points <= 0:
            raise ValueError("every_k_points must be positive")
        return self.select_by_index(np.arange(0, len(self.points), every_k_points))

    def random_down_sample(self, sampling_ratio: float):
        if not 0 <= sampling_ratio <= 1:
            raise ValueError("sampling_ratio must be between 0 and 1")
        count = int(sampling_ratio * len(self.points))
        return self.select_by_index(np.random.permutation(len(self.points))[:count])

    def remove_non_finite_points(
        self, remove_nan: bool = True, remove_infinite: bool = True
    ):
        mask = np.ones(len(self.points), dtype=bool)
        if remove_nan:
            mask &= ~np.isnan(self.points).any(axis=1)
        if remove_infinite:
            mask &= ~np.isinf(self.points).any(axis=1)
        filtered = self.select_by_index(np.flatnonzero(mask))
        self.points, self.normals, self.colors = (
            filtered.points,
            filtered.normals,
            filtered.colors,
        )
        return self

    def remove_duplicated_points(self):
        if not len(self.points):
            return self
        _, first = np.unique(self.points, axis=0, return_index=True)
        filtered = self.select_by_index(np.sort(first))
        self.points, self.normals, self.colors = (
            filtered.points,
            filtered.normals,
            filtered.colors,
        )
        return self

    def voxel_down_sample(self, voxel_size: float):
        if not len(self.points):
            return PointCloud()
        origin = self.get_min_bound() - float(voxel_size) * 0.5
        result, _, _ = _voxel_aggregate(self, voxel_size, origin)
        return result

    def voxel_down_sample_and_trace(
        self, voxel_size, min_bound, max_bound, approximate_class: bool = False
    ):
        origin = np.asarray(min_bound, dtype=np.float64)
        result, trace_ids, _ = _voxel_aggregate(self, voxel_size, origin)
        count = len(result.points)
        if approximate_class and self.has_colors():
            for group in range(count):
                values = self.colors[trace_ids == group, 0].astype(np.int64)
                if len(values):
                    winner = np.bincount(values).argmax()
                    result.colors[group] = winner
        cubic_id = np.full((count, 8), -1, dtype=np.int32)
        original_indices = []
        relative = (self.points - origin) / float(voxel_size)
        fractions = relative - np.floor(relative)
        corner = (
            (fractions[:, 0] >= 0.5).astype(np.int32)
            + 2 * (fractions[:, 1] >= 0.5).astype(np.int32)
            + 4 * (fractions[:, 2] >= 0.5).astype(np.int32)
        )
        for group in range(count):
            members = np.flatnonzero(trace_ids == group)
            original_indices.append(members.astype(np.int32))
            cubic_id[group, corner[members]] = members
        return result, cubic_id, original_indices

    def compute_point_cloud_distance(self, target: "PointCloud"):
        if not len(self.points):
            return np.empty(0)
        if not len(target.points):
            return np.zeros(len(self.points))
        tree = KDTreeFlann(target)
        _, distances, _ = tree._search_batch(self.points, 1)
        return np.sqrt(distances[:, 0])

    def compute_nearest_neighbor_distance(self):
        if len(self.points) < 2:
            return np.zeros(len(self.points))
        tree = KDTreeFlann(self)
        _, distances, _ = tree._search_batch(self.points, 2)
        return np.sqrt(distances[:, 1])

    def remove_radius_outlier(self, nb_points: int, radius: float, print_progress=False):
        tree = KDTreeFlann(self)
        _, _, counts = tree._search_batch(self.points, int(nb_points) + 1, radius)
        indices = np.flatnonzero(counts > int(nb_points)).astype(np.int32)
        return self.select_by_index(indices), indices

    def remove_statistical_outlier(
        self, nb_neighbors: int, std_ratio: float, print_progress=False
    ):
        tree = KDTreeFlann(self)
        _, distances, counts = tree._search_batch(self.points, nb_neighbors)
        valid = counts > 0
        means = np.full(len(self.points), -1.0)
        means[valid] = np.sqrt(distances[valid]).mean(axis=1)
        values = means[means > 0]
        threshold = values.mean() + float(std_ratio) * values.std(ddof=1)
        indices = np.flatnonzero((means > 0) & (means < threshold)).astype(np.int32)
        return self.select_by_index(indices), indices


@dataclass
class Voxel:
    grid_index: np.ndarray
    color: np.ndarray

    def __init__(self, grid_index=(0, 0, 0), color=(0, 0, 0)):
        self.grid_index = np.asarray(grid_index, dtype=np.int32)
        self.color = np.asarray(color, dtype=np.float64)


class VoxelGrid:
    def __init__(self):
        self.origin = np.zeros(3, dtype=np.float64)
        self.voxel_size = 0.0
        self._voxels: dict[tuple[int, int, int], Voxel] = {}

    def has_voxels(self):
        return bool(self._voxels)

    def is_empty(self):
        return not self.has_voxels()

    def clear(self):
        self._voxels.clear()
        return self

    def get_voxels(self):
        return list(self._voxels.values())

    def get_voxel(self, point):
        index = np.floor(
            (np.asarray(point, dtype=np.float64) - self.origin) / self.voxel_size
        )
        limits = np.iinfo(np.int32)
        if not np.all(np.isfinite(index)) or np.any(index < limits.min) or np.any(
            index > limits.max
        ):
            raise OverflowError("voxel index does not fit in int32")
        return index.astype(np.int32)

    def check_if_included(self, queries):
        return [
            tuple(self.get_voxel(point).tolist()) in self._voxels for point in queries
        ]

    def get_min_bound(self):
        return self.origin.copy()

    def get_max_bound(self):
        if not self._voxels:
            return self.origin.copy()
        indices = np.array(list(self._voxels), dtype=np.float64)
        return self.origin + self.voxel_size * (indices.max(axis=0) + 1)

    @staticmethod
    def create_from_point_cloud(point_cloud: PointCloud, voxel_size: float):
        grid = VoxelGrid()
        grid.voxel_size = float(voxel_size)
        grid.origin = point_cloud.get_min_bound() - 0.5 * grid.voxel_size
        down, trace, _ = _voxel_aggregate(point_cloud, voxel_size, grid.origin)
        voxel_indices = np.floor(
            (down.points - grid.origin) / grid.voxel_size
        ).astype(np.int32)
        for group, index in enumerate(voxel_indices):
            color = down.colors[group] if down.has_colors() else np.zeros(3)
            grid._voxels[tuple(index.tolist())] = Voxel(index, color)
        return grid

    @staticmethod
    def create_from_point_cloud_within_bounds(
        point_cloud: PointCloud, voxel_size: float, min_bound, max_bound
    ):
        grid = VoxelGrid()
        grid.voxel_size = float(voxel_size)
        grid.origin = np.asarray(min_bound, dtype=np.float64)
        maximum = np.asarray(max_bound, dtype=np.float64)
        voxel_indices = np.floor(
            (point_cloud.points - grid.origin) / grid.voxel_size
        ).astype(np.int64)
        _ = maximum
        for key in np.unique(voxel_indices, axis=0):
            mask = np.all(voxel_indices == key, axis=1)
            color = (
                point_cloud.colors[mask].mean(axis=0)
                if point_cloud.has_colors()
                else np.zeros(3)
            )
            grid._voxels[tuple(key.tolist())] = Voxel(key, color)
        return grid
