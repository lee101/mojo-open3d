"""Point-to-point registration with an Open3D-compatible interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._lib import rigid_estimate, transform_points
from ..geometry import KDTreeFlann, PointCloud


@dataclass
class ICPConvergenceCriteria:
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    max_iteration: int = 30


@dataclass
class RegistrationResult:
    transformation: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    correspondence_set: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    inlier_rmse: float = 0.0
    fitness: float = 0.0

    def __repr__(self):
        return (
            "RegistrationResult with fitness="
            f"{self.fitness:.6e}, inlier_rmse={self.inlier_rmse:.6e}, "
            f"and correspondence_set size of {len(self.correspondence_set)}"
        )


def _correspondences(
    source_points: np.ndarray,
    tree: KDTreeFlann,
    max_correspondence_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(source_points) or not len(tree._points):
        return (
            np.full(len(source_points), -1, dtype=np.int64),
            np.zeros(len(source_points), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
        )
    indices, distances, counts = tree._search_batch(
        source_points, 1, max_correspondence_distance
    )
    target_indices = indices[:, 0]
    valid = counts == 1
    target_indices[~valid] = -1
    source_indices = np.flatnonzero(valid)
    pairs = np.column_stack((source_indices, target_indices[valid])).astype(np.int32)
    return target_indices, distances[:, 0], pairs


def _result(
    source_points: np.ndarray,
    target_count: int,
    tree: KDTreeFlann,
    max_correspondence_distance: float,
    transformation: np.ndarray,
) -> RegistrationResult:
    transformed = transform_points(source_points, transformation)
    _, distances, pairs = _correspondences(
        transformed, tree, max_correspondence_distance
    )
    count = len(pairs)
    fitness = count / len(source_points) if len(source_points) else 0.0
    rmse = (
        float(np.sqrt(distances[pairs[:, 0]].mean()))
        if count
        else 0.0
    )
    return RegistrationResult(
        transformation=np.array(transformation, copy=True),
        correspondence_set=pairs,
        inlier_rmse=rmse,
        fitness=fitness,
    )


class TransformationEstimation:
    pass


class TransformationEstimationPointToPoint(TransformationEstimation):
    def __init__(self, with_scaling: bool = False):
        self.with_scaling = bool(with_scaling)

    def compute_transformation(self, source, target, correspondence_set):
        source_points = np.asarray(source.points, dtype=np.float64)
        target_points = np.asarray(target.points, dtype=np.float64)
        pairs = np.asarray(correspondence_set, dtype=np.int64).reshape((-1, 2))
        if len(pairs) and (
            np.any(pairs[:, 0] < 0)
            or np.any(pairs[:, 0] >= len(source_points))
            or np.any(pairs[:, 1] < 0)
            or np.any(pairs[:, 1] >= len(target_points))
        ):
            raise IndexError("correspondence index is outside its point cloud")
        target_indices = np.full(len(source_points), -1, dtype=np.int64)
        if len(pairs):
            target_indices[pairs[:, 0]] = pairs[:, 1]
        return rigid_estimate(
            source_points, target_points, target_indices, self.with_scaling
        )

    def compute_rmse(self, source, target, correspondence_set):
        pairs = np.asarray(correspondence_set, dtype=np.int64).reshape((-1, 2))
        if not len(pairs):
            return 0.0
        delta = source.points[pairs[:, 0]] - target.points[pairs[:, 1]]
        return float(np.sqrt(np.sum(delta * delta) / len(pairs)))


def evaluate_registration(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    transformation=np.eye(4),
):
    tree = KDTreeFlann(target)
    return _result(
        source.points,
        len(target.points),
        tree,
        max_correspondence_distance,
        np.asarray(transformation, dtype=np.float64),
    )


def registration_icp(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    init=np.eye(4),
    estimation_method=None,
    criteria=None,
):
    if estimation_method is None:
        estimation_method = TransformationEstimationPointToPoint()
    if not isinstance(estimation_method, TransformationEstimationPointToPoint):
        raise NotImplementedError("only TransformationEstimationPointToPoint is covered")
    if criteria is None:
        criteria = ICPConvergenceCriteria()
    transformation = np.array(init, dtype=np.float64, copy=True)
    tree = KDTreeFlann(target)
    if not len(source.points) or not len(target.points):
        return RegistrationResult(transformation=transformation)
    previous_fitness = -1.0
    previous_rmse = -1.0
    for _ in range(int(criteria.max_iteration)):
        current = transform_points(source.points, transformation)
        indices, distances, counts = tree._search_batch(
            current, 1, max_correspondence_distance
        )
        target_indices = indices[:, 0]
        valid = counts == 1
        target_indices[~valid] = -1
        correspondence_count = int(np.count_nonzero(valid))
        if not correspondence_count:
            break
        fitness = correspondence_count / len(source.points)
        rmse = float(
            np.sqrt(
                np.sum(distances[:, 0], where=valid) / correspondence_count
            )
        )
        delta = rigid_estimate(
            current, target.points, target_indices, estimation_method.with_scaling
        )
        transformation = delta @ transformation
        if (
            previous_fitness >= 0
            and abs(fitness - previous_fitness) < criteria.relative_fitness
            and abs(rmse - previous_rmse) < criteria.relative_rmse
        ):
            break
        previous_fitness = fitness
        previous_rmse = rmse
    return _result(
        source.points,
        len(target.points),
        tree,
        max_correspondence_distance,
        transformation,
    )


def get_information_matrix_from_point_clouds(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    transformation,
):
    transformed = transform_points(source.points, np.asarray(transformation))
    tree = KDTreeFlann(target)
    _, _, pairs = _correspondences(
        transformed, tree, max_correspondence_distance
    )
    information = np.zeros((6, 6), dtype=np.float64)
    for source_index, _ in pairs:
        x, y, z = transformed[source_index]
        jacobian = np.array(
            [[0, z, -y, 1, 0, 0], [-z, 0, x, 0, 1, 0], [y, -x, 0, 0, 0, 1]],
            dtype=np.float64,
        )
        information += jacobian.T @ jacobian
    return information
