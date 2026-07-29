"""mojo-open3d against Open3D 0.19 on identical point clouds."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import open3d as o3d

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojoopen3d as m3d  # noqa: E402


def timeit(function, repeats=3):
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def point_clouds(points, colors=None):
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mojo = m3d.geometry.PointCloud(points)
    if colors is not None:
        upstream.colors = o3d.utility.Vector3dVector(colors)
        mojo.colors = np.ascontiguousarray(colors)
    return upstream, mojo


CASES = []


def case(name):
    def decorate(function):
        CASES.append((name, function))
        return function

    return decorate


@case("Voxel downsample (1M points)")
def voxel_case():
    rng = np.random.default_rng(1)
    points = rng.uniform(-2, 2, size=(1_000_000, 3))
    colors = rng.random((1_000_000, 3))
    upstream, mojo = point_clouds(points, colors)
    return (
        lambda: mojo.voxel_down_sample(0.04),
        lambda: upstream.voxel_down_sample(0.04),
    )


@case("KD-tree build (250k points)")
def build_case():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(250_000, 3))
    upstream, mojo = point_clouds(points)
    return (
        lambda: m3d.geometry.KDTreeFlann(mojo),
        lambda: o3d.geometry.KDTreeFlann(upstream),
    )


@case("KNN k=8 (250k x 25k)")
def knn_case():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(250_000, 3))
    queries = rng.normal(size=(25_000, 3))
    upstream, mojo = point_clouds(points)
    upstream_tree = o3d.geometry.KDTreeFlann(upstream)
    mojo_tree = m3d.geometry.KDTreeFlann(mojo)

    def upstream_search():
        for query in queries:
            upstream_tree.search_knn_vector_3d(query, 8)

    return (
        lambda: mojo_tree._search_batch(queries, 8),
        upstream_search,
    )


@case("Cloud distance (100k to 250k)")
def distance_case():
    rng = np.random.default_rng(4)
    source_points = rng.normal(size=(100_000, 3))
    target_points = rng.normal(size=(250_000, 3))
    upstream_source, mojo_source = point_clouds(source_points)
    upstream_target, mojo_target = point_clouds(target_points)
    return (
        lambda: mojo_source.compute_point_cloud_distance(mojo_target),
        lambda: upstream_source.compute_point_cloud_distance(upstream_target),
    )


@case("Rigid estimate (500k pairs)")
def estimate_case():
    rng = np.random.default_rng(5)
    source_points = rng.normal(size=(500_000, 3))
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0.1, -0.04, 0.08])
    target_points = source_points @ rotation.T + [0.2, -0.1, 0.05]
    upstream_source, mojo_source = point_clouds(source_points)
    upstream_target, mojo_target = point_clouds(target_points)
    pairs = np.column_stack(
        (np.arange(len(source_points)), np.arange(len(source_points)))
    ).astype(np.int32)
    upstream_pairs = o3d.utility.Vector2iVector(pairs)
    upstream_estimator = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    mojo_estimator = m3d.pipelines.registration.TransformationEstimationPointToPoint()
    return (
        lambda: mojo_estimator.compute_transformation(
            mojo_source, mojo_target, pairs
        ),
        lambda: upstream_estimator.compute_transformation(
            upstream_source, upstream_target, upstream_pairs
        ),
    )


@case("Point-to-point ICP (30k points)")
def icp_case():
    rng = np.random.default_rng(6)
    source_points = rng.normal(size=(30_000, 3))
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0.04, -0.02, 0.05])
    target_points = source_points @ rotation.T + [0.08, -0.06, 0.03]
    upstream_source, mojo_source = point_clouds(source_points)
    upstream_target, mojo_target = point_clouds(target_points)
    init = np.eye(4)
    upstream_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=15
    )
    mojo_criteria = m3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=15
    )
    return (
        lambda: m3d.pipelines.registration.registration_icp(
            mojo_source,
            mojo_target,
            0.25,
            init,
            criteria=mojo_criteria,
        ),
        lambda: o3d.pipelines.registration.registration_icp(
            upstream_source,
            upstream_target,
            0.25,
            init,
            criteria=upstream_criteria,
        ),
    )


def machine_name():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu or 'unknown CPU'}; {platform.system()} {platform.machine()}"


def main():
    print(f"Machine: {machine_name()}")
    print()
    print("| case | mojo-open3d | Open3D 0.19 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, prepare in CASES:
        mojo_function, upstream_function = prepare()
        mojo_function()
        upstream_function()
        mojo_seconds = timeit(mojo_function)
        upstream_seconds = timeit(upstream_function)
        ratio = upstream_seconds / mojo_seconds
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1.0 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{upstream_seconds * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
