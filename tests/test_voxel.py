import numpy as np
import open3d as o3d
import pytest

import mojoopen3d as m3d


def make_cloud(seed=0, count=500):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-2, 2, size=(count, 3))
    colors = rng.random((count, 3))
    normals = rng.normal(size=(count, 3))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    upstream.colors = o3d.utility.Vector3dVector(colors)
    upstream.normals = o3d.utility.Vector3dVector(normals)
    mojo = m3d.geometry.PointCloud(points)
    mojo.colors = colors.copy()
    mojo.normals = normals.copy()
    return upstream, mojo


def sorted_attributes(cloud):
    points = np.asarray(cloud.points)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return (
        points[order],
        np.asarray(cloud.colors)[order],
        np.asarray(cloud.normals)[order],
    )


@pytest.mark.parametrize("voxel_size", [0.15, 0.5, 1.25])
def test_voxel_down_sample_parity(voxel_size):
    upstream, mojo = make_cloud(seed=2)
    expected = sorted_attributes(upstream.voxel_down_sample(voxel_size))
    actual = sorted_attributes(mojo.voxel_down_sample(voxel_size))
    for left, right in zip(actual, expected):
        assert np.allclose(left, right, atol=1e-14)


def test_voxel_trace_membership_and_corners_parity():
    upstream, mojo = make_cloud(seed=7, count=200)
    minimum = np.array([-2.2, -2.2, -2.2])
    maximum = np.array([2.2, 2.2, 2.2])
    expected_cloud, expected_corners, expected_groups = (
        upstream.voxel_down_sample_and_trace(0.7, minimum, maximum)
    )
    actual_cloud, actual_corners, actual_groups = mojo.voxel_down_sample_and_trace(
        0.7, minimum, maximum
    )
    expected_by_group = {
        tuple(sorted(np.asarray(group).tolist())): np.asarray(expected_corners)[i]
        for i, group in enumerate(expected_groups)
    }
    actual_by_group = {
        tuple(sorted(np.asarray(group).tolist())): actual_corners[i]
        for i, group in enumerate(actual_groups)
    }
    assert actual_by_group.keys() == expected_by_group.keys()
    for key in actual_by_group:
        assert np.array_equal(actual_by_group[key], expected_by_group[key])
    assert len(actual_cloud.points) == len(expected_cloud.points)


def test_voxel_grid_parity():
    upstream, mojo = make_cloud(seed=12, count=250)
    expected = o3d.geometry.VoxelGrid.create_from_point_cloud(upstream, 0.4)
    actual = m3d.geometry.VoxelGrid.create_from_point_cloud(mojo, 0.4)
    expected_voxels = {
        tuple(voxel.grid_index): np.asarray(voxel.color)
        for voxel in expected.get_voxels()
    }
    actual_voxels = {
        tuple(voxel.grid_index): np.asarray(voxel.color)
        for voxel in actual.get_voxels()
    }
    assert actual.origin == pytest.approx(expected.origin)
    assert actual.voxel_size == expected.voxel_size
    assert actual_voxels.keys() == expected_voxels.keys()
    for key in actual_voxels:
        assert np.allclose(actual_voxels[key], expected_voxels[key])
    queries = np.asarray(upstream.points)[:20]
    assert actual.check_if_included(queries) == expected.check_if_included(
        o3d.utility.Vector3dVector(queries)
    )
    assert np.allclose(actual.get_min_bound(), expected.get_min_bound())
    assert np.allclose(actual.get_max_bound(), expected.get_max_bound())
    assert np.array_equal(
        actual.get_voxel(queries[0]), expected.get_voxel(queries[0])
    )
    assert actual.has_voxels() and not actual.is_empty()
    actual.clear()
    assert actual.is_empty()


def test_voxel_grid_within_bounds_parity():
    upstream, mojo = make_cloud(seed=13, count=180)
    minimum = np.array([-1.5, -1.5, -1.5])
    maximum = np.array([1.5, 1.5, 1.5])
    expected = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        upstream, 0.3, minimum, maximum
    )
    actual = m3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        mojo, 0.3, minimum, maximum
    )
    expected_keys = {tuple(voxel.grid_index) for voxel in expected.get_voxels()}
    actual_keys = {tuple(voxel.grid_index) for voxel in actual.get_voxels()}
    assert actual_keys == expected_keys


def test_voxel_index_overflow_is_not_silently_narrowed():
    grid = m3d.geometry.VoxelGrid()
    grid.voxel_size = 1.0
    with pytest.raises(OverflowError, match="int32"):
        grid.get_voxel([2**40, 0, 0])
    cloud = m3d.geometry.PointCloud([[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="finite points"):
        cloud.voxel_down_sample(1.0)
