import numpy as np
import open3d as o3d
import pytest

import mojoopen3d as m3d


def clouds(seed=0, count=300):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    colors = rng.random((count, 3))
    normals = rng.normal(size=(count, 3))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    upstream.colors = o3d.utility.Vector3dVector(colors)
    upstream.normals = o3d.utility.Vector3dVector(normals)
    mojo = m3d.geometry.PointCloud(m3d.utility.Vector3dVector(points))
    mojo.colors = m3d.utility.Vector3dVector(colors)
    mojo.normals = m3d.utility.Vector3dVector(normals)
    return points, upstream, mojo


def test_vector_adapters_and_empty_cloud():
    vector = m3d.utility.Vector3dVector([[1, 2, 3], [4, 5, 6]])
    assert vector.dtype == np.float64
    assert vector.flags.c_contiguous
    cloud = m3d.geometry.PointCloud(vector)
    assert cloud.has_points()
    assert not cloud.has_normals()
    assert np.array_equal(cloud.get_min_bound(), [1, 2, 3])
    cloud.clear()
    assert cloud.is_empty()
    assert np.array_equal(cloud.get_center(), np.zeros(3))
    assert m3d.utility.Vector3iVector([[1, 2, 3]]).dtype == np.int32
    assert m3d.utility.Vector2iVector([[1, 2]]).shape == (1, 2)
    assert m3d.utility.DoubleVector([1, 2]).dtype == np.float64
    assert m3d.utility.IntVector([1, 2]).dtype == np.int32


def test_transform_translate_scale_rotate_parity():
    _, upstream, mojo = clouds(count=50)
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.2, -0.3, 0.1))
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = (0.5, -0.2, 0.3)
    upstream.transform(transformation)
    mojo.transform(transformation)
    assert np.allclose(mojo.points, upstream.points)
    assert np.allclose(mojo.normals, upstream.normals)
    upstream.translate((1, 2, 3), relative=False).scale(1.7, (0, 0, 0))
    mojo.translate((1, 2, 3), relative=False).scale(1.7, (0, 0, 0))
    upstream.rotate(rotation, (0.1, 0.2, 0.3))
    mojo.rotate(rotation, (0.1, 0.2, 0.3))
    assert np.allclose(mojo.points, upstream.points)
    assert np.allclose(mojo.normals, upstream.normals)


def test_rotation_helpers_parity():
    cases = [
        (
            m3d.geometry.get_rotation_matrix_from_xyz,
            o3d.geometry.get_rotation_matrix_from_xyz,
            (0.2, -0.1, 0.7),
        ),
        (
            m3d.geometry.get_rotation_matrix_from_axis_angle,
            o3d.geometry.get_rotation_matrix_from_axis_angle,
            (0.2, -0.1, 0.7),
        ),
        (
            m3d.geometry.get_rotation_matrix_from_quaternion,
            o3d.geometry.get_rotation_matrix_from_quaternion,
            (0.8, 0.2, -0.1, 0.3),
        ),
    ]
    for ours, theirs, value in cases:
        assert np.allclose(ours(value), theirs(value))


def test_select_uniform_and_cloud_addition_parity():
    _, upstream, mojo = clouds(count=51)
    indices = [0, 2, 8, 13, 50]
    assert np.allclose(
        mojo.select_by_index(indices).points,
        np.asarray(upstream.select_by_index(indices).points),
    )
    assert np.allclose(
        mojo.select_by_index(indices, invert=True).points,
        np.asarray(upstream.select_by_index(indices, invert=True).points),
    )
    assert np.allclose(
        mojo.uniform_down_sample(4).points,
        np.asarray(upstream.uniform_down_sample(4).points),
    )
    assert np.allclose((mojo + mojo).points, np.asarray((upstream + upstream).points))
    random_sample = mojo.random_down_sample(0.25)
    assert len(random_sample.points) == int(0.25 * len(mojo.points))
    assert {tuple(point) for point in random_sample.points} <= {
        tuple(point) for point in mojo.points
    }


def test_remove_non_finite_and_duplicates_parity():
    points = np.array(
        [[0, 1, 2], [0, 1, 2], [np.nan, 0, 0], [3, 4, np.inf], [2, 3, 4.0]]
    )
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mojo = m3d.geometry.PointCloud(points)
    upstream.remove_non_finite_points()
    mojo.remove_non_finite_points()
    upstream.remove_duplicated_points()
    mojo.remove_duplicated_points()
    assert np.array_equal(mojo.points, np.asarray(upstream.points))


@pytest.mark.parametrize("kind", ["knn", "radius", "hybrid"])
def test_kdtree_search_parity(kind):
    points, upstream, mojo = clouds(seed=4)
    query = points[17] + np.array([0.03, -0.08, 0.04])
    upstream_tree = o3d.geometry.KDTreeFlann(upstream)
    mojo_tree = m3d.geometry.KDTreeFlann(mojo)
    if kind == "knn":
        expected = upstream_tree.search_knn_vector_3d(query, 20)
        actual = mojo_tree.search_knn_vector_3d(query, 20)
    elif kind == "radius":
        expected = upstream_tree.search_radius_vector_3d(query, 0.9)
        actual = mojo_tree.search_radius_vector_3d(query, 0.9)
    else:
        expected = upstream_tree.search_hybrid_vector_3d(query, 0.9, 12)
        actual = mojo_tree.search_hybrid_vector_3d(query, 0.9, 12)
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], np.asarray(expected[1]))
    assert np.allclose(actual[2], np.asarray(expected[2]), atol=1e-14)


def test_kdtree_matrix_data_parity():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(3, 100))
    query = rng.normal(size=3)
    expected_tree = o3d.geometry.KDTreeFlann(data)
    actual_tree = m3d.geometry.KDTreeFlann(data)
    expected = expected_tree.search_knn_vector_3d(query, 8)
    actual = actual_tree.search_knn_vector_3d(query, 8)
    assert np.array_equal(actual[1], np.asarray(expected[1]))
    assert np.allclose(actual[2], np.asarray(expected[2]))


def test_kdtree_simd_tail_parity():
    rng = np.random.default_rng(15)
    points = rng.normal(size=(513, 3))
    queries = rng.normal(size=(17, 3))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    upstream_tree = o3d.geometry.KDTreeFlann(upstream)
    mojo_tree = m3d.geometry.KDTreeFlann(m3d.geometry.PointCloud(points))
    indices, distances, counts = mojo_tree._search_batch(queries, 7)
    for row, query in enumerate(queries):
        expected = upstream_tree.search_knn_vector_3d(query, 7)
        assert counts[row] == expected[0]
        assert np.array_equal(indices[row], np.asarray(expected[1]))
        assert np.allclose(distances[row], np.asarray(expected[2]), atol=1e-14)


def test_kdtree_parallel_threshold_and_chunk_tail():
    from mojoopen3d._lib import parallel_ready

    assert parallel_ready()
    rng = np.random.default_rng(25)
    points = rng.normal(size=(16_391, 3))
    queries = rng.normal(size=(1_031, 3))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    upstream_tree = o3d.geometry.KDTreeFlann(upstream)
    mojo_tree = m3d.geometry.KDTreeFlann(m3d.geometry.PointCloud(points))
    indices, distances, counts = mojo_tree._search_batch(queries, 7)
    for row, query in enumerate(queries):
        expected = upstream_tree.search_knn_vector_3d(query, 7)
        assert counts[row] == expected[0]
        assert np.array_equal(indices[row], np.asarray(expected[1]))
        assert np.allclose(distances[row], np.asarray(expected[2]), atol=1e-14)


def test_kdtree_parallel_runtime_fallback(monkeypatch):
    monkeypatch.setattr(m3d.geometry, "parallel_ready", lambda: False)
    rng = np.random.default_rng(35)
    points = rng.normal(size=(16_384, 3))
    queries = rng.normal(size=(1_024, 3))
    tree = m3d.geometry.KDTreeFlann(m3d.geometry.PointCloud(points))
    indices, distances, counts = tree._search_batch(queries, 7)
    expected_tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    )
    expected = expected_tree.search_knn_vector_3d(queries[-1], 7)
    assert counts[-1] == expected[0]
    assert np.array_equal(indices[-1], np.asarray(expected[1]))
    assert np.allclose(distances[-1], np.asarray(expected[2]), atol=1e-14)


def test_kdtree_owns_geometry_data():
    points = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0]])
    cloud = m3d.geometry.PointCloud(points)
    tree = m3d.geometry.KDTreeFlann(cloud)
    cloud.points[0] = (100, 0, 0)
    count, indices, distances = tree.search_knn_vector_3d((0, 0, 0), 1)
    assert count == 1
    assert indices[0] == 0
    assert distances[0] == 0


def test_point_cloud_distances_parity():
    _, upstream_a, mojo_a = clouds(seed=10, count=220)
    _, upstream_b, mojo_b = clouds(seed=11, count=180)
    assert np.allclose(
        mojo_a.compute_point_cloud_distance(mojo_b),
        np.asarray(upstream_a.compute_point_cloud_distance(upstream_b)),
    )
    assert np.allclose(
        mojo_a.compute_nearest_neighbor_distance(),
        np.asarray(upstream_a.compute_nearest_neighbor_distance()),
    )


def test_radius_outlier_parity():
    rng = np.random.default_rng(8)
    points = np.vstack((rng.normal(scale=0.08, size=(150, 3)), [[2, 2, 2], [-3, 0, 1]]))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mojo = m3d.geometry.PointCloud(points)
    expected_cloud, expected_indices = upstream.remove_radius_outlier(5, 0.25)
    actual_cloud, actual_indices = mojo.remove_radius_outlier(5, 0.25)
    assert np.array_equal(actual_indices, np.asarray(expected_indices))
    assert np.allclose(actual_cloud.points, np.asarray(expected_cloud.points))


def test_statistical_outlier_parity():
    rng = np.random.default_rng(18)
    points = np.vstack((rng.normal(scale=0.1, size=(200, 3)), [[4, 4, 4], [-4, 3, 2]]))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mojo = m3d.geometry.PointCloud(points)
    expected_cloud, expected_indices = upstream.remove_statistical_outlier(12, 1.5)
    actual_cloud, actual_indices = mojo.remove_statistical_outlier(12, 1.5)
    assert np.array_equal(actual_indices, np.asarray(expected_indices))
    assert np.allclose(actual_cloud.points, np.asarray(expected_cloud.points))


def test_estimate_normals_on_plane():
    rng = np.random.default_rng(9)
    xy = rng.uniform(-1, 1, size=(300, 2))
    points = np.column_stack((xy, np.zeros(len(xy))))
    upstream = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    mojo = m3d.geometry.PointCloud(points)
    upstream.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
    mojo.estimate_normals(m3d.geometry.KDTreeSearchParamKNN(knn=20))
    assert np.allclose(np.abs(mojo.normals[:, 2]), 1.0)
    assert np.allclose(
        np.abs(np.sum(mojo.normals * np.asarray(upstream.normals), axis=1)), 1.0
    )


def test_paint_and_orient_normals():
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    cloud = m3d.geometry.PointCloud(points)
    assert cloud.paint_uniform_color([0.2, 0.4, 0.6]) is cloud
    assert np.allclose(cloud.colors, [[0.2, 0.4, 0.6]] * 2)
    cloud.normals = np.array([[0, 0, -2.0], [0, 0, 3.0]])
    cloud.normalize_normals().orient_normals_to_align_with_direction()
    assert np.allclose(cloud.normals, [[0, 0, 1], [0, 0, 1]])
    cloud.normals[:] = [1, 0, 0]
    cloud.orient_normals_towards_camera_location([-1, 0, 0])
    assert np.allclose(cloud.normals, [[-1, 0, 0], [-1, 0, 0]])


def test_search_parameter_dispatch_and_ffi_input_validation():
    tree = m3d.geometry.KDTreeFlann(np.eye(3))
    direct = tree.search_knn_vector_3d([0, 0, 0], 2)
    dispatched = tree.search_vector_3d(
        [0, 0, 0], m3d.geometry.KDTreeSearchParamKNN(2)
    )
    assert direct[0] == dispatched[0]
    assert np.array_equal(direct[1], dispatched[1])
    with pytest.raises(ValueError, match="shape"):
        m3d.geometry.PointCloud(np.ones((2, 3))).transform(np.eye(3))
    with pytest.raises(ValueError, match="non-negative"):
        tree.search_radius_vector_3d([0, 0, 0], -1)
