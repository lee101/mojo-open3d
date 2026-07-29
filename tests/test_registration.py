import numpy as np
import open3d as o3d
import pytest

import mojoopen3d as m3d


def pair(seed=0, count=400):
    rng = np.random.default_rng(seed)
    source_points = rng.normal(size=(count, 3))
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0.08, -0.05, 0.12])
    translation = np.array([0.18, -0.11, 0.07])
    target_points = source_points @ rotation.T + translation
    upstream_source = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(source_points)
    )
    upstream_target = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(target_points)
    )
    mojo_source = m3d.geometry.PointCloud(source_points)
    mojo_target = m3d.geometry.PointCloud(target_points)
    truth = np.eye(4)
    truth[:3, :3] = rotation
    truth[:3, 3] = translation
    return upstream_source, upstream_target, mojo_source, mojo_target, truth


def test_point_to_point_transformation_parity():
    upstream_source, upstream_target, mojo_source, mojo_target, truth = pair()
    count = len(mojo_source.points)
    pairs = np.column_stack((np.arange(count), np.arange(count))).astype(np.int32)
    expected = (
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
        .compute_transformation(
            upstream_source,
            upstream_target,
            o3d.utility.Vector2iVector(pairs),
        )
    )
    estimator = m3d.pipelines.registration.TransformationEstimationPointToPoint()
    actual = estimator.compute_transformation(mojo_source, mojo_target, pairs)
    assert np.allclose(actual, expected, atol=1e-11)
    assert np.allclose(actual, truth, atol=1e-11)
    assert estimator.compute_rmse(mojo_source, mojo_target, pairs) == pytest.approx(
        o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_rmse(
            upstream_source, upstream_target, o3d.utility.Vector2iVector(pairs)
        )
    )


def test_point_to_point_with_scaling_parity():
    upstream_source, _, mojo_source, _, _ = pair(seed=3)
    target_points = np.asarray(upstream_source.points) * 1.7 + [0.2, -0.4, 0.1]
    upstream_target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))
    mojo_target = m3d.geometry.PointCloud(target_points)
    pairs = np.column_stack(
        (np.arange(len(target_points)), np.arange(len(target_points)))
    ).astype(np.int32)
    expected = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        True
    ).compute_transformation(
        upstream_source, upstream_target, o3d.utility.Vector2iVector(pairs)
    )
    actual = m3d.pipelines.registration.TransformationEstimationPointToPoint(
        True
    ).compute_transformation(mojo_source, mojo_target, pairs)
    assert np.allclose(actual, expected, atol=1e-10)


def test_evaluate_registration_parity():
    upstream_source, upstream_target, mojo_source, mojo_target, truth = pair(seed=4)
    expected = o3d.pipelines.registration.evaluate_registration(
        upstream_source, upstream_target, 0.02, truth
    )
    actual = m3d.pipelines.registration.evaluate_registration(
        mojo_source, mojo_target, 0.02, truth
    )
    assert actual.fitness == pytest.approx(expected.fitness)
    assert actual.inlier_rmse == pytest.approx(expected.inlier_rmse, abs=1e-14)
    expected_pairs = np.asarray(expected.correspondence_set)
    assert {tuple(pair) for pair in actual.correspondence_set} == {
        tuple(pair) for pair in expected_pairs
    }


def test_icp_parity():
    upstream_source, upstream_target, mojo_source, mojo_target, _ = pair(seed=6)
    init = np.eye(4)
    init[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [0.04, -0.02, 0.06]
    )
    init[:3, 3] = [0.1, -0.05, 0.02]
    expected = o3d.pipelines.registration.registration_icp(
        upstream_source,
        upstream_target,
        0.5,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    actual = m3d.pipelines.registration.registration_icp(
        mojo_source,
        mojo_target,
        0.5,
        init,
        m3d.pipelines.registration.TransformationEstimationPointToPoint(),
        m3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )
    assert actual.fitness == pytest.approx(expected.fitness)
    assert actual.inlier_rmse == pytest.approx(expected.inlier_rmse, abs=1e-10)
    assert np.allclose(actual.transformation, expected.transformation, atol=1e-9)


def test_information_matrix_parity():
    upstream_source, upstream_target, mojo_source, mojo_target, truth = pair(seed=7)
    expected = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        upstream_source, upstream_target, 0.02, truth
    )
    actual = m3d.pipelines.registration.get_information_matrix_from_point_clouds(
        mojo_source, mojo_target, 0.02, truth
    )
    assert np.allclose(actual, expected, atol=1e-10)


def test_empty_registration():
    source = m3d.geometry.PointCloud()
    target = m3d.geometry.PointCloud()
    result = m3d.pipelines.registration.registration_icp(source, target, 1.0)
    assert result.fitness == 0
    assert result.inlier_rmse == 0
    assert result.correspondence_set.shape == (0, 2)
    assert np.array_equal(result.transformation, np.eye(4))


def test_correspondence_bounds_are_checked_before_ffi():
    _, _, source, target, _ = pair(count=3)
    estimator = m3d.pipelines.registration.TransformationEstimationPointToPoint()
    with pytest.raises(IndexError, match="outside"):
        estimator.compute_transformation(source, target, [[0, 99]])
    with pytest.raises(IndexError, match="outside"):
        estimator.compute_transformation(source, target, [[-1, 0]])
