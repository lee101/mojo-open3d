# mojo-open3d

`mojo-open3d` is a standalone Mojo port of the compute-heavy point-cloud core
of [Open3D](https://www.open3d.org/). It provides a NumPy-friendly Python API
for a documented subset of Open3D's legacy `geometry` and
`pipelines.registration` namespaces.

The Python package is named `mojoopen3d`, so switching a covered program is
normally one import change:

```python
import mojoopen3d as o3d
```

This is a focused port, not a reimplementation of the entire Open3D project.
The test suite exercises every listed operation and includes numerical parity
checks against Open3D 0.19.0.

## Covered subset

| Open3D namespace | Covered API |
| --- | --- |
| `utility` | `Vector3dVector`, `Vector3iVector`, `Vector2iVector`, `DoubleVector`, `IntVector` |
| `geometry.PointCloud` | bounds and center, transforms, selection, cloud addition, uniform/random sampling, duplicate/non-finite removal, KNN normal estimation and orientation, uniform color, voxel downsampling and tracing, cloud/nearest-neighbor distances, radius/statistical outlier removal |
| `geometry` KD-tree | `KDTreeFlann`, matrix or point-cloud construction, KNN, radius, hybrid, and parameter-dispatched 3D search |
| `geometry.VoxelGrid` | point-cloud construction, bounded-origin construction, voxel lookup, inclusion tests, bounds, colors, and voxel enumeration |
| `geometry` rotations | rotation matrices from XYZ angles, axis-angle, and quaternion |
| `pipelines.registration` | `TransformationEstimationPointToPoint` with optional scaling, `evaluate_registration`, point-to-point `registration_icp`, convergence criteria, registration results, and information matrices |

The KD-tree is a real balanced median-split tree built and searched in Mojo.
Voxel downsampling uses an open-addressed hash table and accumulates points,
colors, normals, counts, and trace IDs in one pass. Point-to-point registration
uses Mojo KD-tree correspondence search and Horn's quaternion rigid transform.

Not covered are meshes, images, RGB-D, file I/O, visualization, tensor/CUDA
APIs, FPFH and global registration, point-to-plane ICP, colored ICP,
generalized ICP, reconstruction, and Open3D-ML. Passing a point-to-plane
estimator to the covered ICP entry point is rejected instead of silently
changing the algorithm.

## Install

```bash
pixi install
pixi run build
```

Pixi installs the pinned Mojo nightly, Python, NumPy, pytest, and Open3D 0.19
used by the parity suite. Build output is
`dist/libmojo-open3d.so`.

## Usage

This example runs as written after `pixi run build`:

```python
import numpy as np
import mojoopen3d as o3d

rng = np.random.default_rng(0)
points = rng.normal(size=(20_000, 3))

source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
target = o3d.geometry.PointCloud(
    o3d.utility.Vector3dVector(points + np.array([0.05, -0.02, 0.01]))
)

coarse = source.voxel_down_sample(0.08)
tree = o3d.geometry.KDTreeFlann(coarse)
count, indices, squared_distances = tree.search_knn_vector_3d([0, 0, 0], 8)

result = o3d.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.2
)
print(count, result.fitness, result.transformation)
```

Run it inside the environment with `pixi run python example.py`.

## Correctness

```bash
pixi run build
pixi run test
```

The suite contains 33 tests against Open3D 0.19.0. It checks neighbor indices
and squared distances, voxel membership and averaged attributes, trace corner
IDs, point-cloud operations, outlier selections, rigid transforms with and
without scale, correspondence metrics, ICP convergence, and information
matrices. The test task pins OpenMP to one thread because Open3D 0.19's legacy
radius-outlier loop is nondeterministic under this environment's OpenMP
runtime; benchmarks use Open3D's normal parallel settings.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of three warmed runs on identical arrays.
These are the real results from this repository.

| case | mojo-open3d | Open3D 0.19 | result |
| --- | ---: | ---: | ---: |
| Voxel downsample (1M points) | 308.55 ms | 795.21 ms | 2.58x faster |
| KD-tree build (250k points) | 37.79 ms | 108.12 ms | 2.86x faster |
| KNN k=8 (250k x 25k) | 7.93 ms | 122.76 ms | 15.48x faster |
| Cloud distance (100k to 250k) | 64.71 ms | 119.91 ms | 1.85x faster |
| Rigid estimate (500k pairs) | 9.25 ms | 18.22 ms | 1.97x faster |
| Point-to-point ICP (30k points) | 34.77 ms | 69.63 ms | 2.00x faster |

Large independent query batches and large tree builds use CPU parallelism;
smaller jobs remain serial to avoid launch overhead. Search-result and
tree-buffer initialization use native-width SIMD stores with scalar remainder
loops. There is no GPU path.

## How it works

All Mojo kernels live in one compilation unit, `src/open3d.mojo`, which is
built once as a shared library. A thin `ctypes` layer makes one C ABI call per
bulk operation.

NumPy owns every allocation. Point and attribute matrices are contiguous
row-major `float64` arrays with shape `(n, 3)`; tree nodes, voxel hash slots,
correspondences, and trace IDs are contiguous `int64` arrays. Buffers cross
the ABI as integer addresses because exported Mojo functions cannot have an
inferred pointer origin. Each exported wrapper reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` internally. Mojo never retains a
pointer or allocates caller-visible memory, so lifetimes remain entirely on
the Python side.

## License

MIT
