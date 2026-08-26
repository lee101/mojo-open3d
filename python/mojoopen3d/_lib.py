"""ctypes bridge to the single mojo-open3d shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-open3d.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "m3d_kdtree_build": ([I, I, I, I, I, I, I], None),
    "m3d_kdtree_build_subtree": ([I, I, I, I, I, I], None),
    "m3d_kdtree_search": ([I, I, I, I, I, I, I, F, I, I, I, I], None),
    "m3d_voxel_aggregate": (
        [I, I, I, I, F, F, F, F] + [I] * 16,
        I,
    ),
    "m3d_transform_points": ([I, I, I, I], None),
    "m3d_rigid_estimate": ([I, I, I, I, I, I], I),
}

_library: ctypes.CDLL | None = None
_runtime: ctypes.CDLL | None = None
_parallel_ready = False


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "open3d.mojo")
    script = os.path.join(ROOT, "build", "build.sh")
    stale = not os.path.exists(LIB_PATH) or os.path.getmtime(LIB_PATH) < os.path.getmtime(source)
    if force or stale:
        result = subprocess.run(
            ["bash", script], cwd=ROOT, text=True, capture_output=True, timeout=1800
        )
        if result.returncode or not os.path.exists(LIB_PATH):
            raise RuntimeError((result.stderr or result.stdout).strip())
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library, _parallel_ready, _runtime
    if _library is None:
        _library = ctypes.CDLL(build())
        try:
            _runtime = ctypes.CDLL("libKGENCompilerRTShared.so")
            initialize = _runtime.KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
            initialize.argtypes = []
            initialize.restype = ctypes.c_void_p
            _parallel_ready = bool(initialize())
        except (AttributeError, OSError):
            _parallel_ready = False
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def parallel_ready() -> bool:
    lib()
    return _parallel_ready


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if array.size == 0:
        raise ValueError("empty arrays do not have a usable FFI buffer")
    return int(array.ctypes.data)


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    points = f64(points)
    transformation = f64(transformation)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points must have shape (n, 3)")
    if transformation.shape != (4, 4):
        raise ValueError("transformation must have shape (4, 4)")
    result = np.empty_like(points)
    if len(points):
        lib().m3d_transform_points(
            addr(points), addr(transformation), addr(result), len(points)
        )
    return result


def rigid_estimate(
    source: np.ndarray,
    target: np.ndarray,
    target_indices: np.ndarray,
    with_scaling: bool = False,
) -> np.ndarray:
    source = f64(source)
    target = f64(target)
    target_indices = i64(target_indices)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("source must have shape (n, 3)")
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target must have shape (n, 3)")
    if target_indices.shape != (len(source),):
        raise ValueError("target_indices must have one entry per source point")
    if np.any(target_indices < -1) or np.any(target_indices >= len(target)):
        raise IndexError("target index is outside the target point cloud")
    result = np.eye(4, dtype=np.float64)
    if len(source):
        count = lib().m3d_rigid_estimate(
            addr(source),
            addr(target),
            addr(target_indices),
            len(source),
            int(with_scaling),
            addr(result),
        )
        expected = int(np.count_nonzero(target_indices >= 0))
        if count != expected:
            raise RuntimeError(
                f"rigid-estimation kernel processed {count} of {expected} pairs"
            )
    return result
