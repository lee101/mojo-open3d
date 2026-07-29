"""Open3D-compatible NumPy vector adapters."""

from __future__ import annotations

import numpy as np


def Vector3dVector(value=()) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(array.reshape((-1, 3)))


def Vector3iVector(value=()) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.ascontiguousarray(array.reshape((-1, 3)))


def Vector2iVector(value=()) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.ascontiguousarray(array.reshape((-1, 2)))


def DoubleVector(value=()) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).ravel()


def IntVector(value=()) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int32).ravel()
