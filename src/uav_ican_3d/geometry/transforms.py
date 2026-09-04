"""Rigid transforms following the project-wide R_AB convention.

``R_AB`` maps a vector expressed in frame B into frame A. A transform ``T_AB``
therefore acts on a point as ``p_A = R_AB @ p_B + t_A_B``, where ``t_A_B`` is
the position of B's origin expressed in A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return array.copy()


def is_rotation_matrix(rotation: ArrayLike, atol: float = 1e-10) -> bool:
    """Return whether ``rotation`` is a proper 3-D rotation matrix."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False
    return bool(
        np.allclose(matrix.T @ matrix, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=atol, rtol=0.0)
    )


@dataclass(frozen=True)
class RigidTransform:
    """A rigid transform ``T_AB`` mapping points from frame B to frame A."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        if not is_rotation_matrix(rotation):
            raise ValueError("rotation must be a finite proper rotation matrix")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", _vector3(self.translation, "translation"))

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply_point(self, point_b: ArrayLike) -> FloatArray:
        """Transform a point from B into A."""

        return self.rotation @ _vector3(point_b, "point_b") + self.translation

    def apply_vector(self, vector_b: ArrayLike) -> FloatArray:
        """Rotate a free vector from B into A without applying translation."""

        return self.rotation @ _vector3(vector_b, "vector_b")

    def inverse(self) -> "RigidTransform":
        """Return ``T_BA``."""

        rotation_ba = self.rotation.T
        return RigidTransform(rotation_ba, -(rotation_ba @ self.translation))

    def compose(self, transform_bc: "RigidTransform") -> "RigidTransform":
        """Compose ``T_AB`` with ``T_BC`` and return ``T_AC``."""

        return RigidTransform(
            self.rotation @ transform_bc.rotation,
            self.rotation @ transform_bc.translation + self.translation,
        )

