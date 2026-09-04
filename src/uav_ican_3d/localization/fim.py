"""Fisher information and CRLB metrics with explicit observability checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rf import SingularGeometryError


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PositionBounds:
    covariance: FloatArray
    peb_3d_m: float
    peb_xy_m: float
    zeb_m: float
    fim_condition_number: float


def _symmetric_positive_definite(matrix: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    if not np.allclose(array, array.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(array)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return array


def fisher_information(jacobian: ArrayLike, measurement_covariance: ArrayLike) -> FloatArray:
    """Compute ``H.T @ R^-1 @ H`` without explicitly inverting R."""

    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be a finite matrix with three columns")
    covariance = _symmetric_positive_definite(
        measurement_covariance, matrix.shape[0], "measurement_covariance"
    )
    information = matrix.T @ np.linalg.solve(covariance, matrix)
    return 0.5 * (information + information.T)


def position_bounds(
    information: ArrayLike, rank_relative_tolerance: float = 1e-12
) -> PositionBounds:
    """Compute position CRLB metrics after rejecting singular information."""

    fim = np.asarray(information, dtype=float)
    if fim.shape != (3, 3) or not np.all(np.isfinite(fim)):
        raise ValueError("information must be a finite (3, 3) matrix")
    if not np.allclose(fim, fim.T, atol=1e-10, rtol=1e-10):
        raise ValueError("information must be symmetric")
    eigenvalues = np.linalg.eigvalsh(fim)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    if maximum <= 0.0 or minimum <= maximum * rank_relative_tolerance:
        raise SingularGeometryError(
            f"singular position FIM: eigenvalues={eigenvalues.tolist()}"
        )
    covariance = np.linalg.solve(fim, np.eye(3))
    covariance = 0.5 * (covariance + covariance.T)
    return PositionBounds(
        covariance=covariance,
        peb_3d_m=float(np.sqrt(np.trace(covariance))),
        peb_xy_m=float(np.sqrt(covariance[0, 0] + covariance[1, 1])),
        zeb_m=float(np.sqrt(covariance[2, 2])),
        fim_condition_number=float(maximum / minimum),
    )

