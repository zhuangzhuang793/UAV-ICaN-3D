"""Frozen C0/C1 detector-box to antenna-pixel calibration models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def obb_viewpoint_features(boxes_xywhr: ArrayLike, image_size: tuple[int, int]) -> FloatArray:
    """Return scale-stable linear features derived only from OBB geometry."""

    boxes = np.asarray(boxes_xywhr, dtype=float)
    if boxes.shape == (0,):
        boxes = np.empty((0, 5), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not np.all(np.isfinite(boxes)):
        raise ValueError("boxes_xywhr must have finite shape (N, 5)")
    width_px, height_px = map(float, image_size)
    if width_px <= 0.0 or height_px <= 0.0 or np.any(boxes[:, 2:4] <= 0.0):
        raise ValueError("image size and OBB dimensions must be positive")
    angle = boxes[:, 4]
    return np.column_stack(
        (
            boxes[:, 0] / width_px,
            boxes[:, 1] / height_px,
            np.log(boxes[:, 2]),
            np.log(boxes[:, 3]),
            np.log(boxes[:, 2] / boxes[:, 3]),
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
        )
    )


@dataclass(frozen=True)
class FrozenVisualCalibration:
    """A constant-bias or ridge OBB correction with frozen pixel covariance."""

    kind: str
    covariance_uv: FloatArray
    correction_intercept_uv: FloatArray
    correction_coefficients: FloatArray
    feature_mean: FloatArray
    feature_scale: FloatArray
    matched_samples: int

    def corrected_pixels(
        self, boxes_xywhr: ArrayLike, image_size: tuple[int, int]
    ) -> FloatArray:
        boxes = np.asarray(boxes_xywhr, dtype=float)
        features = obb_viewpoint_features(boxes, image_size)
        centered = (features - self.feature_mean) / self.feature_scale
        correction = self.correction_intercept_uv + centered @ self.correction_coefficients
        return boxes[:, :2] + correction

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "covariance_uv_px2": self.covariance_uv.tolist(),
            "correction_intercept_uv_px": self.correction_intercept_uv.tolist(),
            "center_minus_antenna_bias_uv_px": (-self.correction_intercept_uv).tolist(),
            "correction_coefficients": self.correction_coefficients.tolist(),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "matched_samples": self.matched_samples,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "FrozenVisualCalibration":
        return cls(
            kind=str(values["kind"]),
            covariance_uv=np.asarray(values["covariance_uv_px2"], dtype=float),
            correction_intercept_uv=np.asarray(
                values["correction_intercept_uv_px"], dtype=float
            ),
            correction_coefficients=np.asarray(values["correction_coefficients"], dtype=float),
            feature_mean=np.asarray(values["feature_mean"], dtype=float),
            feature_scale=np.asarray(values["feature_scale"], dtype=float),
            matched_samples=int(values["matched_samples"]),
        )


def _covariance(errors: FloatArray, floor_px2: float) -> FloatArray:
    if errors.shape[0] < 3 or floor_px2 <= 0.0:
        raise ValueError("at least three errors and a positive covariance floor are required")
    centered = errors - np.mean(errors, axis=0)
    covariance = centered.T @ centered / (errors.shape[0] - 1)
    covariance += np.eye(2) * floor_px2
    return covariance


def fit_visual_calibrations(
    boxes_xywhr: ArrayLike,
    antenna_pixels_uv: ArrayLike,
    image_size: tuple[int, int],
    ridge_alpha: float,
    covariance_floor_px2: float,
) -> tuple[FrozenVisualCalibration, FrozenVisualCalibration]:
    """Fit C0 constant correction and C1 ridge correction on calibration samples."""

    boxes = np.asarray(boxes_xywhr, dtype=float)
    truth = np.asarray(antenna_pixels_uv, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 5 or truth.shape != (boxes.shape[0], 2):
        raise ValueError("calibration arrays have incompatible shapes")
    if boxes.shape[0] < 3 or ridge_alpha < 0.0:
        raise ValueError("calibration needs at least three samples and nonnegative ridge alpha")
    features = obb_viewpoint_features(boxes, image_size)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0, ddof=1)
    feature_scale = np.maximum(feature_scale, 1e-8)
    standardized = (features - feature_mean) / feature_scale
    correction_truth = truth - boxes[:, :2]
    intercept = np.mean(correction_truth, axis=0)

    zero_coefficients = np.zeros((features.shape[1], 2), dtype=float)
    c0_errors = boxes[:, :2] + intercept - truth
    c0 = FrozenVisualCalibration(
        "C0",
        _covariance(c0_errors, covariance_floor_px2),
        intercept,
        zero_coefficients,
        feature_mean,
        feature_scale,
        boxes.shape[0],
    )

    target = correction_truth - intercept
    gram = standardized.T @ standardized + ridge_alpha * np.eye(features.shape[1])
    coefficients = np.linalg.solve(gram, standardized.T @ target)
    c1_errors = boxes[:, :2] + intercept + standardized @ coefficients - truth
    c1 = FrozenVisualCalibration(
        "C1",
        _covariance(c1_errors, covariance_floor_px2),
        intercept,
        coefficients,
        feature_mean,
        feature_scale,
        boxes.shape[0],
    )
    return c0, c1


def mean_pixel_nll(errors_uv: ArrayLike, covariance_uv: ArrayLike) -> float:
    """Return the mean two-dimensional Gaussian negative log likelihood."""

    errors = np.asarray(errors_uv, dtype=float)
    covariance = np.asarray(covariance_uv, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 2 or covariance.shape != (2, 2):
        raise ValueError("errors and covariance have incompatible shapes")
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("covariance must be finite and positive definite")
    mahalanobis = np.einsum(
        "ni,ij,nj->n", errors, np.linalg.inv(covariance), errors
    )
    return float(np.mean(np.log(2.0 * np.pi) + 0.5 * log_determinant + 0.5 * mahalanobis))
