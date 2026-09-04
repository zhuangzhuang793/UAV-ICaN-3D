"""Calibration helpers for real detector boxes used as antenna-pixel measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VisualMeasurementCalibration:
    """Empirical detector-center bias and residual covariance relative to the UE pixel."""

    bias_uv: FloatArray
    covariance_uv: FloatArray
    matched_samples: int


def box_iou(box_xyxy: ArrayLike, boxes_xyxy: ArrayLike) -> FloatArray:
    """Return axis-aligned IoU between one box and each row of ``boxes_xyxy``."""

    box = np.asarray(box_xyxy, dtype=float)
    boxes = np.asarray(boxes_xyxy, dtype=float)
    if box.shape != (4,) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("box shapes must be (4,) and (N, 4)")
    if not np.all(np.isfinite(box)) or not np.all(np.isfinite(boxes)):
        raise ValueError("boxes must be finite")
    lower = np.maximum(boxes[:, :2], box[:2])
    upper = np.minimum(boxes[:, 2:], box[2:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=1)
    box_area = np.prod(np.maximum(box[2:] - box[:2], 0.0))
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0.0), axis=1)
    union = box_area + areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def best_box_match(
    gt_box_xyxy: ArrayLike, detector_boxes_xyxy: ArrayLike, minimum_iou: float
) -> int | None:
    """Return the highest-IoU detector index when it clears a declared threshold."""

    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must lie in [0, 1]")
    boxes = np.asarray(detector_boxes_xyxy, dtype=float)
    if boxes.shape == (0,):
        boxes = np.empty((0, 4), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detector_boxes_xyxy must have shape (N, 4)")
    if boxes.shape[0] == 0:
        return None
    overlaps = box_iou(gt_box_xyxy, boxes)
    index = int(np.argmax(overlaps))
    return index if overlaps[index] >= minimum_iou else None


def calibrate_bbox_center_to_ue_pixel(
    detector_boxes_by_frame: list[ArrayLike],
    gt_boxes_by_frame: list[ArrayLike],
    true_ue_pixels_by_frame: list[ArrayLike],
    minimum_iou: float,
    covariance_floor_px2: float,
) -> VisualMeasurementCalibration:
    """Fit an anisotropic residual model on a small synchronized validation subset.

    This does not equate a box center with the antenna.  It estimates and removes the empirical
    center-to-antenna bias, with uncertainty calibrated against the independently projected UE
    phase-center pixel.
    """

    frame_count = len(detector_boxes_by_frame)
    if not (
        frame_count == len(gt_boxes_by_frame) == len(true_ue_pixels_by_frame)
    ):
        raise ValueError("calibration inputs must contain the same number of frames")
    if covariance_floor_px2 <= 0.0:
        raise ValueError("covariance_floor_px2 must be positive")
    residuals = []
    for boxes, gt_box, true_pixel in zip(
        detector_boxes_by_frame,
        gt_boxes_by_frame,
        true_ue_pixels_by_frame,
        strict=True,
    ):
        detector_boxes = np.asarray(boxes, dtype=float)
        match = best_box_match(gt_box, detector_boxes, minimum_iou)
        if match is None:
            continue
        center = 0.5 * (detector_boxes[match, :2] + detector_boxes[match, 2:])
        residuals.append(center - np.asarray(true_pixel, dtype=float))
    if len(residuals) < 3:
        raise RuntimeError("at least three matched detector residuals are needed for calibration")
    residual_matrix = np.asarray(residuals, dtype=float)
    bias = np.mean(residual_matrix, axis=0)
    centered = residual_matrix - bias
    covariance = centered.T @ centered / (len(residuals) - 1)
    covariance += np.eye(2) * covariance_floor_px2
    return VisualMeasurementCalibration(bias, covariance, len(residuals))


def calibrated_box_centers(
    boxes_xyxy: ArrayLike, calibration: VisualMeasurementCalibration
) -> FloatArray:
    """Convert detector boxes to bias-corrected UE phase-center pixel candidates."""

    boxes = np.asarray(boxes_xyxy, dtype=float)
    if boxes.shape == (0,):
        boxes = np.empty((0, 4), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    return 0.5 * (boxes[:, :2] + boxes[:, 2:]) - calibration.bias_uv
