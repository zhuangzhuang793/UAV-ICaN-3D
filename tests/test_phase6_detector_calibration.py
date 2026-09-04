import numpy as np

from uav_ican_3d.vision import (
    best_box_match,
    box_iou,
    calibrate_bbox_center_to_ue_pixel,
    calibrated_box_centers,
)


def test_box_iou_and_thresholded_match() -> None:
    gt = np.array([10.0, 10.0, 30.0, 30.0])
    boxes = np.array([[0.0, 0.0, 5.0, 5.0], [12.0, 12.0, 28.0, 28.0]])
    overlaps = box_iou(gt, boxes)
    assert np.allclose(overlaps, [0.0, 0.64])
    assert best_box_match(gt, boxes, 0.5) == 1
    assert best_box_match(gt, boxes, 0.7) is None


def test_empirical_calibration_removes_bias_and_keeps_anisotropy() -> None:
    true_pixels = [np.array([100.0 + index, 80.0]) for index in range(4)]
    residuals = [
        np.array([4.0, -1.0]),
        np.array([6.0, 1.0]),
        np.array([5.0, -2.0]),
        np.array([5.0, 2.0]),
    ]
    detector_boxes = []
    gt_boxes = []
    for pixel, residual in zip(true_pixels, residuals, strict=True):
        center = pixel + residual
        detector_boxes.append(
            np.array([[center[0] - 5, center[1] - 4, center[0] + 5, center[1] + 4]])
        )
        gt_boxes.append(np.array([pixel[0] - 8, pixel[1] - 8, pixel[0] + 8, pixel[1] + 8]))
    calibration = calibrate_bbox_center_to_ue_pixel(
        detector_boxes, gt_boxes, true_pixels, minimum_iou=0.2, covariance_floor_px2=0.1
    )
    assert np.allclose(calibration.bias_uv, [5.0, 0.0])
    assert calibration.covariance_uv[1, 1] > calibration.covariance_uv[0, 0]
    corrected = calibrated_box_centers(detector_boxes[0], calibration)
    assert np.allclose(corrected[0], true_pixels[0] + [-1.0, -1.0])
