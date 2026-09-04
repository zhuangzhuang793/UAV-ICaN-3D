import numpy as np

from uav_ican_3d.vision import fit_visual_calibrations, mean_pixel_nll


def test_ridge_viewpoint_model_reduces_structured_pixel_error() -> None:
    rng = np.random.default_rng(12)
    count = 100
    boxes = np.column_stack(
        (
            rng.uniform(100, 540, count),
            rng.uniform(80, 400, count),
            rng.uniform(20, 80, count),
            rng.uniform(10, 40, count),
            rng.uniform(0, np.pi, count),
        )
    )
    correction = np.column_stack((0.08 * boxes[:, 2], -0.12 * boxes[:, 3]))
    truth = boxes[:, :2] + correction
    c0, c1 = fit_visual_calibrations(boxes, truth, (640, 480), 1.0, 1.0)
    error0 = c0.corrected_pixels(boxes, (640, 480)) - truth
    error1 = c1.corrected_pixels(boxes, (640, 480)) - truth
    assert np.sqrt(np.mean(error1**2)) < np.sqrt(np.mean(error0**2))
    assert mean_pixel_nll(error1, c1.covariance_uv) < mean_pixel_nll(
        error0, c0.covariance_uv
    )
    assert np.linalg.eigvalsh(c1.covariance_uv)[0] > 0.0


def test_c0_is_preferred_when_ridge_has_no_validation_advantage() -> None:
    boxes = np.tile(np.array([320.0, 240.0, 40.0, 20.0, 0.2]), (8, 1))
    boxes[:, 0] += np.arange(8)
    truth = boxes[:, :2] + np.array([3.0, -2.0])
    c0, c1 = fit_visual_calibrations(boxes, truth, (640, 480), 1.0, 1.0)
    assert np.allclose(c0.corrected_pixels(boxes, (640, 480)), truth)
    assert np.allclose(c1.corrected_pixels(boxes, (640, 480)), truth)
