import numpy as np

from uav_ican_3d.localization import (
    SRSWaveformConfig,
    WaveformRFEstimator,
    simulate_los_srs,
    upa_steering,
)


def _estimator() -> WaveformRFEstimator:
    return WaveformRFEstimator(
        SRSWaveformConfig(),
        np.arange(20.0, 60.01, 0.1),
        np.deg2rad(np.arange(-45.0, 45.01, 0.5)),
        np.deg2rad(np.arange(-20.0, 30.01, 0.5)),
    )


def test_upa_steering_has_sixteen_unit_magnitude_elements() -> None:
    steering = upa_steering(0.2, 0.1)
    assert steering.shape == (16,)
    assert np.allclose(np.abs(steering), 1.0)


def test_noiseless_like_waveform_recovers_delay_and_two_angles() -> None:
    config = SRSWaveformConfig()
    waveform = simulate_los_srs(
        34.2, np.deg2rad(25.0), np.deg2rad(5.0), 80.0, config, np.random.default_rng(4)
    )
    estimate = _estimator().estimate(waveform)
    assert abs(estimate.range_m - 34.2) <= 0.11
    assert abs(estimate.azimuth_rad - np.deg2rad(25.0)) <= np.deg2rad(0.51)
    assert abs(estimate.elevation_rad - np.deg2rad(5.0)) <= np.deg2rad(0.51)


def test_waveform_estimator_does_not_accept_ground_truth() -> None:
    estimator = _estimator()
    assert set(estimator.estimate.__annotations__) == {"received", "return"}
