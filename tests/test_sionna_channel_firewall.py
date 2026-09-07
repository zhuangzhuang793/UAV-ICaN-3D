import inspect

import numpy as np

from uav_ican_3d.localization import (
    SRSWaveformConfig,
    SionnaPathChannel,
    sionna_received_power_gain,
    synthesize_sionna_waveform,
)


def test_path_truth_is_consumed_into_waveform_before_estimation() -> None:
    coefficients = np.ones((16, 1), dtype=np.complex128)
    channel = SionnaPathChannel(coefficients, np.array([100e-9]))
    waveform = synthesize_sionna_waveform(
        channel, 20.0, SRSWaveformConfig(), np.random.default_rng(1)
    )
    assert waveform.shape == (32, 16)
    assert np.iscomplexobj(waveform)


def test_sionna_truth_fields_are_absent_from_waveform_estimator_signature() -> None:
    from uav_ican_3d.localization import WaveformRFEstimator

    parameters = set(inspect.signature(WaveformRFEstimator.estimate).parameters)
    assert not {"tau", "delay", "aoa", "path_angles", "coefficients"} & parameters


def test_absolute_power_mode_preserves_raw_channel_gain() -> None:
    config = SRSWaveformConfig()
    weak = SionnaPathChannel(np.full((16, 1), 1e-4 + 0j), np.array([100e-9]))
    strong = SionnaPathChannel(np.full((16, 1), 2e-4 + 0j), np.array([100e-9]))
    assert np.isclose(
        sionna_received_power_gain(strong, config),
        4.0 * sionna_received_power_gain(weak, config),
    )
    weak_waveform = synthesize_sionna_waveform(
        weak,
        0.0,
        config,
        np.random.default_rng(1),
        normalize_received_power=False,
        transmit_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1e-30,
    )
    strong_waveform = synthesize_sionna_waveform(
        strong,
        0.0,
        config,
        np.random.default_rng(1),
        normalize_received_power=False,
        transmit_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1e-30,
    )
    assert np.isclose(np.mean(np.abs(strong_waveform) ** 2), 4.0 * np.mean(np.abs(weak_waveform) ** 2))
