import inspect

import numpy as np

from uav_ican_3d.localization import (
    SRSWaveformConfig,
    SionnaPathChannel,
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
