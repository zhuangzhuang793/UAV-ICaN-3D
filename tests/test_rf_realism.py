import numpy as np

from uav_ican_3d.localization.rf_realism import (
    LinkBudget,
    correlated_shadowing_db,
    draw_sequence_state,
    link_budget_for_reference_snr,
    simulate_impaired_los_srs,
)
from uav_ican_3d.localization.rf_waveform import SRSWaveformConfig


def test_reference_link_budget_and_distance_slope() -> None:
    waveform = SRSWaveformConfig()
    budget = link_budget_for_reference_snr(
        50.0,
        0.0,
        waveform,
        tx_gain_dbi=0.0,
        rx_gain_dbi=0.0,
        receiver_noise_figure_db=7.0,
    )
    assert np.isclose(budget.snr_db(50.0, waveform), 0.0)
    assert np.isclose(budget.snr_db(100.0, waveform), -20.0 * np.log10(2.0))
    assert isinstance(budget, LinkBudget)


def test_shadowing_has_requested_scale() -> None:
    values = correlated_shadowing_db(100_000, 3.0, 0.95, np.random.default_rng(4))
    assert abs(np.std(values) - 3.0) < 0.1
    assert np.corrcoef(values[:-1], values[1:])[0, 1] > 0.93


def test_impaired_waveform_is_finite_and_has_expected_shape() -> None:
    waveform = SRSWaveformConfig()
    rng = np.random.default_rng(8)
    state = draw_sequence_state(
        rng,
        clock_bias_std_m=1.0,
        clock_drift_std_mps=0.02,
        residual_cfo_std_hz=2_000.0,
        array_amplitude_std_db=0.5,
        array_phase_std_deg=5.0,
    )
    received = simulate_impaired_los_srs(
        50.0,
        0.2,
        0.4,
        -2.0,
        waveform,
        rng,
        state=state,
        elapsed_s=1.0,
        range_rate_mps=5.0,
        enable_synchronization_errors=True,
        enable_array_errors=True,
        phase_noise_std_deg=3.0,
        enable_interference=True,
        interference_probability=1.0,
        interference_to_noise_db=-3.0,
    )
    assert received.shape == (32, 16)
    assert np.iscomplexobj(received)
    assert np.all(np.isfinite(received))
