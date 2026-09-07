"""Controlled RF impairments for the post-F7 realism audit.

The original F7 waveform is deliberately left unchanged.  This module adds a
cumulative ladder of channel and hardware effects for diagnostic experiments.
All functions consume physical observations and emit only complex waveforms;
no path or localization truth is exposed to the online estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rf_waveform import (
    SPEED_OF_LIGHT_MPS,
    ComplexArray,
    SRSWaveformConfig,
    srs_reference,
    upa_steering,
)


FloatArray = NDArray[np.float64]
THERMAL_NOISE_DBM_PER_HZ = -174.0


@dataclass(frozen=True)
class LinkBudget:
    """Per-subcarrier free-space link budget.

    ``tx_power_total_dbm`` is the power across all occupied subcarriers.  The
    split is explicit so that changing the subcarrier count cannot silently
    create processing gain in the link budget.
    """

    tx_power_total_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    receiver_noise_figure_db: float
    system_loss_db: float = 0.0

    def noise_power_per_subcarrier_dbm(self, config: SRSWaveformConfig) -> float:
        return (
            THERMAL_NOISE_DBM_PER_HZ
            + 10.0 * np.log10(config.subcarrier_spacing_hz)
            + self.receiver_noise_figure_db
        )

    def snr_db(self, range_m: ArrayLike, config: SRSWaveformConfig) -> FloatArray:
        distance = np.asarray(range_m, dtype=float)
        if np.any(distance <= 0.0):
            raise ValueError("range must be positive")
        wavelength = SPEED_OF_LIGHT_MPS / config.carrier_hz
        fspl_db = 20.0 * np.log10(4.0 * np.pi * distance / wavelength)
        tx_per_subcarrier_dbm = self.tx_power_total_dbm - 10.0 * np.log10(
            config.subcarrier_count
        )
        received_dbm = (
            tx_per_subcarrier_dbm
            + self.tx_gain_dbi
            + self.rx_gain_dbi
            - fspl_db
            - self.system_loss_db
        )
        return received_dbm - self.noise_power_per_subcarrier_dbm(config)


def link_budget_for_reference_snr(
    reference_range_m: float,
    reference_snr_db: float,
    config: SRSWaveformConfig,
    *,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    receiver_noise_figure_db: float,
    system_loss_db: float = 0.0,
) -> LinkBudget:
    """Return the total pilot power required by a declared reference point."""

    provisional = LinkBudget(
        0.0,
        tx_gain_dbi,
        rx_gain_dbi,
        receiver_noise_figure_db,
        system_loss_db,
    )
    snr_at_zero_dbm = float(provisional.snr_db(reference_range_m, config))
    return LinkBudget(
        reference_snr_db - snr_at_zero_dbm,
        tx_gain_dbi,
        rx_gain_dbi,
        receiver_noise_figure_db,
        system_loss_db,
    )


@dataclass(frozen=True)
class RFSequenceState:
    """Slowly varying hardware errors held fixed within one sequence/seed."""

    clock_bias_m: float
    clock_drift_mps: float
    residual_cfo_hz: float
    array_amplitude_db: FloatArray
    array_phase_rad: FloatArray

    def __post_init__(self) -> None:
        if np.asarray(self.array_amplitude_db).shape != (16,):
            raise ValueError("array amplitude errors must have 16 entries")
        if np.asarray(self.array_phase_rad).shape != (16,):
            raise ValueError("array phase errors must have 16 entries")


def draw_sequence_state(
    rng: np.random.Generator,
    *,
    clock_bias_std_m: float,
    clock_drift_std_mps: float,
    residual_cfo_std_hz: float,
    array_amplitude_std_db: float,
    array_phase_std_deg: float,
) -> RFSequenceState:
    return RFSequenceState(
        clock_bias_m=float(rng.normal(scale=clock_bias_std_m)),
        clock_drift_mps=float(rng.normal(scale=clock_drift_std_mps)),
        residual_cfo_hz=float(rng.normal(scale=residual_cfo_std_hz)),
        array_amplitude_db=rng.normal(scale=array_amplitude_std_db, size=16),
        array_phase_rad=np.deg2rad(rng.normal(scale=array_phase_std_deg, size=16)),
    )


def correlated_shadowing_db(
    count: int, sigma_db: float, correlation: float, rng: np.random.Generator
) -> FloatArray:
    if count < 1 or sigma_db < 0.0 or not 0.0 <= correlation < 1.0:
        raise ValueError("invalid shadowing parameters")
    values = np.empty(count, dtype=float)
    values[0] = rng.normal(scale=sigma_db)
    innovation_std = sigma_db * np.sqrt(1.0 - correlation**2)
    for index in range(1, count):
        values[index] = correlation * values[index - 1] + rng.normal(scale=innovation_std)
    return values


def _apply_cfo_ici(samples: ComplexArray, normalized_cfo: float) -> ComplexArray:
    """Apply exact rectangular-window OFDM inter-carrier interference."""

    if np.isclose(normalized_cfo, 0.0):
        return samples
    subcarriers = samples.shape[0]
    time_samples = np.fft.ifft(samples, axis=0, norm="ortho")
    phase = np.exp(1j * 2.0 * np.pi * normalized_cfo * np.arange(subcarriers) / subcarriers)
    return np.fft.fft(time_samples * phase[:, None], axis=0, norm="ortho")


def simulate_impaired_los_srs(
    true_range_m: float,
    true_azimuth_rad: float,
    true_elevation_rad: float,
    snr_db: float,
    config: SRSWaveformConfig,
    rng: np.random.Generator,
    *,
    state: RFSequenceState,
    elapsed_s: float,
    range_rate_mps: float,
    enable_synchronization_errors: bool,
    enable_array_errors: bool,
    phase_noise_std_deg: float,
    enable_interference: bool,
    interference_probability: float,
    interference_to_noise_db: float,
    reference: ComplexArray | None = None,
) -> ComplexArray:
    """Generate a LoS SRS waveform with cumulative, explicitly enabled effects."""

    if true_range_m <= 0.0 or not np.isfinite(snr_db):
        raise ValueError("range and SNR must be valid")
    pilots = srs_reference(config.subcarrier_count) if reference is None else reference
    apparent_range_m = float(true_range_m)
    if enable_synchronization_errors:
        apparent_range_m += state.clock_bias_m + state.clock_drift_mps * elapsed_s
    if apparent_range_m <= 0.0:
        raise RuntimeError("clock impairment produced a nonpositive apparent range")
    delay_phase = np.exp(
        -1j
        * 2.0
        * np.pi
        * config.frequency_offsets_hz
        * (apparent_range_m / SPEED_OF_LIGHT_MPS)
    )
    spatial = upa_steering(true_azimuth_rad, true_elevation_rad)
    desired = pilots[:, None] * delay_phase[:, None] * spatial[None, :]

    if enable_array_errors:
        gains = 10.0 ** (state.array_amplitude_db / 20.0) * np.exp(
            1j * state.array_phase_rad
        )
        desired = desired * gains[None, :]

    if enable_synchronization_errors:
        doppler_hz = -range_rate_mps * config.carrier_hz / SPEED_OF_LIGHT_MPS
        normalized_cfo = (state.residual_cfo_hz + doppler_hz) / config.subcarrier_spacing_hz
        desired = _apply_cfo_ici(desired, normalized_cfo)
        phase_noise = np.deg2rad(phase_noise_std_deg) * rng.normal(
            size=config.subcarrier_count
        )
        desired = desired * np.exp(1j * phase_noise[:, None])

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_std = 1.0 / np.sqrt(snr_linear)
    received = desired.copy()
    if enable_interference and rng.random() < interference_probability:
        # Co-pilot interference is intentionally a declared stress condition.
        interferer_range = max(10.0, true_range_m + rng.uniform(-20.0, 20.0))
        interferer_azimuth = true_azimuth_rad + rng.uniform(-np.pi / 3.0, np.pi / 3.0)
        interferer_elevation = np.clip(
            true_elevation_rad + rng.uniform(-np.pi / 6.0, np.pi / 6.0),
            -np.pi / 2.0,
            np.pi / 2.0,
        )
        interferer_delay = np.exp(
            -1j
            * 2.0
            * np.pi
            * config.frequency_offsets_hz
            * (interferer_range / SPEED_OF_LIGHT_MPS)
        )
        interferer = (
            pilots[:, None]
            * interferer_delay[:, None]
            * upa_steering(interferer_azimuth, interferer_elevation)[None, :]
        )
        inr_linear = 10.0 ** (interference_to_noise_db / 10.0)
        received += noise_std * np.sqrt(inr_linear) * interferer
    noise = (
        rng.normal(size=received.shape) + 1j * rng.normal(size=received.shape)
    ) * (noise_std / np.sqrt(2.0))
    return received + noise
