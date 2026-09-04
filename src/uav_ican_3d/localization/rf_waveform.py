"""SRS-like LoS waveform simulation and 4x4 UPA delay/2-D-AoA estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rf import wrap_angle


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True)
class SRSWaveformConfig:
    carrier_hz: float = 3.5e9
    subcarrier_count: int = 32
    subcarrier_spacing_hz: float = 480e3
    upa_rows: int = 4
    upa_columns: int = 4

    def __post_init__(self) -> None:
        if self.carrier_hz <= 0.0 or self.subcarrier_count < 8:
            raise ValueError("invalid carrier or subcarrier configuration")
        if self.subcarrier_spacing_hz <= 0.0:
            raise ValueError("subcarrier spacing must be positive")
        if (self.upa_rows, self.upa_columns) != (4, 4):
            raise ValueError("the project uses one fixed 4x4 UPA")

    @property
    def frequency_offsets_hz(self) -> FloatArray:
        indices = np.arange(self.subcarrier_count) - 0.5 * (self.subcarrier_count - 1)
        return indices * self.subcarrier_spacing_hz


@dataclass(frozen=True)
class RFEstimate:
    range_m: float
    azimuth_rad: float
    elevation_rad: float

    @property
    def vector(self) -> FloatArray:
        return np.array([self.range_m, self.azimuth_rad, self.elevation_rad])


def srs_reference(subcarrier_count: int, seed: int = 17) -> ComplexArray:
    """Return a deterministic unit-power QPSK reference sequence."""

    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, 4, size=subcarrier_count)
    return np.exp(1j * (np.pi / 4.0 + symbols * np.pi / 2.0))


def upa_steering(
    azimuth_rad: ArrayLike,
    elevation_rad: ArrayLike,
    rows: int = 4,
    columns: int = 4,
) -> ComplexArray:
    """Return half-wavelength UPA steering vectors with array plane on y/z."""

    azimuth = np.asarray(azimuth_rad, dtype=float)
    elevation = np.asarray(elevation_rad, dtype=float)
    azimuth, elevation = np.broadcast_arrays(azimuth, elevation)
    direction_y = np.cos(elevation) * np.sin(azimuth)
    direction_z = np.sin(elevation)
    row, column = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    phase = np.pi * (
        direction_y[..., None] * row.reshape(-1)
        + direction_z[..., None] * column.reshape(-1)
    )
    return np.exp(-1j * phase)


def simulate_los_srs(
    true_range_m: float,
    true_azimuth_rad: float,
    true_elevation_rad: float,
    snr_db: float,
    config: SRSWaveformConfig,
    rng: np.random.Generator,
    reference: ComplexArray | None = None,
) -> ComplexArray:
    """Generate received frequency-domain samples; truth is used only by this channel simulator."""

    if true_range_m <= 0.0 or not np.isfinite(snr_db):
        raise ValueError("range and SNR must be valid")
    pilots = srs_reference(config.subcarrier_count) if reference is None else reference
    if pilots.shape != (config.subcarrier_count,):
        raise ValueError("reference has wrong shape")
    delay_s = true_range_m / SPEED_OF_LIGHT_MPS
    delay_phase = np.exp(-1j * 2.0 * np.pi * config.frequency_offsets_hz * delay_s)
    spatial = upa_steering(true_azimuth_rad, true_elevation_rad)
    noiseless = pilots[:, None] * delay_phase[:, None] * spatial[None, :]
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise = (
        rng.normal(size=noiseless.shape) + 1j * rng.normal(size=noiseless.shape)
    ) / np.sqrt(2.0 * snr_linear)
    return noiseless + noise


class WaveformRFEstimator:
    """Grid maximum-likelihood delay followed by 2-D spatial maximum likelihood."""

    def __init__(
        self,
        config: SRSWaveformConfig,
        range_grid_m: ArrayLike,
        azimuth_grid_rad: ArrayLike,
        elevation_grid_rad: ArrayLike,
        reference: ComplexArray | None = None,
    ) -> None:
        self.config = config
        self.reference = (
            srs_reference(config.subcarrier_count) if reference is None else np.asarray(reference)
        )
        self.range_grid_m = np.asarray(range_grid_m, dtype=float)
        self.azimuth_grid_rad = np.asarray(azimuth_grid_rad, dtype=float)
        self.elevation_grid_rad = np.asarray(elevation_grid_rad, dtype=float)
        if any(grid.ndim != 1 or grid.size < 2 for grid in self._grids):
            raise ValueError("estimator grids must be nontrivial vectors")
        delays = self.range_grid_m / SPEED_OF_LIGHT_MPS
        self.delay_dictionary = np.exp(
            1j * 2.0 * np.pi * delays[:, None] * config.frequency_offsets_hz[None, :]
        )
        azimuth_mesh, elevation_mesh = np.meshgrid(
            self.azimuth_grid_rad, self.elevation_grid_rad, indexing="ij"
        )
        self.angle_pairs = np.column_stack(
            (azimuth_mesh.reshape(-1), elevation_mesh.reshape(-1))
        )
        self.spatial_dictionary = upa_steering(
            self.angle_pairs[:, 0], self.angle_pairs[:, 1]
        )

    @property
    def _grids(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        return self.range_grid_m, self.azimuth_grid_rad, self.elevation_grid_rad

    def estimate(self, received: ArrayLike) -> RFEstimate:
        samples = np.asarray(received, dtype=np.complex128)
        antenna_count = self.config.upa_rows * self.config.upa_columns
        if samples.shape != (self.config.subcarrier_count, antenna_count):
            raise ValueError("received waveform has wrong shape")
        despread_reference = samples[:, 0] * np.conj(self.reference)
        delay_scores = np.abs(self.delay_dictionary @ despread_reference) ** 2
        range_index = int(np.argmax(delay_scores))
        range_m = float(self.range_grid_m[range_index])
        delay_s = range_m / SPEED_OF_LIGHT_MPS
        compensated = samples * np.conj(self.reference[:, None]) * np.exp(
            1j * 2.0 * np.pi * self.config.frequency_offsets_hz[:, None] * delay_s
        )
        channel = np.mean(compensated, axis=0)
        angle_scores = np.abs(np.conj(self.spatial_dictionary) @ channel) ** 2
        angle_index = int(np.argmax(angle_scores))
        return RFEstimate(
            range_m,
            float(wrap_angle(self.angle_pairs[angle_index, 0])),
            float(self.angle_pairs[angle_index, 1]),
        )

    def estimate_batch(self, received: ArrayLike) -> list[RFEstimate]:
        """Vectorized equivalent of :meth:`estimate` for independent waveforms."""

        samples = np.asarray(received, dtype=np.complex128)
        antenna_count = self.config.upa_rows * self.config.upa_columns
        expected_tail = (self.config.subcarrier_count, antenna_count)
        if samples.ndim != 3 or samples.shape[1:] != expected_tail:
            raise ValueError("received batch has wrong shape")
        despread = samples[:, :, 0] * np.conj(self.reference)[None, :]
        delay_scores = np.abs(self.delay_dictionary @ despread.T) ** 2
        range_indices = np.argmax(delay_scores, axis=0)
        ranges = self.range_grid_m[range_indices]
        delays = ranges / SPEED_OF_LIGHT_MPS
        compensation = np.exp(
            1j
            * 2.0
            * np.pi
            * delays[:, None]
            * self.config.frequency_offsets_hz[None, :]
        )
        channels = np.mean(
            samples
            * np.conj(self.reference)[None, :, None]
            * compensation[:, :, None],
            axis=1,
        )
        angle_scores = np.abs(np.conj(self.spatial_dictionary) @ channels.T) ** 2
        angle_indices = np.argmax(angle_scores, axis=0)
        pairs = self.angle_pairs[angle_indices]
        return [
            RFEstimate(
                float(ranges[index]),
                float(wrap_angle(pairs[index, 0])),
                float(pairs[index, 1]),
            )
            for index in range(samples.shape[0])
        ]
