"""Sionna-RT path-truth boundary used only to synthesize complex waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rf_waveform import SRSWaveformConfig, srs_reference


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SionnaPathChannel:
    """Intermediate propagation truth that must not cross the waveform boundary."""

    coefficients_by_antenna_path: ComplexArray
    delays_s: FloatArray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients_by_antenna_path, dtype=np.complex128)
        delays = np.asarray(self.delays_s, dtype=float)
        if coefficients.ndim != 2 or delays.shape != (coefficients.shape[1],):
            raise ValueError("Sionna path coefficient and delay shapes are incompatible")
        if coefficients.shape[0] != 16 or coefficients.shape[1] < 1:
            raise ValueError("Sionna channel requires a 4x4 UPA and at least one path")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(delays)):
            raise ValueError("Sionna path values must be finite")


def trace_sionna_channel(
    scene_path: Path,
    carrier_hz: float,
    transmitter_position_xyz_m: ArrayLike,
    receiver_position_xyz_m: ArrayLike,
    *,
    max_depth: int,
    reflections: bool,
    samples_per_source: int,
    max_paths_per_source: int,
    synthetic_array: bool,
    seed: int,
    diffuse_reflections: bool = False,
    refraction: bool = False,
    diffraction: bool = False,
    edge_diffraction: bool = False,
) -> SionnaPathChannel:
    """Trace paths and return only the channel-side intermediate representation."""

    import mitsuba as mi
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(str(scene_path))
    scene.frequency = float(carrier_hz)
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    transmitter = Transmitter(
        name="vehicle_ue", position=mi.Point3f(np.asarray(transmitter_position_xyz_m, dtype=float))
    )
    # A pi roll maps Sionna's z-up frame to the project's FRD array frame.
    receiver = Receiver(
        name="uav_array",
        position=mi.Point3f(np.asarray(receiver_position_xyz_m, dtype=float)),
        orientation=mi.Point3f(0.0, 0.0, np.pi),
    )
    scene.add(transmitter)
    scene.add(receiver)
    paths = PathSolver()(
        scene,
        max_depth=int(max_depth),
        max_num_paths_per_src=int(max_paths_per_source),
        samples_per_src=int(samples_per_source),
        synthetic_array=bool(synthetic_array),
        los=True,
        specular_reflection=bool(reflections),
        diffuse_reflection=bool(diffuse_reflections),
        refraction=bool(refraction),
        diffraction=bool(diffraction),
        edge_diffraction=bool(edge_diffraction),
        seed=int(seed),
    )
    valid = np.asarray(paths.valid.numpy(), dtype=bool).reshape(-1)
    delays = np.asarray(paths.tau.numpy(), dtype=float).reshape(-1)[valid]
    real, imaginary = paths.a
    coefficients = np.asarray(real.numpy()) + 1j * np.asarray(imaginary.numpy())
    coefficients = coefficients[0, :, 0, 0, :][:, valid]
    return SionnaPathChannel(coefficients, delays)


def synthesize_sionna_waveform(
    channel: SionnaPathChannel,
    snr_db: float,
    config: SRSWaveformConfig,
    rng: np.random.Generator,
    *,
    normalize_received_power: bool = True,
    transmit_power_per_subcarrier_w: float | None = None,
    noise_power_per_subcarrier_w: float | None = None,
) -> ComplexArray:
    """Consume path truth and emit only a complex received waveform.

    The default preserves the frozen F6 behavior.  Setting
    ``normalize_received_power=False`` activates an absolute-power link budget;
    both per-subcarrier powers must then be supplied in watts.
    """

    coefficients = np.asarray(channel.coefficients_by_antenna_path)
    delays = np.asarray(channel.delays_s)
    frequency_response = np.einsum(
        "al,kl->ka",
        coefficients,
        np.exp(-1j * 2.0 * np.pi * config.frequency_offsets_hz[:, None] * delays[None, :]),
    )
    power = float(np.mean(np.abs(frequency_response) ** 2))
    if not np.isfinite(power) or power <= 0.0:
        raise RuntimeError("Sionna channel has zero or non-finite received power")
    pilots = srs_reference(config.subcarrier_count)
    if normalize_received_power:
        normalized = frequency_response / np.sqrt(power)
        noiseless = pilots[:, None] * normalized
        noise_variance = 1.0 / (10.0 ** (float(snr_db) / 10.0))
    else:
        if (
            transmit_power_per_subcarrier_w is None
            or noise_power_per_subcarrier_w is None
            or transmit_power_per_subcarrier_w <= 0.0
            or noise_power_per_subcarrier_w <= 0.0
        ):
            raise ValueError("absolute-power synthesis requires positive transmit/noise powers")
        noiseless = pilots[:, None] * frequency_response * np.sqrt(
            transmit_power_per_subcarrier_w
        )
        noise_variance = float(noise_power_per_subcarrier_w)
    noise = (
        rng.normal(size=noiseless.shape) + 1j * rng.normal(size=noiseless.shape)
    ) * np.sqrt(noise_variance / 2.0)
    return noiseless + noise


def sionna_received_power_gain(channel: SionnaPathChannel, config: SRSWaveformConfig) -> float:
    """Return mean received power for unit per-subcarrier transmit power."""

    coefficients = np.asarray(channel.coefficients_by_antenna_path)
    delays = np.asarray(channel.delays_s)
    response = np.einsum(
        "al,kl->ka",
        coefficients,
        np.exp(-1j * 2.0 * np.pi * config.frequency_offsets_hz[:, None] * delays[None, :]),
    )
    power = float(np.mean(np.abs(response) ** 2))
    if not np.isfinite(power) or power <= 0.0:
        raise RuntimeError("Sionna channel has zero or non-finite received power")
    return power
