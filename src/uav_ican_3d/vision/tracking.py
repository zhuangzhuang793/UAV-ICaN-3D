"""Image-only constant-velocity multi-object Kalman tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


FloatArray = NDArray[np.float64]


@dataclass
class _Track:
    state: FloatArray
    covariance: FloatArray
    missed: int = 0


class ImageCVHungarianTracker:
    """Track image candidates without any RF or ground-truth input."""

    def __init__(
        self,
        process_std_px: float,
        measurement_std_px: float,
        gate_confidence: float,
        max_missed_frames: int,
    ) -> None:
        if process_std_px <= 0.0 or measurement_std_px <= 0.0:
            raise ValueError("tracker noise scales must be positive")
        if not 0.0 < gate_confidence < 1.0 or max_missed_frames < 0:
            raise ValueError("invalid tracker gate or lifetime")
        self.process_variance = float(process_std_px) ** 2
        self.measurement_covariance = np.eye(2) * float(measurement_std_px) ** 2
        self.gate_threshold = float(chi2.ppf(gate_confidence, df=2))
        self.max_missed_frames = int(max_missed_frames)
        self.tracks: dict[int, _Track] = {}
        self.next_id = 0
        self.served_track_id: int | None = None
        self.last_timestamp: float | None = None

    def _new_track(self, pixel: FloatArray) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = _Track(
            np.array([pixel[0], pixel[1], 0.0, 0.0]),
            np.diag([16.0, 16.0, 100.0, 100.0]),
        )
        return track_id

    def step(
        self, pixels_uv: ArrayLike, confidences: ArrayLike, timestamp_s: float
    ) -> int | None:
        pixels = np.asarray(pixels_uv, dtype=float)
        confidence = np.asarray(confidences, dtype=float)
        if pixels.shape == (0,):
            pixels = np.empty((0, 2), dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or confidence.shape != (len(pixels),):
            raise ValueError("tracker candidates have incompatible shapes")
        if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(confidence)):
            raise ValueError("tracker candidates must be finite")
        timestamp = float(timestamp_s)
        dt = 1.0 if self.last_timestamp is None else max(1e-3, min(timestamp - self.last_timestamp, 1.0))
        self.last_timestamp = timestamp
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        process = self.process_variance * np.array(
            [[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2], [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]]
        )
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        for track in self.tracks.values():
            track.state = transition @ track.state
            track.covariance = transition @ track.covariance @ transition.T + process
            track.missed += 1

        if not self.tracks:
            created = [self._new_track(pixel) for pixel in pixels]
            if created:
                self.served_track_id = created[int(np.argmax(confidence))]
                return int(np.argmax(confidence))
            return None

        track_ids = list(self.tracks)
        costs = np.full((len(track_ids), len(pixels)), np.inf)
        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            innovation_covariance = (
                observation @ track.covariance @ observation.T + self.measurement_covariance
            )
            differences = pixels - observation @ track.state
            if len(pixels):
                costs[row] = np.einsum(
                    "ni,ij,nj->n",
                    differences,
                    np.linalg.inv(innovation_covariance),
                    differences,
                )
        accepted: dict[int, int] = {}
        if costs.size:
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] > self.gate_threshold:
                    continue
                track_id = track_ids[int(row)]
                track = self.tracks[track_id]
                innovation_covariance = (
                    observation @ track.covariance @ observation.T + self.measurement_covariance
                )
                gain = track.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
                track.state += gain @ (pixels[column] - observation @ track.state)
                track.covariance = (
                    np.eye(4) - gain @ observation
                ) @ track.covariance
                track.missed = 0
                accepted[track_id] = int(column)

        matched_candidates = set(accepted.values())
        for index, pixel in enumerate(pixels):
            if index not in matched_candidates:
                self._new_track(pixel)
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_missed_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]
        return accepted.get(self.served_track_id) if self.served_track_id is not None else None
