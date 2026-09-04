"""Lightweight Gaussian trajectory predictor with an explicit covariance path."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BeliefTrajectoryPredictor(nn.Module):
    """Predict a factorized joint Gaussian over future 3-D positions.

    The full model encodes the complete belief covariance history and also propagates the final
    position/velocity covariance through a constant-velocity prior.  The mean-only ablation has
    neither path.
    """

    def __init__(
        self,
        history_steps: int,
        horizon_steps: int,
        hidden_size: int = 64,
        dt_s: float = 0.5,
        use_covariance: bool = True,
    ) -> None:
        super().__init__()
        if history_steps < 2 or horizon_steps < 1 or hidden_size < 8 or dt_s <= 0.0:
            raise ValueError("invalid trajectory predictor dimensions")
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.dt_s = dt_s
        self.use_covariance = use_covariance
        self.register_buffer(
            "mean_feature_scale", torch.tensor([20.0, 20.0, 5.0, 10.0, 10.0, 2.0])
        )
        mean_features = history_steps * 6
        covariance_features = history_steps * 21 if use_covariance else 0
        self.register_buffer("upper_rows", torch.triu_indices(6, 6)[0])
        self.register_buffer("upper_columns", torch.triu_indices(6, 6)[1])
        self.mean_encoder = nn.Sequential(
            nn.Linear(mean_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_size, horizon_steps * 3)
        self.scale_encoder = nn.Sequential(
            nn.Linear(mean_features + covariance_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.scale_head = nn.Linear(hidden_size, horizon_steps * 3)
        if use_covariance:
            # Multiple history beliefs make the conditional state uncertainty smaller than a
            # one-step open-loop propagation. Learn one positive calibration factor instead of
            # double-counting that uncertainty in the neural process-noise head.
            self.covariance_log_scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("covariance_log_scale", None)

    def forward(self, means: Tensor, covariances: Tensor) -> tuple[Tensor, Tensor]:
        if means.ndim != 3 or means.shape[1:] != (self.history_steps, 6):
            raise ValueError("means must have shape (batch, history_steps, 6)")
        if covariances.shape != (
            means.shape[0],
            self.history_steps,
            6,
            6,
        ):
            raise ValueError("covariances must have shape (batch, history_steps, 6, 6)")
        mean_features = (means / self.mean_feature_scale).reshape(means.shape[0], -1)
        scale_features = [mean_features]
        if self.use_covariance:
            upper = covariances[:, :, self.upper_rows, self.upper_columns]
            diagonal_mask = self.upper_rows == self.upper_columns
            upper = upper.clone()
            upper[:, :, diagonal_mask] = torch.log(
                torch.clamp(upper[:, :, diagonal_mask], min=1e-8)
            )
            scale_features.append(upper.reshape(means.shape[0], -1))
        mean_encoded = self.mean_encoder(mean_features)
        scale_encoded = self.scale_encoder(torch.cat(scale_features, dim=1))
        correction = self.mean_head(mean_encoded).reshape(-1, self.horizon_steps, 3)
        process_std = functional.softplus(
            self.scale_head(scale_encoded).reshape(-1, self.horizon_steps, 3)
        ) + 1e-3
        times = (
            torch.arange(1, self.horizon_steps + 1, device=means.device, dtype=means.dtype)
            * self.dt_s
        )
        baseline = means[:, -1, :3].unsqueeze(1) + (
            times.reshape(1, -1, 1) * means[:, -1, 3:].unsqueeze(1)
        )
        predicted_mean = baseline + correction
        variance = process_std.square()
        if self.use_covariance:
            final_covariance = covariances[:, -1]
            position_variance = torch.diagonal(final_covariance[:, :3, :3], dim1=-2, dim2=-1)
            velocity_variance = torch.diagonal(final_covariance[:, 3:, 3:], dim1=-2, dim2=-1)
            position_velocity = torch.diagonal(
                final_covariance[:, :3, 3:], dim1=-2, dim2=-1
            )
            propagated = (
                position_variance.unsqueeze(1)
                + times.square().reshape(1, -1, 1) * velocity_variance.unsqueeze(1)
                + 2.0 * times.reshape(1, -1, 1) * position_velocity.unsqueeze(1)
            )
            assert self.covariance_log_scale is not None
            variance = variance + torch.exp(self.covariance_log_scale) * torch.clamp(
                propagated, min=0.0
            )
        return predicted_mean, torch.sqrt(torch.clamp(variance, min=1e-8))


def diagonal_gaussian_nll(target: Tensor, mean: Tensor, standard_deviation: Tensor) -> Tensor:
    """Mean negative log likelihood for a diagonal Gaussian trajectory distribution."""

    if target.shape != mean.shape or standard_deviation.shape != mean.shape:
        raise ValueError("target, mean, and standard_deviation shapes must match")
    if torch.any(standard_deviation <= 0.0):
        raise ValueError("standard_deviation must be positive")
    normalized = (target - mean) / standard_deviation
    return 0.5 * (
        normalized.square()
        + 2.0 * torch.log(standard_deviation)
        + float(np.log(2.0 * np.pi))
    ).mean()
