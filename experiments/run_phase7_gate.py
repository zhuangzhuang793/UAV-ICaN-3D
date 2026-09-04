"""Quick-train and validate covariance-aware probabilistic trajectory prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import chi2
from torch.utils.data import DataLoader, TensorDataset

from uav_ican_3d.prediction import (
    BeliefTrajectoryPredictor,
    diagonal_gaussian_nll,
    generate_synthetic_beliefs,
)


def _train(
    model: BeliefTrajectoryPredictor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(means, covariances, targets),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        samples = 0
        for batch_means, batch_covariances, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predicted_mean, predicted_std = model(batch_means, batch_covariances)
            loss = diagonal_gaussian_nll(batch_targets, predicted_mean, predicted_std)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch_means)
            samples += len(batch_means)
        history.append(total / samples)
    return history


def _metrics(
    model: BeliefTrajectoryPredictor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    targets: torch.Tensor,
    confidence: float,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        predicted_mean, predicted_std = model(means, covariances)
        nll = float(diagonal_gaussian_nll(targets, predicted_mean, predicted_std))
        squared_distance = torch.sum(
            ((targets - predicted_mean) / predicted_std).square(), dim=2
        )
        coverage = float(
            torch.mean((squared_distance <= chi2.ppf(confidence, df=3)).float())
        )
    return nll, coverage


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 7 gate refuses to run unless mode is 'quick'")
    phase = config["phase7"]
    seed = int(config["seed"]) + 7
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    history_steps = int(phase["history_steps"])
    horizon_steps = int(phase["horizon_steps"])
    dt_s = float(phase["dt_s"])
    training = generate_synthetic_beliefs(
        int(phase["training_samples"]), history_steps, horizon_steps, dt_s, seed
    )
    validation = generate_synthetic_beliefs(
        int(phase["validation_samples"]), history_steps, horizon_steps, dt_s, seed + 1
    )

    full_model = BeliefTrajectoryPredictor(
        history_steps,
        horizon_steps,
        int(phase["hidden_size"]),
        dt_s,
        use_covariance=True,
    )
    torch.manual_seed(seed)
    mean_only_model = BeliefTrajectoryPredictor(
        history_steps,
        horizon_steps,
        int(phase["hidden_size"]),
        dt_s,
        use_covariance=False,
    )
    full_history = _train(
        full_model,
        training.means,
        training.covariances,
        training.future_positions,
        int(phase["epochs"]),
        int(phase["batch_size"]),
        float(phase["learning_rate"]),
        seed,
    )
    mean_only_history = _train(
        mean_only_model,
        training.means,
        training.covariances,
        training.future_positions,
        int(phase["epochs"]),
        int(phase["batch_size"]),
        float(phase["learning_rate"]),
        seed,
    )
    confidence = float(phase["coverage_confidence"])
    full_nll, full_coverage = _metrics(
        full_model,
        validation.means,
        validation.covariances,
        validation.future_positions,
        confidence,
    )
    mean_nll, mean_coverage = _metrics(
        mean_only_model,
        validation.means,
        validation.covariances,
        validation.future_positions,
        confidence,
    )
    with torch.no_grad():
        probe_means = validation.means[:1]
        probe_covariance = validation.covariances[:1]
        low_covariance = probe_covariance * 0.05
        high_covariance = probe_covariance * 20.0
        low_mean, low_std = full_model(probe_means, low_covariance)
        high_mean, high_std = full_model(probe_means, high_covariance)
    standard_deviation_ratio = float(torch.mean(high_std) / torch.mean(low_std))
    distribution_shift = float(
        torch.linalg.vector_norm(high_mean - low_mean)
        + torch.linalg.vector_norm(high_std - low_std)
    )
    covariance_scale = float(torch.exp(full_model.covariance_log_scale).detach())
    nll_gain = mean_nll - full_nll
    gate = phase["gate"]
    passed = (
        standard_deviation_ratio >= float(gate["minimum_high_low_std_ratio"])
        and nll_gain >= float(gate["minimum_nll_gain"])
        and abs(full_coverage - confidence) <= float(gate["coverage_tolerance"])
        and distribution_shift > 0.0
    )
    result_path = Path(phase["results_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,full_train_nll,mean_only_train_nll"]
    lines.extend(
        f"{epoch + 1},{full_value},{mean_value}"
        for epoch, (full_value, mean_value) in enumerate(
            zip(full_history, mean_only_history, strict=True)
        )
    )
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checkpoint_path = Path(phase["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "hidden_size": int(phase["hidden_size"]),
            "dt_s": dt_s,
        },
        checkpoint_path,
    )
    status = "PASS" if passed else "FAIL"
    Path("docs/PHASE7_DECISION.md").write_text(
        f"""# Phase 7 decision

Status: **{status}**

A lightweight Gaussian trajectory network was quick-trained on {len(training.means)} synthetic
belief histories containing equal straight and turning examples. It predicts a future joint
trajectory mean and diagonal covariance. The full model encodes covariance histories and includes
an explicit kinematic covariance-propagation path; the ablation receives means only.

- Validation full / mean-only NLL: `{full_nll:.4f} / {mean_nll:.4f}`
- Full-model NLL gain: `{nll_gain:.4f}`
- Target / full / mean-only 3-D coverage:
  `{confidence:.3f} / {full_coverage:.3f} / {mean_coverage:.3f}`
- High/low input-covariance output-std ratio: `{standard_deviation_ratio:.3f}`
- Learned kinematic covariance scale: `{covariance_scale:.4f}`
- Distribution change under the covariance intervention: `{distribution_shift:.3f}`
- QUICK epochs: `{int(phase['epochs'])}`
- Local checkpoint: `{checkpoint_path}`

Ground truth is used only to quick-train and evaluate this synthetic predictor. Its online API
accepts belief means and covariances; no future state or simulator truth is an input.
""",
        encoding="utf-8",
    )
    print(f"full_validation_nll={full_nll:.4f} mean_only_validation_nll={mean_nll:.4f}")
    print(f"nll_gain={nll_gain:.4f}")
    print(
        f"target_coverage={confidence:.3f} full_coverage={full_coverage:.3f} "
        f"mean_only_coverage={mean_coverage:.3f}"
    )
    print(f"high_low_output_std_ratio={standard_deviation_ratio:.3f}")
    print(f"learned_kinematic_covariance_scale={covariance_scale:.4f}")
    print(f"covariance_intervention_distribution_shift={distribution_shift:.3f}")
    print(f"PHASE 7 QUICK GATE: {status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)
