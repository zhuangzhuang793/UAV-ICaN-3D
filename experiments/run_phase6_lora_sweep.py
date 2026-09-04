"""Train several YOLO11n-OBB LoRA configurations and run Gate 6 with the winner."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer

from prepare_visdrone_obb import prepare
from run_phase6_gate import run as run_phase6
from uav_ican_3d.vision.lora import (
    freeze_batch_norm_statistics,
    freeze_except_lora,
    inject_lora,
    lora_stats,
    merge_lora,
)


class LoRAOBBTrainer(OBBTrainer):
    """Ultralytics OBB trainer that optimizes adapter matrices only."""

    lora_rank = 8
    lora_alpha = 16.0
    lora_dropout = 0.05
    lora_target_pattern = r"^model\.(13|16|17|19|20|22|23)\.(?!dfl)"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.lora_training_stats = inject_lora(
            model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_pattern=self.lora_target_pattern,
        )
        return model

    def _setup_train(self) -> None:
        super()._setup_train()
        freeze_except_lora(self.model)
        freeze_batch_norm_statistics(self.model)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.args.lr0),
            betas=(float(self.args.momentum), 0.999),
            weight_decay=float(self.args.weight_decay),
        )
        self.optimizer.param_groups[0]["param_group"] = "lora"
        self._setup_scheduler()

    def _model_train(self) -> None:
        super()._model_train()
        freeze_batch_norm_statistics(self.model)

    @staticmethod
    def _merge_checkpoint(path: Path) -> None:
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model", "ema"):
            saved_model = checkpoint.get(key)
            if isinstance(saved_model, nn.Module):
                merge_lora(saved_model)
        torch.save(checkpoint, path)

    def final_eval(self) -> None:
        # Ultralytics fuses Conv+BN while loading the final checkpoint. Merge first so the saved
        # artifact is a regular OBBModel and needs neither this trainer nor LoRAConv2d at inference.
        self._merge_checkpoint(self.last)
        self._merge_checkpoint(self.best)
        super().final_eval()


def configured_trainer(candidate: dict, target_pattern: str) -> type[LoRAOBBTrainer]:
    class CandidateTrainer(LoRAOBBTrainer):
        lora_rank = int(candidate["rank"])
        lora_alpha = float(candidate["alpha"])
        lora_dropout = float(candidate["dropout"])
        lora_target_pattern = target_pattern

    CandidateTrainer.__name__ = f"LoRAOBBTrainer_{candidate['name']}"
    return CandidateTrainer


def metric_value(results: dict[str, float], fragment: str) -> float:
    matches = [float(value) for key, value in results.items() if fragment.lower() in key.lower()]
    return matches[0] if matches else float("nan")


def evaluate_baseline(phase: dict, dataset_yaml: Path) -> dict[str, float | int | str]:
    model = YOLO(str(phase["base_model"]))
    metrics = model.val(
        data=str(dataset_yaml),
        batch=int(phase["batch_size"]),
        imgsz=int(phase["image_size_px"]),
        device=str(phase["device"]),
        workers=int(phase["workers"]),
        project=str(phase["runs_dir"]),
        name="baseline",
        plots=False,
        verbose=False,
    )
    results = dict(metrics.results_dict)
    total = sum(parameter.numel() for parameter in model.model.parameters())
    return {
        "name": "baseline",
        "rank": 0,
        "alpha": 0.0,
        "dropout": 0.0,
        "learning_rate": 0.0,
        "adapted_layers": 0,
        "trainable_parameters": 0,
        "total_parameters": total,
        "map50_95": metric_value(results, "mAP50-95"),
        "map50": metric_value(results, "mAP50("),
        "fitness": float(metrics.fitness),
        "checkpoint": str(phase["base_model"]),
        "metrics_json": json.dumps(results, sort_keys=True),
    }


def train_candidate(
    phase: dict, dataset_yaml: Path, candidate: dict
) -> dict[str, float | int | str]:
    name = str(candidate["name"])
    model = YOLO(str(phase["base_model"]))
    trainer = configured_trainer(candidate, str(phase["target_pattern"]))
    model.train(
        trainer=trainer,
        data=str(dataset_yaml),
        epochs=int(phase["epochs"]),
        fraction=float(phase["training_fraction"]),
        patience=int(phase["patience"]),
        batch=int(phase["batch_size"]),
        imgsz=int(phase["image_size_px"]),
        device=str(phase["device"]),
        workers=int(phase["workers"]),
        optimizer="AdamW",
        lr0=float(candidate["learning_rate"]),
        lrf=0.1,
        warmup_epochs=1.0,
        project=str(phase["runs_dir"]),
        name=name,
        seed=int(phase["seed"]),
        deterministic=True,
        plots=False,
        save=True,
        exist_ok=True,
        verbose=False,
    )
    stats = model.trainer.lora_training_stats
    metrics = dict(model.metrics.results_dict)
    merged_path = Path(phase["runs_dir"]) / name / "weights" / "merged_best.pt"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if lora_stats(model.model).adapted_layers:
        raise AssertionError("final YOLO checkpoint still contains unmerged LoRA layers")
    model.save(merged_path)
    return {
        "name": name,
        "rank": int(candidate["rank"]),
        "alpha": float(candidate["alpha"]),
        "dropout": float(candidate["dropout"]),
        "learning_rate": float(candidate["learning_rate"]),
        "adapted_layers": stats.adapted_layers,
        "trainable_parameters": stats.trainable_parameters,
        "total_parameters": stats.total_parameters,
        "map50_95": metric_value(metrics, "mAP50-95"),
        "map50": metric_value(metrics, "mAP50("),
        "fitness": float(model.metrics.fitness),
        "checkpoint": str(merged_path),
        "metrics_json": json.dumps(metrics, sort_keys=True),
    }


def write_results(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_sweep_report(
    path: Path,
    phase: dict,
    counts: dict[str, tuple[int, int]],
    rows: list[dict[str, float | int | str]],
    winner: dict[str, float | int | str],
) -> None:
    table = [
        "| Candidate | Rank | Alpha | Dropout | LR | Trainable | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['name']} | {row['rank']} | {float(row['alpha']):g} | "
            f"{float(row['dropout']):g} | {float(row['learning_rate']):g} | "
            f"{int(row['trainable_parameters']):,} | {float(row['map50']):.3f} | "
            f"{float(row['map50_95']):.3f} |"
        )
    existing = path.read_text(encoding="utf-8").replace(
        "# Phase 6 decision", "# Phase 6 LoRA decision", 1
    )
    training_images = round(counts["train"][0] * float(phase["training_fraction"]))
    appendix = f"""

## LoRA quick sweep

VisDrone car/van labels were mapped to DOTA small-vehicle class 10; truck/bus labels were mapped
to large-vehicle class 9. Horizontal boxes were encoded as valid four-corner OBB labels. The
source images remained read-only and were linked rather than copied.

- Training subset: `{training_images}/{counts['train'][0]}` images
- Validation subset: all `{counts['val'][0]}` images
- Training schedule: `{phase['epochs']}` epochs at `{phase['image_size_px']}` px
- Runtime: `torch {torch.__version__}`, `torchvision {version('torchvision')}`,
  `ultralytics {version('ultralytics')}`
- Adapted modules: neck and OBB head convolutions; backbone and BN statistics frozen
- Winner selected only by held-out VisDrone validation fitness: `{winner['name']}`
- The 20 Gate 6 evaluation frames were used once after selection, not for hyperparameter search

{chr(10).join(table)}

The winning adapters were merged into a normal Ultralytics OBB checkpoint at
`{phase['merged_checkpoint']}`. This is a QUICK comparison, not a publication-scale detector
benchmark; it uses one seed, 25% of the training images, and five epochs.
"""
    path.write_text(existing.rstrip() + appendix, encoding="utf-8")


def run(config_path: Path, smoke: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config["phase6_lora"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 6 LoRA sweep")
    counts = prepare(Path(phase["source_dataset"]), Path(phase["converted_dataset"]))
    print(f"prepared_visdrone={counts}")
    dataset_yaml = Path(phase["converted_dataset"]) / "visdrone_obb.yaml"
    if smoke:
        smoke_phase = dict(phase)
        smoke_phase.update(
            {
                "epochs": 1,
                "training_fraction": 0.005,
                "image_size_px": 320,
                "batch_size": 8,
                "workers": 2,
                "runs_dir": "/tmp/uav_ican_phase6_lora_smoke",
            }
        )
        result = train_candidate(smoke_phase, dataset_yaml, phase["candidates"][0])
        print(f"smoke_result={json.dumps(result, sort_keys=True)}")
        return
    rows = [evaluate_baseline(phase, dataset_yaml)]
    rows.extend(
        train_candidate(phase, dataset_yaml, candidate) for candidate in phase["candidates"]
    )
    write_results(Path(phase["sweep_results_csv"]), rows)

    # Select only on held-out VisDrone validation fitness. Gate 6 remains a final test, not a
    # hyperparameter-selection set.
    winner = max(rows[1:], key=lambda row: float(row["fitness"]))
    final_checkpoint = Path(phase["merged_checkpoint"])
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(winner["checkpoint"]), final_checkpoint)
    print(f"selected_candidate={winner['name']} validation_fitness={winner['fitness']}")
    gate_metrics = run_phase6(
        config_path,
        detector_model=final_checkpoint,
        detector_device=str(phase["device"]),
        detector_cache=Path("data/synchronized_quick/yolo11n_obb_lora_detections.jsonl"),
        results_csv=Path("results/phase6_lora_real_vision.csv"),
        decision_path=Path("docs/PHASE6_LORA_DECISION.md"),
        detector_label=f"VisDrone-LoRA YOLO11n-OBB ({winner['name']})",
    )
    append_sweep_report(
        Path("docs/PHASE6_LORA_DECISION.md"), phase, counts, rows, winner
    )
    print(f"gate6_lora={json.dumps(gate_metrics, sort_keys=True)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, smoke=arguments.smoke)
