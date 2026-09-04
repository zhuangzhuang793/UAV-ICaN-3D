"""Train the preselected LoRA configuration on the full VisDrone training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml

try:
    from experiments.prepare_visdrone_obb import prepare
    from experiments.run_phase6_lora_sweep import train_candidate, write_results
except ModuleNotFoundError:  # Direct execution places experiments/ rather than the repo on sys.path.
    from prepare_visdrone_obb import prepare
    from run_phase6_lora_sweep import train_candidate, write_results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path, smoke: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("formal LoRA training requires mode: full_localization")
    phase = dict(config["f0"]["detector"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal LoRA training")
    counts = prepare(Path(phase["source_dataset"]), Path(phase["converted_dataset"]))
    dataset_yaml = Path(phase["converted_dataset"]) / "visdrone_obb.yaml"
    candidate = {
        "name": "formal_r4_a8",
        "rank": int(phase["rank"]),
        "alpha": float(phase["alpha"]),
        "dropout": float(phase["dropout"]),
        "learning_rate": float(phase["learning_rate"]),
    }
    if smoke:
        phase.update(
            {
                "epochs": 1,
                "training_fraction": 0.005,
                "image_size_px": 320,
                "batch_size": 8,
                "workers": 2,
                "runs_dir": "/tmp/uav_ican_full_lora_smoke",
            }
        )
        candidate["name"] = "formal_r4_a8_smoke"
        result = train_candidate(phase, dataset_yaml, candidate)
        print(f"smoke_result={json.dumps(result, sort_keys=True)}")
        return

    if float(phase["training_fraction"]) != 1.0:
        raise ValueError("formal detector must use the full VisDrone training split")
    result = train_candidate(phase, dataset_yaml, candidate)
    checkpoint = Path(str(result["checkpoint"]))
    final_checkpoint = Path(phase["merged_checkpoint"])
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, final_checkpoint)
    result["checkpoint"] = str(final_checkpoint)
    result["checkpoint_sha256"] = _sha256(final_checkpoint)
    result["training_images"] = counts["train"][0]
    result["validation_images"] = counts["val"][0]
    result["selection_metric"] = "validation mAP50-95"
    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_results(output_root / "detector_training.csv", [result])
    (output_root / "detector_training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"formal_detector={final_checkpoint}")
    print(f"validation_map50_95={float(result['map50_95']):.6f}")
    print(f"checkpoint_sha256={result['checkpoint_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, arguments.smoke)
