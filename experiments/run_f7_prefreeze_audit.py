"""Audit the F7 online boundary and frozen inputs before opening final results."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import yaml

from uav_ican_3d.localization import WaveformRFEstimator, localize_online
from uav_ican_3d.vision import associate_candidates


ONLINE_SOURCES = (
    Path("src/uav_ican_3d/localization/final_pipeline.py"),
    Path("src/uav_ican_3d/localization/map_estimator.py"),
    Path("src/uav_ican_3d/localization/rf_waveform.py"),
    Path("src/uav_ican_3d/vision/association.py"),
)
FORBIDDEN_IDENTIFIERS = {
    "user_gt_position",
    "served_target_gt_id",
    "evaluation_only",
    "path_delay",
    "path_aoa",
    "path_angles",
    "sionna_paths",
    "coefficients_by_antenna_path",
    "delays_s",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    values.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    values.update(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return values


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config["f7"]
    violations = {
        str(path): sorted(_identifiers(path) & FORBIDDEN_IDENTIFIERS)
        for path in ONLINE_SOURCES
    }
    violations = {path: values for path, values in violations.items() if values}
    signatures = {
        "localize_online": list(inspect.signature(localize_online).parameters),
        "WaveformRFEstimator.estimate": list(
            inspect.signature(WaveformRFEstimator.estimate).parameters
        ),
        "WaveformRFEstimator.estimate_batch": list(
            inspect.signature(WaveformRFEstimator.estimate_batch).parameters
        ),
        "associate_candidates": list(inspect.signature(associate_candidates).parameters),
    }
    signature_violations = {
        name: sorted(set(parameters) & FORBIDDEN_IDENTIFIERS)
        for name, parameters in signatures.items()
        if set(parameters) & FORBIDDEN_IDENTIFIERS
    }
    root = Path(config["f0"]["output_root"])
    manifest_rows: dict[str, int] = {}
    manifest_hashes: dict[str, str] = {}
    for sequence_id in phase["test_sequences"]:
        path = root / sequence_id / "manifest.jsonl"
        manifest_rows[sequence_id] = sum(bool(line) for line in path.read_text().splitlines())
        manifest_hashes[sequence_id] = _sha256(path)
    output_root = Path(config["outputs"]["root"])
    detection_path = output_root / "f4_test_detections.jsonl"
    detection_rows = sum(bool(line) for line in detection_path.read_text().splitlines())
    calibration = json.loads((output_root / "f1_visual_calibration.json").read_text())
    passed = (
        not violations
        and not signature_violations
        and list(phase["test_sequences"]) == list(config["f0"]["sequence_splits"]["test"])
        and len(phase["rf_seeds"]) == 5
        and len(set(map(int, phase["rf_seeds"]))) == 5
        and all(count == 400 for count in manifest_rows.values())
        and sum(manifest_rows.values()) == 3200
        and detection_rows == 3200
        and calibration["status"] == "PASS"
        and calibration["selected_model"]["kind"] == "C0"
        and config["f6"]["path_truth_firewall"] is True
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "online_source_identifier_violations": violations,
        "online_signature_violations": signature_violations,
        "online_signatures": signatures,
        "test_sequences": list(phase["test_sequences"]),
        "manifest_rows": manifest_rows,
        "manifest_sha256": manifest_hashes,
        "held_out_frames": sum(manifest_rows.values()),
        "detection_cache_rows": detection_rows,
        "detection_cache_sha256": _sha256(detection_path),
        "waveform_seeds": list(map(int, phase["rf_seeds"])),
        "visual_model": calibration["selected_model"]["kind"],
        "sionna_path_truth_firewall": bool(config["f6"]["path_truth_firewall"]),
        "online_source_sha256": {str(path): _sha256(path) for path in ONLINE_SOURCES},
    }
    path = output_root / "f7_prefreeze_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("F7 pre-freeze leakage audit failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)
