# FULL localization progress

Current phase: **F1 — formal visual geometry calibration**

## Frozen scope

- Carrier: `3.5 GHz`
- Array: `4x4 UPA`
- RF observation: range, azimuth and elevation
- Camera: onboard monocular RGB
- Target: vehicle-mounted UE with fixed FRD antenna lever arm
- Shared UAV/camera pose uncertainty remains active
- QUICK configuration SHA-256:
  `7be6cdd844a6f0c5694b7acf787934dbc7cd48449c93a77d83f89aedf67547f6`

## Phase status

- R0: **PASS**
- F0: **PASS** — 12 disjoint Cosys-AirSim sequences, 4800 synchronized frames
- F1–F10: not started

## F0 frozen artifacts

- Formal detector: `checkpoints/full_localization_lora.pt`
- Detector SHA-256: `c1a8f9176df331cdf5f695b482322f8f60f3ad0933a4a9a1dca2ae143febb6d1`
- Full VisDrone train / validation images: `6471 / 548`
- Selection metric / value: validation mAP50-95 / `0.257586`
- Calibration / validation / held-out test frames: `800 / 800 / 3200`
- Dataset and manifest hashes: `docs/FULL_LOCALIZATION_DATA_FREEZE.md`

Formal outputs are isolated under `results/full_localization/`. Prediction, MPC and beam-control
FULL experiments remain outside this workflow.
