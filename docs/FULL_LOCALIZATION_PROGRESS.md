# FULL localization progress

Current phase: **F5 — analytic LoS RF waveform fidelity**

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
- F1: **PASS** — C0 selected on independent validation pixel NLL
- F2: **PASS** — formal shared-pose complementarity region preserved
- F3: **PASS** — nonlinear paired Monte Carlo agrees with EFIM trends
- F4: **PASS** — RF gating outperforms visual CV/Hungarian on Hard frames
- F5–F10: not started

## F0 frozen artifacts

- Formal detector: `checkpoints/full_localization_lora.pt`
- Detector SHA-256: `c1a8f9176df331cdf5f695b482322f8f60f3ad0933a4a9a1dca2ae143febb6d1`
- Full VisDrone train / validation images: `6471 / 548`
- Selection metric / value: validation mAP50-95 / `0.257586`
- Calibration / validation / held-out test frames: `800 / 800 / 3200`
- Dataset and manifest hashes: `docs/FULL_LOCALIZATION_DATA_FREEZE.md`

## F1 frozen visual model

- Calibration / validation frames: `800 / 800`; final-test frames read: `0`
- GT-matched detector samples: `800 / 800`
- Selected model: `C0` (fixed OBB-center bias correction)
- C0 / C1 validation pixel NLL: `4.123548 / 8.876864`
- Frozen pixel covariance: `[[2.998059, -1.009502], [-1.009502, 2.309385]]` px²
- Full protocol: `docs/FULL_VISUAL_CALIBRATION.md`

## F2 EFIM freeze

- Evaluated cells: `770` across separate reference, RF-quality, pose-quality and position-sensitivity families
- Nominal PEB / ZEB gain ranges: `15.07%–65.35% / 5.34%–73.24%`
- Strong reference cells: `63/70`; robust cells at 0.5-degree attitude uncertainty: `70/70`
- F3 representative geometries: 8 cells frozen by gain octile plus spatial-diversity rule
- Decision: `docs/FULL_EFIM_STUDY.md`

## F3 nonlinear MAP validation

- Paired trials: `8 x 500 = 4000`
- 3-D / Z improvement: `8/8 / 8/8` geometries
- EFIM-to-empirical gain Spearman: `1.000` (3-D), `0.976` (Z)
- RF / joint 95% coverage ranges: `93.6%–95.8% / 93.0%–96.2%`
- Decision: `docs/FULL_OBSERVATION_LOCALIZATION.md`

## F4 association freeze

- Easy / Hard frames: `2538 / 662`, frozen before correctness evaluation
- Visual-only Easy / Hard accuracy: `15.29% / 7.25%`
- RF-guided Easy / Hard accuracy: `97.91% / 86.56%`
- Oracle Easy / Hard accuracy: `99.92% / 98.19%`
- F4 gate: PASS; F8 Hard target `>=90%`: **not met and preserved**
- Decision: `docs/FULL_ASSOCIATION.md`

Formal outputs are isolated under `results/full_localization/`. Prediction, MPC and beam-control
FULL experiments remain outside this workflow.
