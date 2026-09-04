# UAV-ICaN 3D progress

Workflow status: QUICK phases 0–10 complete

Latest gate result: Phase 10 **PASS**

## Post-LoRA regression

Post-LoRA regression: **PASS**

- Detector: `checkpoints/phase6_lora_best.pt`
- Phase 9: **PASS**
- Phase 10: **PASS**
- Frozen detector SHA-256:
  `2c55ddd640fbb04c683b12e1126efb70cdbcb443bad39f1dff1c3df24ed3d541`
- The QUICK configuration remained unchanged during the R0 rerun.

Important outputs:

- World ENU, body FRD, calibrated array, and OpenCV camera conventions are fixed in
  `docs/COORDINATES.md`.
- QUICK configuration lives in `configs/quick.yaml`.
- Eight seeded transform round trips had worst absolute error `1.279e-13`.
- The hand-computed camera projection had zero numerical error.
- The RF analytic Jacobian had worst relative finite-difference error `8.589e-10` over eight
  seeded observable geometries.
- Representative PEB increased from `2.034711 m` to `2.211345 m` with higher range noise and to
  `3.976203 m` with higher AoA noise.
- Shared-pose/camera analytic Jacobians had worst relative finite-difference error `1.742e-9`.
- Camera-off EFIM exactly matched RF-only. Representative RF-only and joint PEB were `1.354980 m`
  and `0.950565 m`; lowering pixel noise improved joint PEB to `0.917249 m`.
- Scaling shared pose covariance by four worsened joint PEB to `1.612477 m`.
- Schur-complement and joint-marginal target covariances agreed to relative error `5.281e-16`;
  all 18 tests through Phase 2 passed.
- The sparse QUICK scan evaluated 324 cells over 30–100 m altitude, 20–100 m horizontal distance,
  three RF noise levels, and four nonzero attitude-uncertainty levels.
- Thirty-one predeclared reasonable cells passed both 20% PEB and 20% ZEB gains, spanning all nine
  geometries. Vision-only target information remained rank 2 throughout.
- The decision report is `docs/PHASE3_DECISION.md`; the low-resolution map is
  `docs/figures/phase3_complementarity_map.png`.
- Nonlinear MAP jointly estimates UE position and the same six-dimensional shared UAV pose error;
  its initializer uses only the RF observation.
- Across three representative geometries with 30 trials each, RF+Vision 3-D RMSE was
  `1.240/1.555/2.471 m`, versus RF-only `1.974/3.912/5.071 m`; Z-RMSE improved in every case.
- Mean position NEES stayed between `2.166` and `3.435`, and every reported covariance was
  positive definite. All 23 tests through Phase 4 passed.
- The RF estimator's complete 9-D target/shared-pose covariance is propagated into a full 2-D
  image covariance, preserving target/pose cross-correlation.
- Chi-square Mahalanobis gating and the local VisDrone YOLO-label adapter are implemented. The
  adapter inspected 30 validation frames with 321 car/van/truck/bus candidates.
- Cosys-AirSim 3.4.1 / UE 5.8 runs headlessly on GPU 0 and exported 30 synchronized frames with
  RGB, camera calibration/pose, three vehicle poses, and instance-segmentation GT boxes.
- The served antenna phase center is independently computed from the vehicle pose and a fixed FRD
  lever arm. Its projection falls inside the served GT box in all 30 frames.
- Empirical true-UE pixel coverage was `1.000` for the target `0.950` confidence region. In the
  final oblique-camera export, RF-guided GT association was `0.967`, versus `0.000` for the
  unguided image-center baseline.
- Gate 5 passed. See `docs/PHASE5_DECISION.md`,
  `docs/COSYS_AIRSIM_SETUP.md`, and `results/phase5_association.csv`.
- A first COCO YOLO11n probe localized the simulator cars but mislabeled them as kites, so it was
  rejected as a vehicle detector. YOLO11n-OBB pretrained on aerial DOTA was used instead, with its
  native small/large-vehicle classes and no retraining.
- Ten synchronized calibration frames produced bbox-center-to-antenna bias
  `[-0.463, 3.827] px` and anisotropic residual covariance
  `[[4.845, -0.040], [-0.040, 10.947]] px²`; detector confidence was not used as covariance.
- On 20 held-out sequence frames, 19 real-vision updates and 18 correct RF-guided associations
  reduced 3-D RMSE from `1.912 m` to `1.134 m` and Z-RMSE from `0.801 m` to `0.625 m`.
- A deliberate far detector outlier was rejected before fusion with RF-only fallback. Gate 6
  passed. See `docs/PHASE6_DECISION.md` and
  `results/phase6_real_vision.csv`.
- A post-gate QUICK LoRA sweep adapted 50 neck/head convolutions of YOLO11n-OBB on 25% of the
  local VisDrone training split while freezing the backbone and BN statistics. Of three tested
  configurations, rank 4 / alpha 8 / dropout 0.05 / LR 1e-3 won on the complete held-out
  VisDrone validation split (`mAP50-95 0.160`, versus `0.028` without adaptation).
- The merged LoRA winner produced 20/20 visual updates and 19/20 correct associations, reducing
  Gate 6 3-D RMSE to `0.988 m` and Z-RMSE to `0.589 m`. See
  `docs/PHASE6_LORA_DECISION.md`, `results/phase6_lora_sweep.csv`, and
  `results/phase6_lora_real_vision.csv`.
- A 64-hidden-unit Gaussian predictor was quick-trained for 60 epochs (about eight seconds) on 600
  synthetic belief histories split evenly between straight and constant-turn motion.
- The full covariance-aware model achieved validation NLL `0.4101` versus `0.5828` for mean-only;
  its 3-D coverage was `0.890` for a `0.900` target.
- Scaling an otherwise fixed input covariance from low to high changed the predicted distribution
  and increased mean output standard deviation by `2.842x`. Gate 7 passed; all 35 tests pass. See
  `docs/PHASE7_DECISION.md` and `results/phase7_prediction.csv`.
- A six-step slow closed loop consumed Kalman belief histories, sampled full predicted
  trajectories, and produced feasible sampling-based robust-MPC actions; three fast 3-D beam
  updates ran after every slow action.
- The 4x4 UPA selector chose a narrow 4x4 beam for low angular uncertainty and a broader 2x2 taper
  with a different elevation for the high-uncertainty probe. All 18 fast updates were finite and
  every speed/altitude constraint held. Gate 8 passed. See
  `docs/PHASE8_DECISION.md` and `results/phase8_closed_loop.csv`.
- A known QPSK SRS-like uplink at 3.5 GHz drove independent grid-ML delay and 2-D AoA estimation
  on the same 4x4 UPA. The estimator API receives only the complex waveform and reference.
- Equivalent range/angle error fell from `14.590 m` through `8.881 m` to `0.986 m` over
  `-15/-5/5 dB`, and the nominal residuals supplied a full empirical RF covariance.
- With that waveform covariance and temporal RF-gated LoRA detections, a ten-frame subset reduced
  3-D RMSE from `2.804 m` to `2.033 m` and Z-RMSE from `2.131 m` to `1.653 m`. Gate 9 passed. See
  `docs/PHASE9_DECISION.md`, `results/phase9_waveform.csv`, and
  `results/phase9_pipeline.csv`.
- The final eight-step loop connected SRS reception, waveform RF estimation, image-belief
  projection, served association, joint localization, belief tracking, probabilistic prediction,
  slow MPC, and 16 fast beam updates through timestamped interfaces.
- Online modules received only complex waveforms, noisy visual candidates, nominal pose beliefs,
  and previous estimates. Environment ground truth entered only post-output evaluation logging.
- RF/joint localization RMSE was `3.030/2.466 m`; UAV motion was `10.000 m`, and the changed UAV
  geometry affected later observations. All values and timestamps were finite and all constraints
  held. Gate 10 passed; all 44 tests pass. The downstream LoRA regression is recorded in
  `docs/LORA_DOWNSTREAM_REGRESSION.md`. See `docs/PHASE10_DECISION.md` and
  `results/phase10_end_to_end.csv`.

Next: Stop at QUICK completion. Do not switch `mode: quick` to FULL or run publication-scale
experiments until the user explicitly authorizes formal experimental design.

Reproducible quick-test command:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_phase0_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase0_geometry.py
PYTHONPATH=src .venv/bin/python experiments/run_phase1_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase1_rf_fim.py
PYTHONPATH=src .venv/bin/python experiments/run_phase2_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase2_shared_pose_efim.py
PYTHONPATH=src .venv/bin/python experiments/run_phase3_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase3_scan.py
PYTHONPATH=src .venv/bin/python experiments/run_phase4_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase4_map_estimator.py
PYTHONPATH=src .venv/bin/python experiments/run_phase5_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase5_association.py
PYTHONPATH=src .venv/bin/python experiments/run_phase6_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase6_detector_calibration.py
PYTHONPATH=src .venv/bin/python experiments/run_phase6_lora_sweep.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase6_lora.py
PYTHONPATH=src .venv/bin/python experiments/run_phase7_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase7_prediction.py
PYTHONPATH=src .venv/bin/python experiments/run_phase8_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase8_control.py
PYTHONPATH=src .venv/bin/python experiments/run_phase9_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase9_waveform.py
PYTHONPATH=src .venv/bin/python experiments/run_phase10_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase10_tracking.py
.venv/bin/python -m pytest -q
```
