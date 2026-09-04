# UAV-ICaN 3D progress

Current phase: Phase 7 — uncertainty-aware probabilistic prediction

Latest gate result: Phase 6 **PASS**

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
  passed; all 32 tests pass. See `docs/PHASE6_DECISION.md` and
  `results/phase6_real_vision.csv`.

Next phase: Build a lightweight probabilistic trajectory predictor that consumes belief mean and
covariance histories, train only on a small straight/turning synthetic set, and run Gate 7.

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
```
