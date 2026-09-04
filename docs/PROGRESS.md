# UAV-ICaN 3D progress

Current phase: Phase 5 — RF-guided visual target association

Gate result: BLOCKED

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
- All 27 tests through the available Phase 5 components passed, but the formal GT-first Gate is
  blocked because no independently synchronized pose/image/vehicle dataset or simulator exists.
- See `docs/PHASE5_BLOCKER.md` and `data/README.md` for the blocker and required data contract.

Next phase: Stay in Phase 5. Provide a small synchronized dataset following `data/README.md`, or
authorize setup of a compatible simulator. Phase 6 is forbidden until Gate 5 passes.

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
```
