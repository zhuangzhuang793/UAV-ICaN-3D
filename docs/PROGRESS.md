# UAV-ICaN 3D progress

Current phase: Phase 2 — camera observation and shared-pose EFIM

Gate result: PASS

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

Next phase: Phase 3 — modality complementarity map and research decision Gate.

Reproducible quick-test command:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_phase0_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase0_geometry.py
PYTHONPATH=src .venv/bin/python experiments/run_phase1_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase1_rf_fim.py
PYTHONPATH=src .venv/bin/python experiments/run_phase2_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase2_shared_pose_efim.py
```
