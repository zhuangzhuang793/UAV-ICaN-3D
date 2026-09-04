# UAV-ICaN 3D progress

Current phase: Phase 1 — RF 3-D observation and RF-only FIM

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
- The array-axis singular geometry was explicitly detected; all 14 tests through Phase 1 passed.

Next phase: Phase 2 — camera observation and shared-pose EFIM.

Reproducible quick-test command:

```bash
PYTHONPATH=src .venv/bin/python experiments/run_phase0_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase0_geometry.py
PYTHONPATH=src .venv/bin/python experiments/run_phase1_gate.py --config configs/quick.yaml
.venv/bin/python -m pytest -q tests/test_phase1_rf_fim.py
```
