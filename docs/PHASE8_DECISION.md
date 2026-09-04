# Phase 8 decision

Status: **PASS**

One 6-step slow loop consumed Kalman belief histories with the saved
Phase 7 predictor, sampled complete future trajectories, and selected feasible robust-MPC actions.
Each slow update drove 3 uncertainty-aware beam updates on
a physical 4x4 UPA codebook with 4x4 narrow and 2x2 broad tapers.

- Slow / fast updates: `6 / 18`
- All values finite and UAV constraints satisfied: `True`
- Low-covariance beam: aperture `4`, angles
  `[-0.2617993877991494, 0.7853981633974483]` rad
- High-covariance beam: aperture `2`, angles
  `[-0.2617993877991494, 0.5235987755982988]` rad
- Beam selection changed with angular covariance: `True`

The controller receives only filtered beliefs and predicted distributions. Environment truth is
used to generate noisy measurements and log error, not as predictor, MPC, or beam-selector input.
