# Phase 10 final end-to-end smoke test

Status: **PASS**

The short online loop connected SRS reception, waveform range/2-D-AoA estimation, 3-D RF belief,
image projection, served-target association, joint localization, six-state belief tracking,
probabilistic trajectory prediction, slow robust MPC, and fast 3-D UPA beam selection.

- Waveform/localization/prediction/slow-control steps: `8`
- Successful visual associations: `8/8`
- Fast beam updates: `16`
- RF / joint localization RMSE: `3.030 / 2.464` m
- Mean logged prediction uncertainty:
  `2.506` m
- Nominal UAV motion: `10.000` m
- New geometry changed later observations: `True`
- All timestamps/values finite and constraints satisfied: `True`

`QuickClosedLoopEnvironment` alone owns UE/UAV truth and emits only complex SRS samples plus noisy
visual candidates. `_online_localize`, the tracker, predictor, MPC, and beam selector receive no
truth. Ground truth appears only in the evaluation logger columns explicitly suffixed
`evaluation_only` or in localization-error calculations after the online outputs exist.
