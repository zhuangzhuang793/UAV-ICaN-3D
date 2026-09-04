# Phase 10 final end-to-end smoke test

Status: **PASS**

The short online loop connected SRS reception, waveform range/2-D-AoA estimation, 3-D RF belief,
image projection, served-target association, joint localization, six-state belief tracking,
probabilistic trajectory prediction, slow robust MPC, and fast 3-D UPA beam selection.

- Waveform/localization/prediction/slow-control steps: `8`
- Successful visual associations: `8/8`
- Fast beam updates: `16`
- RF / joint localization RMSE: `3.030 / 2.466` m
- Visual covariance source: `phase6_lora_calibration`
- Visual covariance: `[[4.570583054924847, 0.5910402178258557], [0.5910402178258557, 7.737659849363675]]` px^2
- Mean logged prediction uncertainty:
  `2.496` m
- Nominal UAV motion: `10.000` m
- New geometry changed later observations: `True`
- All timestamps/values finite and constraints satisfied: `True`

`QuickClosedLoopEnvironment` alone owns UE/UAV truth and emits only complex SRS samples plus noisy
visual candidates. `_online_localize`, the tracker, predictor, MPC, and beam selector receive no
truth. Ground truth appears only in the evaluation logger columns explicitly suffixed
`evaluation_only` or in localization-error calculations after the online outputs exist.
