# Phase 6 decision

Status: **PASS**

Frozen YOLO11n-OBB checkpoint `checkpoints/phase6_lora_best.pt` supplied real
small/large-vehicle detections. The
first 10 synchronized frames calibrated bbox-center-to-antenna bias and a full
anisotropic pixel covariance from 10 GT matches. No detector confidence
was reinterpreted as pixel variance.

- Calibration bias `[u, v]`: `[-1.8931271829842728, 3.684146422321331]` px
- Calibration covariance: `[[4.570583054924847, 0.5910402178258557], [0.5910402178258557, 7.737659849363675]]` px²
- Evaluation visual updates: `20/20`
- Correct RF-guided detector associations: `19/20`
- RF-only / RF+real-Vision 3-D RMSE: `1.912 / 0.988` m
- RF-only / RF+real-Vision Z-RMSE: `0.801 / 0.589` m
- Relative 3-D / Z gains: `0.483 / 0.265`
- Deliberate far visual outlier rejected before fusion: `True`

The detector center is bias-corrected against the independently projected antenna pixel and uses
the empirical residual covariance; it is not directly declared to be the antenna phase center.
Mahalanobis pre-fusion gating falls back to RF-only when a visual candidate is an outlier.
