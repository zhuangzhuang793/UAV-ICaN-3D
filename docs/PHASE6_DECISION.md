# Phase 6 decision

Status: **PASS**

YOLO11n-OBB pretrained on the aerial DOTA task supplied real small/large-vehicle detections. The
first 10 synchronized frames calibrated bbox-center-to-antenna bias and a full
anisotropic pixel covariance from 10 GT matches. No detector confidence
was reinterpreted as pixel variance.

- Calibration bias `[u, v]`: `[-0.46339695837489786, 3.8270732106514087]` px
- Calibration covariance: `[[4.844950210282485, -0.039905564191527354], [-0.039905564191527354, 10.94677215117185]]` px²
- Evaluation visual updates: `19/20`
- Correct RF-guided detector associations: `18/20`
- RF-only / RF+real-Vision 3-D RMSE: `1.912 / 1.134` m
- RF-only / RF+real-Vision Z-RMSE: `0.801 / 0.625` m
- Relative 3-D / Z gains: `0.407 / 0.220`
- Deliberate far visual outlier rejected before fusion: `True`

The detector center is bias-corrected against the independently projected antenna pixel and uses
the empirical residual covariance; it is not directly declared to be the antenna phase center.
Mahalanobis pre-fusion gating falls back to RF-only when a visual candidate is an outlier.
