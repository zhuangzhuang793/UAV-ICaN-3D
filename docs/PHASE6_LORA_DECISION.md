# Phase 6 LoRA decision

Status: **PASS**

VisDrone-LoRA YOLO11n-OBB (r4_a8) supplied real small/large-vehicle detections. The
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

## LoRA quick sweep

VisDrone car/van labels were mapped to DOTA small-vehicle class 10; truck/bus labels were mapped
to large-vehicle class 9. Horizontal boxes were encoded as valid four-corner OBB labels. The
source images remained read-only and were linked rather than copied.

- Training subset: `1618/6471` images
- Validation subset: all `548` images
- Training schedule: `5` epochs at `640` px
- Runtime: `torch 2.7.0+cu126`, `torchvision 0.22.0+cu126`, `ultralytics 8.4.138`
- Adapted modules: neck and OBB head convolutions; backbone and BN statistics frozen
- Winner selected only by held-out VisDrone validation fitness: `r4_a8`
- The 20 Gate 6 evaluation frames were used once after selection, not for hyperparameter search

| Candidate | Rank | Alpha | Dropout | LR | Trainable | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 0 | 0 | 0.060 | 0.028 |
| r4_a8 | 4 | 8 | 0.05 | 0.001 | 92,608 | 0.262 | 0.160 |
| r8_a16 | 8 | 16 | 0.05 | 0.0005 | 185,216 | 0.243 | 0.148 |
| r16_a32 | 16 | 32 | 0.1 | 0.0002 | 370,432 | 0.223 | 0.131 |

The winning adapters were merged into a normal Ultralytics OBB checkpoint at
`checkpoints/phase6_lora_best.pt`. This is a QUICK comparison, not a publication-scale detector
benchmark; it uses one seed, 25% of the training images, and five epochs.

Against the original DOTA detector on the same synchronized sequence, the winner changed 3-D
RMSE from `1.134` to `0.988 m`, Z-RMSE from `0.625` to `0.589 m`, visual updates from `19/20` to
`20/20`, and correct associations from `18/20` to `19/20`.
