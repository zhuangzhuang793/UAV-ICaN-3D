# FULL visual calibration

Status: **PASS**

The model was fitted on the 800-frame calibration split (`seq00, seq01`) and
selected by mean pixel NLL on the independent 800-frame validation split
(`seq02, seq03`). No held-out test frame was read.

## Selection

- Selected model: **C0**
- Calibration / validation GT-matched detections: `800 / 800`
- C0 validation pixel NLL / RMSE: `4.123548 / 2.633 px`
- C1 validation pixel NLL / RMSE: `8.876864 / 4.340 px`
- C0-minus-C1 NLL: `-4.753316` nats/frame
- Negligible-improvement threshold: `0.010000` nats/frame

## Frozen measurement model

- OBB-center-minus-antenna bias: `[-0.6547020695366703, 2.5080813216528246]` px
- Full anisotropic covariance: `[[2.9980587980412743, -1.0095021178475656], [-1.0095021178475656, 2.3093853153280897]]` px²
- Covariance eigenvalues: `[1.5871095134494184, 3.7203345999199455]` px²
- Formal detector SHA-256: `c1a8f9176df331cdf5f695b482322f8f60f3ad0933a4a9a1dca2ae143febb6d1`
- Machine-readable artifact: `results/full_localization/f1_visual_calibration.json`

C0 is a constant 2-D correction. C1 is ridge regression over normalized OBB center, log size,
log aspect ratio, and doubled-angle sine/cosine features; it outputs only `[delta_u, delta_v]`.
Both covariances are estimated only from calibration residuals. The independently projected
vehicle-pose plus fixed-FRD-lever-arm antenna pixel is the ground truth; detector box centers are
never treated as antenna ground truth.
