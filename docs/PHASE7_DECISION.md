# Phase 7 decision

Status: **PASS**

A lightweight Gaussian trajectory network was quick-trained on 600 synthetic
belief histories containing equal straight and turning examples. It predicts a future joint
trajectory mean and diagonal covariance. The full model encodes covariance histories and includes
an explicit kinematic covariance-propagation path; the ablation receives means only.

- Validation full / mean-only NLL: `0.4101 / 0.5828`
- Full-model NLL gain: `0.1727`
- Target / full / mean-only 3-D coverage:
  `0.900 / 0.890 / 0.899`
- High/low input-covariance output-std ratio: `2.842`
- Learned kinematic covariance scale: `0.0481`
- Distribution change under the covariance intervention: `1.123`
- QUICK epochs: `60`
- Local checkpoint: `checkpoints/phase7_full.pt`

Ground truth is used only to quick-train and evaluate this synthetic predictor. Its online API
accepts belief means and covariances; no future state or simulator truth is an input.
