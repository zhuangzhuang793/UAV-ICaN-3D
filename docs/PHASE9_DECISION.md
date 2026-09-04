# Phase 9 decision

Status: **PASS**

A 3.5 GHz single-antenna uplink transmitted a known QPSK SRS-like reference to the same 4x4
half-wavelength UPA. A grid maximum-likelihood delay estimator used received subcarrier phases;
a separate 2-D spatial maximum-likelihood stage estimated azimuth and elevation. Ground-truth
path values enter only the channel simulator and evaluation residuals, never the estimator API.

- SNR / equivalent position RMSE: `{-15.0: 14.590106680479064, -5.0: 8.881155126020415, 5.0: 0.9855088278481186}` m
- High-to-low SNR error improvement: `0.932`
- Nominal empirical RF bias `[range, az, el]`: `[0.586666666666874, -0.038106355335209466, -0.007853981633974438]`
- Nominal empirical RF covariance: `[[21.74645402298913, -0.8783058511197154, -0.06712295568101041], [-0.8783058511197154, 0.048775883965443925, 0.0033090380230450973], [-0.06712295568101041, 0.0033090380230450973, 0.000530086830673663]]`
- Waveform RF-only / RF+real-Vision 3-D RMSE: `2.804 / 2.033` m
- Waveform RF-only / RF+real-Vision Z-RMSE: `2.131 / 1.653` m
- Pipeline relative 3-D / Z gains: `0.275 / 0.225`
- Detector cache: `data/synchronized_quick/yolo11n_obb_lora_detections.jsonl`

The empirical waveform-estimator covariance replaces the hand-set RF covariance in the pipeline
subset. The visual inputs are cached real VisDrone-LoRA YOLO11n-OBB detections from Phase 6.
