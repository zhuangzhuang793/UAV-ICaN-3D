# LoRA downstream regression

Status: **PASS**

The downstream QUICK workflow was rerun in order after selecting the Phase 6 `r4_a8` LoRA
detector. Phase 7 used the existing checkpoint for inference-only regression; it was not
retrained.

- Phase 7 checkpoint NLL / 3-D coverage: `0.4101 / 0.890`
- Phase 7 high/low covariance output-std ratio: `2.842`
- LoRA / original maximum visual-covariance eigenvalue ratio: `0.717`
- Phase 8 slow / fast updates: `6 / 18`; constraints and uncertainty response passed
- Phase 9 detector cache: `data/synchronized_quick/yolo11n_obb_lora_detections.jsonl`
- Phase 9 waveform RF-only / fused 3-D RMSE: `2.804 / 2.033` m
- Phase 9 waveform RF-only / fused Z-RMSE: `2.131 / 1.653` m
- Phase 10 RF-only / joint localization RMSE: `3.030 / 2.466` m
- Phase 10 visual associations / slow steps: `8 / 8`
- Full test suite: `44 passed`

The LoRA calibration covariance has a smaller maximum eigenvalue than the original detector
calibration, while the saved Phase 7 predictor retains its expected covariance response and
coverage. Therefore the configured input change did not trigger Phase 7 retraining. Phase 9 now
selects its detector cache explicitly, and Phase 10 uses the LoRA-calibrated pixel covariance.
