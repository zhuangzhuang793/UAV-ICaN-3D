# Phase R0 decision

Status: **PASS**

The frozen merged LoRA detector was loaded directly and rerun on all 30 synchronized QUICK
frames. No LoRA training or Phase 7/8 training was performed.

- Detector: `checkpoints/phase6_lora_best.pt`
- Detector SHA-256:
  `2c55ddd640fbb04c683b12e1126efb70cdbcb443bad39f1dff1c3df24ed3d541`
- Phase 6 RF-only / fused 3-D RMSE: `1.912 / 0.988` m
- Phase 6 RF-only / fused Z-RMSE: `0.801 / 0.589` m
- Phase 6 visual updates / correct associations: `20/20 / 19/20`
- Phase 9 RF-only / fused 3-D RMSE: `2.804 / 2.033` m
- Phase 9 RF-only / fused Z-RMSE: `2.131 / 1.653` m
- Phase 10 RF-only / joint localization RMSE: `3.030 / 2.466` m
- Phase 10 visual associations: `8/8`
- Full test suite: `44 passed`

All numeric CSV outputs were checked for finite values. The online waveform estimator accepts
only received complex samples, and the Phase 10 localization function receives the sensor
observation, nominal pose and frozen uncertainty models; environment truth is used only after
online outputs exist for evaluation logging.
