# Synchronized QUICK data contract

VisDrone remains external at the absolute path in `configs/quick.yaml`; do not copy it here.

Phase 5 requires a small independently synchronized dataset at
`data/synchronized_quick/manifest.jsonl`. Each JSON-lines record must contain:

- `timestamp_s`
- `image_path`
- `camera_intrinsics` (`fx_px`, `fy_px`, `cx_px`, `cy_px`, width, height)
- `transform_world_body` (`R_WB`, `t_W_B`)
- `transform_body_camera` (`R_BC`, `t_B_C`)
- `vehicle_reference_world_m`
- `antenna_lever_arm_vehicle_m`
- `served_gt_box_xyxy`

Images and large generated metadata are ignored by Git. A simulator export must use the ENU/body
FRD/OpenCV conventions in `docs/COORDINATES.md`, or provide an explicit conversion at ingestion.

The bbox center may only be used in an explicitly marked `TEMPORARY_APPROXIMATION` smoke test. The
formal visual reference plus antenna lever arm is required for the finished association pipeline.

