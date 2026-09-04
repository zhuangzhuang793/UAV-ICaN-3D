# Synchronized QUICK data contract

VisDrone remains external at the absolute path in `configs/quick.yaml`; do not copy it here.

Phase 5 requires a small independently synchronized dataset at
`data/synchronized_quick/manifest.jsonl`. Each JSON-lines record must contain:

- `timestamp_s`
- `segmentation_timestamp_s` and the RGB/segmentation time delta
- `image_path`
- `camera_intrinsics` (`fx_px`, `fy_px`, `cx_px`, `cy_px`, width, height)
- `transform_world_body` (`R_WB`, `t_W_B`)
- `transform_body_camera` (`R_BC`, `t_B_C`)
- `vehicle_reference_world_m`
- `transform_world_vehicle` (`R_WV`, `t_W_V`)
- `antenna_lever_arm_vehicle_m`
- `antenna_phase_center_world_m` and independently projected `true_ue_pixel_uv`
- `served_gt_box_xyxy`
- all candidate GT boxes and the served-candidate index

Images and large generated metadata are ignored by Git. A simulator export must use the ENU/body
FRD/OpenCV conventions in `docs/COORDINATES.md`, or provide an explicit conversion at ingestion.

The bbox center may only be used in an explicitly marked `TEMPORARY_APPROXIMATION` smoke test. The
formal visual reference plus antenna lever arm is required for the finished association pipeline.

The current QUICK export is generated from Cosys-AirSim 3.4.1 with:

```bash
PYTHONPATH=src .venv/bin/python experiments/generate_cosys_phase5_data.py \
  --frames 30 --output-dir data/synchronized_quick
```

The simulator is paused before pose reads and the batched RGB/segmentation request. Vehicle boxes
come from unique simulator instance-segmentation IDs, while the UE pixel is projected from the
vehicle pose and fixed antenna lever arm. This avoids circularly deriving 3-D truth from a box.
