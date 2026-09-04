# Cosys-AirSim QUICK setup

Gate 5 uses the official Cosys-AirSim `5.8-v3.4.1` Linux Blocks package and Python client:

- Release: <https://github.com/Cosys-Lab/Cosys-AirSim/releases/tag/5.8-v3.4.1>
- Runtime archive SHA-256:
  `26ae542561b37c24c7cacf39797b7c0ab3d5dce661228a33c087d3bcbc25d6ca`
- Python wheel SHA-256:
  `6c899b4b413dace7dc0eb6c1ed5aadae06b2612c38e584d79d3dc3f7ac952d16`

The local, non-vendored installation is:

```text
/home/xwl/code/simulators/cosys-airsim-5.8-v3.4.1/
├── downloads/Blocks_packaged_Linux_58_341.zip
├── downloads/python_api_client_341.whl
└── runtime/Linux/Blocks.sh
```

The release wheel has a nonstandard filename. Install it into the existing uv environment without
sudo after copying it to a PEP-compatible name:

```bash
cp /home/xwl/code/simulators/cosys-airsim-5.8-v3.4.1/downloads/python_api_client_341.whl \
  /tmp/cosysairsim-3.4.1-py3-none-any.whl
UV_CACHE_DIR=/tmp/uav_ican_uv_cache uv pip install \
  --python .venv/bin/python /tmp/cosysairsim-3.4.1-py3-none-any.whl
```

Start the headless, GPU-backed scene from the runtime directory:

```bash
./Blocks.sh -RenderOffScreen -NoSound -Unattended -NoSplash \
  -graphicsadapter=0 \
  -settings=/home/xwl/code/uav_ican_3d/configs/cosys_airsim_quick.json
```

Then generate and evaluate the small synchronized set from the project directory:

```bash
PYTHONPATH=src .venv/bin/python experiments/generate_cosys_phase5_data.py \
  --frames 30 --output-dir data/synchronized_quick
PYTHONPATH=src .venv/bin/python experiments/run_phase5_gate.py --config configs/quick.yaml
```

The host has eight NVIDIA RTX 4090 GPUs and driver `570.133.07`. UE 5.8 warns that a newer driver
is preferred, but Vulkan offscreen rendering, RGB capture, segmentation, and RPC all completed.
The packaged simulator crashed in `WorldSimApi::createNewBPActor` when dynamically spawning a car
blueprint. The reproducible scene avoids that upstream path by declaring all three PhysX cars in
the settings file at startup.

The final synchronized export uses a `-60°` pitch instead of a nadir camera. The first nadir run
passed association but provided almost no Z-axis improvement in Gate 6; the oblique geometry
restores the visual/RF complementarity predicted in Phase 3.

Phase 6 uses the official Ultralytics `yolo11n-obb.pt` DOTA aerial detector on CPU. The downloaded
weight SHA-256 is `b62898ebf38940ca4df323863e45ee9d84a1a46d5d11ebdde529fb33aa9f3a32`.
The installed PyTorch build is CUDA 13.0 while driver 570 exposes CUDA 12.8 compatibility, so
PyTorch detector inference deliberately stays on CPU; this does not affect the UE Vulkan runtime.
