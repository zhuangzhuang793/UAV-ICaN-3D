# Phase 5 blocker

Status: **BLOCKED**

The RF image-belief projection and Mahalanobis association components passed their geometry smoke
test. The local VisDrone adapter also read 30 frames containing
321 car/van/truck/bus GT candidates.

VisDrone cannot satisfy the Phase 5 GT-first gate because it has no synchronized UAV pose, camera
pose/calibration, vehicle 3-D reference, UE antenna lever arm, or RF observation. No local
AirSim/Cosys-AirSim scene or alternate synchronized dataset was found.

Expected manifest: `data/synchronized_quick/manifest.jsonl`

Do not infer 3-D ground truth by selecting a VisDrone bbox and placing a synthetic vehicle on its
back-projected ray. That would make the projection test circular. Provide a genuinely synchronized
small scene following `data/README.md`, or authorize setup of a compatible simulator. Detector
inference and Phase 6 must not begin before this gate is resolved.
