# Phase 5 blocker — resolved

Status: **RESOLVED**

The RF image-belief projection and Mahalanobis association components passed their geometry smoke
test. The local VisDrone adapter also read 30 frames containing
321 car/van/truck/bus GT candidates.

VisDrone alone could not satisfy the Phase 5 GT-first gate because it has no synchronized UAV
pose, camera pose/calibration, vehicle 3-D reference, UE antenna lever arm, or RF observation.

The blocker was resolved with a 30-frame Cosys-AirSim 3.4.1 / UE 5.8 synchronized export at
`data/synchronized_quick/manifest.jsonl`. See `docs/PHASE5_DECISION.md` for the passing result and
`docs/COSYS_AIRSIM_SETUP.md` for exact reproduction steps.

The export computes the antenna phase center from the simulator vehicle pose and a fixed lever arm;
it does not infer 3-D truth by back-projecting a visual box. Phase 6 is now unblocked.
