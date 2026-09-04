# Phase 5 decision

Status: **PASS**

The synchronized QUICK set contains 30 Cosys-AirSim 3.4.1 / UE 5.8 frames. RGB,
external-camera pose/calibration, three vehicle poses, and instance-segmentation GT boxes were
captured while simulation time was paused. The served UE phase center is computed from the vehicle
reference and a fixed FRD lever arm; it is not inferred by back-projecting a box.

- Target confidence: `0.950`
- Empirical true-UE pixel coverage: `1.000`
- Unguided image-center baseline association rate: `0.000`
- RF-guided Mahalanobis association rate: `0.967`
- Absolute guided gain: `0.967`

GT boxes are used as detector candidates in this gate. Candidate reference pixels remain the
explicitly marked `TEMPORARY_APPROXIMATION` bbox centers; the independently projected antenna
pixel is used for the confidence-coverage test. A detector/keypoint model may replace the GT
candidates in Phase 6 without changing the RF image-belief geometry.
