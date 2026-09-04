# Phase 3 modality complementarity decision

Status: **PASS**

The QUICK scan evaluated 324 non-extreme combinations. It used UAV altitudes of
30–100 m, below the common 400 ft (about 122 m) small-UAS ceiling documented by the
[FAA](https://www.faa.gov/newsroom/small-unmanned-aircraft-systems-uas-regulations-part-107).
The camera model follows the pinhole/reprojection convention documented by
[OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). RF and camera error values are
explicit engineering scan assumptions, not claimed hardware measurements; Phase 6 and Phase 9
must replace them with empirical detector and waveform-estimator residual covariances.

Gate conditions required at least 20% PEB and ZEB improvement in at least four reasonable cells,
spanning at least two altitudes and two horizontal distances. "Reasonable" excludes perfect pose:
attitude standard deviation is 0.5–1.0 degrees, range standard deviation is at most 1 m, AoA
standard deviation is at most 2 degrees, and pixel standard deviation is 2 px.

- Stable reasonable cells: 31
- Stable geometries: 9 across 3 altitudes and 3 distances
- Vision-only target-information rank: [2]
- Reasonable stable PEB-gain range: 35.55%–70.11%
- Overall scan maximum (reported only for audit, not used to pass): 91.85%

Vision-only remains rank deficient for unconstrained 3-D depth, while RF-only is full rank and the
joint model materially improves its bounds. Thus the observed gain is complementary rather than a
vision-only replacement of a deliberately weakened RF baseline.

Decision: Proceed to Phase 4.
