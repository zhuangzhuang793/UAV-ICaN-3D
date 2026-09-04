# FULL target association experiment

Status: **PASS** (Gate F4)

The visual-only baseline used the frozen detector and calibration with a constant-velocity image
Kalman tracker and Hungarian assignment. It had no RF or ground-truth input. The proposed online
method projected the RF target/shared-pose belief into the image and applied a 99% Mahalanobis
gate. Its callable association interface has no served-target identity or target-position input.

Before correctness was evaluated, all 3200 held-out frames were assigned to fixed subsets. A Hard
frame contains at least one detector candidate matched offline to a non-served GT vehicle whose
calibrated antenna candidate lies inside the RF gate. The frozen subset has SHA-256
`6a51776cb40ab25b5e7eb95712d4430bd4d678e0987ce79d428cfe31fa301952`.

| Method | Easy (2538) | Hard (662) |
|---|---:|---:|
| Visual CV + Hungarian | 15.29% | 7.25% |
| RF-guided gating | 97.91% | 86.56% |
| Evaluation-only oracle | 99.92% | 98.19% |

- Proposed visual-update rate: `98.38%`
- Incorrect proposed associations saved: `142`
- Online GT fields: none

RF guidance decisively outperformed the visual-only temporal tracker on Hard frames, so Gate F4
passes. The pre-registered F8 target of at least 90% Hard accuracy was **not achieved**; this result
is frozen and the subset definition will not be altered. The oracle gap shows that the remaining
error is primarily association rather than detector availability.
