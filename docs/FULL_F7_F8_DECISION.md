# FULL F7/F8 final localization decision

Status: **F8 FAIL — stop before F9**

The frozen F7 experiment evaluated all 3200 held-out Cosys-AirSim frames with five independent
0 dB waveform seeds per sequence. This produced 16000 paired RF realizations for the primary
comparison. All 64000 frame/seed/method records are unique and finite.

## Primary localization result

| Method | 3-D RMSE (m) | Z-RMSE (m) | 95th percentile (m) | Mean NEES | 95% coverage |
|---|---:|---:|---:|---:|---:|
| RF-only full 3-D | 1.634 | 1.214 | 2.910 | 2.319 | 97.82% |
| RF+Vision, shared pose | 1.506 | 1.184 | 2.865 | 3.236 | 94.23% |

The paired 3-D RMSE reduction is `7.86%`. Its sequence-cluster bootstrap 95% interval is
`[6.29%, 9.29%]` from 10000 frozen resamples, so the gain is consistently positive but smaller
than the pre-registered `15%` material-improvement threshold. Z-RMSE improves by `2.46%`, below
the pre-registered `10%` target. The shared-pose method's `94.23%` coverage remains inside the
preferred `90%–98%` range.

## Frozen ablations

| Ablation | 3-D RMSE (m) | Z-RMSE (m) | Mean NEES | 95% coverage |
|---|---:|---:|---:|---:|
| Optimistic fixed pose + Vision | 1.506 | 1.184 | 7.786 | 68.91% |
| Range + azimuth + Vision | 6.390 | 3.113 | 297.393 | 79.03% |

The fixed-pose ablation has essentially the same point RMSE but severely under-covers, whereas
the shared-pose covariance is calibrated. Removing elevation produces much larger errors and
poor uncertainty calibration, supporting the full 3-D array observation rather than reduced
Version A.

## Association criterion

The directly comparable, frozen F4 Hard-subset RF-guided association accuracy is `86.56%`, which
outperforms the visual-only tracker's `7.25%` but misses the `90%` target. The F7 repeated-waveform
diagnostic is `95.74%` on Hard frames; it is retained as a diagnostic and does not replace the
pre-frozen F4 criterion.

## Mechanical F8 decision

- 3-D RMSE reduction at least 15%: **FAIL** (`7.86%`).
- Paired 95% improvement CI lower bound above zero: **PASS** (`6.29%`).
- Z-RMSE reduction at least 10%: **FAIL** (`2.46%`).
- Joint nominal coverage within 90%–98%: **PASS** (`94.23%`).
- Frozen Hard association accuracy at least 90%: **FAIL** (`86.56%`).

The formal result is therefore a statistically consistent but not materially large localization
improvement under the pre-registered definition. In accordance with the global serial-gate rule,
the result is preserved without threshold changes and execution stops at F8. F9 figures and F10
final-report work were not started.
