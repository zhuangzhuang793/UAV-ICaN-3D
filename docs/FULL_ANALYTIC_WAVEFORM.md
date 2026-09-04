# FULL analytic LoS waveform benchmark

Status: **PASS** (Gate F5-A)

The frozen delay plus 2-D AoA grid-ML estimator processed 9600 independent complex SRS-like
waveforms: eight F2 geometries, six SNR levels from -15 to 10 dB, and 200 realizations per cell.
Its online boundary received only the complex array waveform and known reference symbols.

| SNR (dB) | Range RMSE (m) | Azimuth RMSE (deg) | Elevation RMSE (deg) |
|---:|---:|---:|---:|
| -15 | 84.089 | 41.508 | 44.539 |
| -10 | 64.077 | 30.536 | 32.271 |
| -5 | 17.443 | 7.748 | 9.094 |
| 0 | 1.408 | 0.734 | 0.711 |
| 5 | 0.777 | 0.416 | 0.422 |
| 10 | 0.437 | 0.239 | 0.273 |

Each error metric has Spearman correlation `-1.000` with SNR. At the frozen nominal 0 dB level,
the pooled bias in `[range_m, azimuth_rad, elevation_rad]` is
`[-0.039116, -0.000529, -0.000230]`; the empirical covariance is
`[[1.980861, 4.684e-5, 6.138e-5], [4.684e-5, 1.64078e-4, 3.523e-6],
[6.138e-5, 3.523e-6, 1.53994e-4]]`. Off-diagonal terms are preserved.

The estimator trend is physically sensible and the empirical covariance is finite positive
definite. Sionna integration is therefore allowed to begin. Detailed errors, per-geometry
summaries and the aggregate SNR curve are stored as CSV files under
`results/full_localization/`.
