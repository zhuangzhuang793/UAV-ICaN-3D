# FULL Sionna RT integration

Status: **PASS** (Gates F6-A and F6-B)

Sionna RT 2.0.1 was integrated as a propagation layer around the frozen delay plus 2-D AoA
waveform estimator. The custom scene contains a ground plane and one side wall. The ray tracer's
path delays and complex coefficients remain inside `sionna_channel.py`; they are converted into a
complex 4-by-4-array waveform before the existing online estimator is called. The estimator
receives only `complex_received_waveform` and `known_reference`.

## F6-A: LoS interface sanity

Eight frozen F2 geometries were tested at six SNR levels with 200 waveform realizations per cell,
for 9600 LoS waveforms. Every geometry contained exactly one valid path.

| SNR (dB) | Range RMSE (m) | Azimuth RMSE (deg) | Elevation RMSE (deg) |
|---:|---:|---:|---:|
| -15 | 85.459 | 41.810 | 43.677 |
| -10 | 64.096 | 30.229 | 31.675 |
| -5 | 14.479 | 7.643 | 7.727 |
| 0 | 1.348 | 0.724 | 0.733 |
| 5 | 0.760 | 0.436 | 0.420 |
| 10 | 0.417 | 0.246 | 0.264 |

All three error metrics have Spearman correlation `-1.000` with SNR. At the frozen 0 dB nominal
condition, the Sionna-to-analytic RMSE ratios are `0.957` for range, `0.986` for azimuth, and
`1.031` for elevation. These results pass the frozen trend and `[0.5, 2.0]` nominal-ratio checks,
so reflected paths were enabled only after F6-A passed.

## F6-B: mild multipath stress

First-order specular reflection produced three paths in each geometry, or 16 reflected paths in
total in addition to the eight LoS paths. No new NLoS mitigation was introduced. At 0 dB, the
pooled multipath waveform errors were:

- range bias `1.837 m`, RMSE `3.224 m`;
- azimuth bias `-0.308 deg`, RMSE `10.137 deg`;
- elevation bias `0.178 deg`, RMSE `0.893 deg`.

The 1600 paired localization trials reused each waveform-derived RF realization for RF-only and
RF+Vision. RF-only versus joint 3-D RMSE was `34.753 m` versus `5.452 m`, an `84.31%`
improvement. Z-RMSE was `2.265 m` versus `1.784 m`, a `21.21%` improvement. Propagation bias
degraded the RF measurements but did not eliminate the aggregate RF--Vision benefit, so F6-B
passes.

The scene SHA-256 is
`e6b932006d97505a076e475935cc691e6b7e051fc019f4a9d874175d222179e2`. Detailed channel,
waveform, and paired-localization records are stored under `results/full_localization/` with the
`f6a_` and `f6b_` prefixes.
