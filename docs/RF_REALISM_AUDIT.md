# Post-F7 RF realism audit

Status: **COMPLETE — diagnostic post-hoc experiment, not a replacement for frozen F7**

## Question and protocol

This audit tests whether the small visual gain in frozen F7 was partly caused by an overly clean
RF baseline. The detector cache, visual calibration, held-out frame IDs, pose uncertainty,
association threshold, optimizer settings, and five F7 waveform seeds were held fixed. Each
analytic condition contains all 3200 test frames and five RF seeds, giving 16000 paired RF-only
and RF+Vision samples. RF bias and covariance were recalibrated for each condition using only
`seq00` and `seq01`, with ten independent calibration seeds (8000 calibration waveforms per
condition).

The conditions are cumulative:

1. `ideal_fixed_0db`: frozen F7 analytic LoS baseline.
2. `link_budget`: free-space loss, explicit per-subcarrier thermal noise and receiver noise
   figure, plus correlated shadowing.
3. `synchronization`: link budget plus clock bias/drift, residual CFO, motion Doppler, and phase
   noise.
4. `array_calibration`: synchronization plus per-element gain and phase errors.
5. Sionna absolute-power audit: separately evaluated on the eight frozen F2 geometries because
   the available Sionna scene is not the Cosys-AirSim map.
6. `interference`: analytic cumulative condition plus intermittent co-pilot interference.

The link budget is anchored at 0 dB per subcarrier at 50 m. With 32 subcarriers at 480 kHz and a
7 dB receiver noise figure, this corresponds to `-110.188 dBm` noise per subcarrier and a derived
`-17.828 dBm` total pilot-resource transmit power for 0 dBi antennas. This power is an explicit
sensitivity-test anchor, not a measured UAV/UE hardware value.

## Full held-out trajectory results

| Condition | RF range / az / el RMSE | RF-only 3-D / Z RMSE | RF+Vision 3-D / Z RMSE | 3-D gain (95% sequence CI) | Z gain |
|---|---:|---:|---:|---:|---:|
| Ideal fixed 0 dB | 1.408 m / 0.734° / 0.711° | 1.634 / 1.214 m | 1.506 / 1.184 m | 7.86% (6.29%, 9.29%) | 2.46% |
| Link budget | 13.589 m / 6.171° / 4.257° | 17.723 / 8.935 m | 17.062 / 9.085 m | 3.73% (1.20%, 4.78%) | -1.69% |
| + synchronization | 13.555 m / 6.368° / 4.571° | 17.815 / 9.614 m | 16.746 / 10.028 m | 6.00% (1.00%, 9.45%) | -4.30% |
| + array calibration | 12.737 m / 6.339° / 4.443° | 16.920 / 9.067 m | 15.658 / 9.597 m | 7.46% (4.59%, 9.12%) | -5.85% |
| + co-pilot interference | 12.401 m / 11.480° / 6.604° | 20.165 / 10.456 m | 16.709 / 10.080 m | 17.13% (12.64%, 18.69%) | 3.59% |

The link-budget conditions have mean test SNR `-0.588 dB`, but the mean hides a threshold effect:
1971/16000 samples are below `-5 dB`. Under link budget alone, 1.8% of RF-only samples exceed
20 m error, the 99th percentile is 103.47 m, and the maximum is 270.33 m. These rare wrong-delay
or wrong-angle peaks dominate RMSE even though the 95th percentile remains 4.89 m.

The visual gain is strongly conditional on SNR and range. Under link budget alone, 3-D gain is
approximately `-0.3%`, `1.7%`, and `4.7%` for ranges below 50 m, 50--65 m, and above 65 m. In the
interference stress condition the corresponding gains are `7.5%`, `15.8%`, and `19.0%`. Thus the
image is most useful when RF is weak or mismatched, rather than when RF already operates near its
matched-model bound.

Z is still a weakness. Vision reduces the total 3-D long tail in every realism condition, but Z
slightly worsens in the first three analytic realism levels and improves by only 3.59% under
interference. This is consistent with monocular image bearings providing no direct absolute
depth and with the RF elevation/pose coupling remaining important.

Association also becomes a limiting component. Primary association accuracy is 96.45%, 96.61%,
96.24%, and 85.51% across the four analytic realism conditions. RF-to-image projection remains
valid above 99.9%, while usable visual-update rates fall from 98.83% to 97.63%. If RF projects
behind the camera, or the unchanged joint optimizer crosses the camera plane, the executable
online fallback returns RF-only and records the failure; evaluation truth is never used to repair
the estimate.

## Sionna absolute-power check

The prior F6 interface normalized every Sionna channel by its mean received power before adding
noise. The new audit preserves raw path gain and uses the same explicit transmit and thermal-noise
powers as the analytic link budget.

| Sionna condition | Waveforms | Range RMSE | Azimuth RMSE | Elevation RMSE |
|---|---:|---:|---:|---:|
| LoS, raw power | 1600 | 59.772 m | 26.540° | 24.836° |
| Multipath, raw power | 1600 | 55.664 m | 26.427° | 22.790° |

At 34.8 m the LoS effective SNR is 3.14 dB, whereas the 176--225 m cells lie around -11 to
-13 dB. The multipath aggregate is not monotonically worse because reflected energy sometimes
raises received power; individual geometry results must be read together with raw gain, effective
SNR, path count, and bias.

This Sionna result is a propagation stress test, not a full-trajectory claim. The available scene
contains a ground plane and one wall rather than the Cosys-AirSim environment mesh. Diffuse
reflection, refraction and diffraction were enabled, but the scene representation remains the
dominant external-validity limitation.

## Interpretation

The audit supports the original concern: frozen F7 substantially overstates the robustness of RF
outside a matched, fixed-SNR LoS model. It does **not** establish one definitive real-world RMSE,
because the 50 m/0 dB power anchor and hardware-error magnitudes are declared sensitivity
settings, not measurements from a particular radio.

It also shows that “make RF weaker and visual gain becomes larger” is not a sufficient paper
claim. The physically declared conditions produce non-monotonic estimator threshold effects;
some levels improve total 3-D error while worsening Z, and the strongest visual gain coincides
with lower association accuracy. A paper should therefore report absolute errors, long-tail
quantiles, association/update rates, range/SNR strata, and the declared link budget alongside
relative fusion gain.

## Reproduction

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m experiments.run_rf_realism_ladder \
  --config configs/rf_realism_ladder.yaml

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src:. MPLCONFIGDIR=/tmp/uav_ican_mpl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m experiments.run_sionna_absolute_power_audit \
  --config configs/rf_realism_ladder.yaml
```

Compact comparison data are in
`results/rf_realism_ladder/rf_realism_comparison.csv`. Per-condition summaries, calibration
artifacts, and the Sionna cell table are stored beside it. Large frame-level and restartable
checkpoint files are intentionally ignored by Git.
