# FULL trajectory visualization diagnostic

Classification: **post-F8 exploratory diagnostic; not formal F9**

This diagnostic uses the already-frozen artifacts without retraining, rerunning inference, or
changing any gate. It covers all 4800 AirSim frames and all 3200 held-out frames. For F7 it reads
both primary methods over all five waveform seeds, or 32000 raw trajectory records.

## Files

- `results/full_localization/diagnostics/full_process_overview.pdf`: detector training history,
  VisDrone validation history, AirSim world trajectories, and split geometry coverage.
- `results/full_localization/diagnostics/f7_trajectory_overview.pdf`: XY trajectory overview for
  all eight held-out sequences.
- `results/full_localization/diagnostics/f7_trajectory_details.pdf`: one page per test sequence,
  with XY trajectory, vertical trajectory, and framewise 3-D error.
- `full_split_trajectory_source.csv`, `f7_trajectory_source.csv`, and
  `detector_training_history.csv`: source data corresponding to the figures.

In the test figures, faint lines are the five individual waveform-seed trajectories. Bold lines
are their mean. Reported RMSE values still use all individual frame/seed results and are not
computed from the visually smoother mean trajectory.

## Reliability assessment

The pipeline is internally consistent: all counts match the frozen protocol, complete sequences
rather than neighboring frames define the splits, and the plotted RF+Vision improvement is small
and uneven in exactly the way reported by F7. There is no visual evidence of a missing sequence,
coordinate-frame discontinuity, or a trajectory/result join error.

The detector history does not show classic training overfit. Training and validation box/class
losses decrease together; validation mAP50-95 rises from `0.15777` to its best value `0.25765` at
epoch 20. Because the best metric occurs at the maximum allowed epoch, the detector may not be
fully saturated, but checkpoint selection itself is coherent and validation-based.

The split is leakage-resistant at the sequence level but is not a cross-scene benchmark. The
served vehicle follows nearly the same world route in all splits, while UAV trajectories and
relative viewpoints differ. Calibration covers horizontal/vertical separations of
`28.64–40.22 / 29.80–49.90 m`; validation covers `10.88–23.33 / 29.70–49.90 m`; final test spans
the much broader `2.79–48.24 / 25.10–65.80 m`. Thus the final test legitimately probes
extrapolation, but the paper must disclose this geometry shift and should not claim generalization
to unseen maps or vehicle routes.

The estimates are visibly framewise and noisy rather than temporally smoothed. RF+Vision reduces
the RMS frame-to-frame error change from `2.312 m` to `2.091 m`, but substantial jitter remains.
This supports the earlier diagnosis that causal multi-frame localization is the most defensible
next method extension. It also confirms that the `7.86%` aggregate gain is not hidden by a plotting
artifact: sequence-level 3-D gains visibly range from about `4.0%` to `10.7%`.

Overall assessment: the frozen experiment is credible as a controlled, single-map simulation
study with independent sequence splits. It is not yet sufficient by itself for a broad real-world
or cross-environment claim. A stronger paper should add new-map trajectories, multiple training
seeds, and a newly frozen held-out set after any temporal-localization development.
