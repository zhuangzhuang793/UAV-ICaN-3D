# FULL observation-level localization

Status: **PASS**

The existing nonlinear joint target/shared-pose MAP estimator was evaluated on the eight F2
geometries frozen before Monte Carlo. Each geometry used 500 independent trials. RF-only and
RF+Vision received the identical RF realization and the same pose prior in every paired trial.

- Geometries improving 3-D RMSE: `8/8`
- Geometries improving Z-RMSE: `8/8`
- Theoretical PEB gain versus empirical 3-D gain Spearman correlation: `1.000`
- Theoretical ZEB gain versus empirical Z gain Spearman correlation: `0.976`
- RF-only mean NEES range: `2.724–3.191`
- RF+Vision mean NEES range: `2.782–3.204`
- RF-only nominal 95% coverage range: `93.6%–95.8%`
- RF+Vision nominal 95% coverage range: `93.0%–96.2%`

The frozen low-gain geometry also retained the expected modest behavior: theoretical PEB gain
`15.1%`, empirical 3-D RMSE gain `15.8%`, and empirical Z-RMSE gain `4.5%`. Clear
complementarity cells produced larger gains without covariance collapse. Thus the F2 trend carries
into nonlinear estimation and the covariance remains calibrated at observation level.

All 4000 paired trial records are stored in `results/full_localization/f3_paired_trials.csv`; the
per-geometry metrics are in `f3_geometry_summary.csv`.
