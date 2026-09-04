# FULL theoretical EFIM study

Status: **PASS**

The study reused the validated RF and shared-pose Schur-complement EFIM implementation with the
F1 covariance `[[2.998059, -1.009502], [-1.009502, 2.309385]]` px². It evaluated 70 geometries
over 30–120 m altitude and 20–200 m horizontal distance.

The experiment families remained separate: 70 nominal reference cells, 280 cells at four paired
log-spaced RF quality levels, 350 cells at five log-spaced attitude uncertainty levels, and 70
position-covariance-times-four sensitivity cells. Thus RF and pose settings were not expanded into
one Cartesian grid.

- Nominal reference PEB gain range: `15.07%–65.35%`
- Nominal reference ZEB gain range: `5.34%–73.24%`
- Cells exceeding both pre-registered 10% reference thresholds: `63/70`
- Cells exceeding both 5% thresholds at the worst tested 0.5-degree attitude uncertainty: `70/70`
- Vision-only target information rank: always below 3

The complementarity region therefore persists with empirical camera covariance and nonzero shared
pose uncertainty. It is not confined to unrealistic RF errors or a perfect camera pose.

The eight F3 cells were frozen before Monte Carlo in
`results/full_localization/f2_representative_geometries.json`: one cell per nominal PEB-gain
octile, choosing maximum normalized spatial separation from previously selected cells within each
octile. This includes the lowest-gain cell as well as clear complementarity cells.

Numerical results and figure sources are in `f2_efim_dense.csv` and
`f2_reference_gain_maps.csv`; PEB and ZEB maps are separate PDFs.
