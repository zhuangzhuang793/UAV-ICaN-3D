# FULL localization final-test freeze

Frozen on: `2026-09-05` (Asia/Shanghai)

Status: **FROZEN — final F7 results have not been opened**

The executable protocol is commit `3163510b5acd29d2a159588ea7de4e5f53d21387`. From this
point onward, final-test results may be reported but must not be used to tune algorithms,
thresholds, covariance models, seeds, or sample definitions.

## Detector and real-Vision inputs

- Detector checkpoint: `checkpoints/full_localization_lora.pt`
- Detector SHA-256: `c1a8f9176df331cdf5f695b482322f8f60f3ad0933a4a9a1dca2ae143febb6d1`
- Frozen visual calibration: `results/full_localization/f1_visual_calibration.json`
- Visual calibration SHA-256: `2864cba465f70aa6c8fb43c56597b43f0f21de654735e80076aeb39076a4aaf7`
- Selected visual model: `C0`, with full frozen 2-D pixel covariance
- Test detection cache SHA-256: `9e5e521177b56f17abda382bc006db55ac946910bb47971d541cb05b5112d700`
- Easy/Hard subset freeze SHA-256: `6a51776cb40ab25b5e7eb95712d4430bd4d678e0987ce79d428cfe31fa301952`

## RF and localization implementation

- F7 waveform model: analytic LoS SRS-like complex array waveform at `0 dB`
- Carrier / array: `3.5 GHz / 4x4 UPA`
- RF observation: range, azimuth, elevation; empirical bias and full covariance frozen in F5
- F5 summary SHA-256: `8b4d6e7a276af09d70203c7221e9d7706f5ee783649a29c9765ed520942206b7`
- RF waveform estimator SHA-256: `db492c8352809124806d0fa5c57dd762c41b28b38ac8972c8f4cdd99c8d768f2`
- Nonlinear MAP estimator SHA-256: `a11218ead465779268b7f426b19e92d05588b4077a6a7190b6bb102b2346fb47`
- Online F7 pipeline SHA-256: `f734543f1f6d011dadbe634eadf1174cc36cfa91af5ae55888b19ad72b5b128a`
- RF-guided association SHA-256: `9737ff8957687bb59fab170bb6d29a04aca366373c3df9533b52e907bfdaf39a`
- Final runner SHA-256: `3d06b0db64b05d930fc2109457e1d5b9741a0b419c7a9bea4a4d02645decb116`

Sionna RT 2.0.1 is frozen as the separate F6 propagation-realism stress test. Its scene SHA-256 is
`e6b932006d97505a076e475935cc691e6b7e051fc019f4a9d874175d222179e2`. F7 does not consume
Sionna path delays, angles, or coefficients. Those fields remain behind the complex-waveform
boundary validated in F6.

## Final test IDs and seeds

The final set contains exactly 3200 held-out frames:

| Sequence | Frames | Manifest SHA-256 |
|---|---:|---|
| seq04 | 400 | `cf968ae9716cbe4d0834fe00729b9487912ac97c9bc81ad30157a4d048587029` |
| seq05 | 400 | `a6338f6ebf9fedc1622bb6392cb8979cd8ee70fe2e0bb3900f4a0973b1cbae5c` |
| seq06 | 400 | `4e35f6433f3add66003c15228b801367003a0a5aa67b610ab217ad58bd8cf3fc` |
| seq07 | 400 | `dcadff9c930a0473f937c4256fb6ff8eddeb0f607de12bb7abfea5bd01e747fc` |
| seq08 | 400 | `b596f532ea0aefd53fc136245d1d1d4137b344032c15e95a87815e7ded38ac8c` |
| seq09 | 400 | `d9ec76e7d070bd052f237383bc2e540f09212e89caf2003dce30ee1874b61f09` |
| seq10 | 400 | `73c0ad0f6ac74fe2799dae8dd890e9584cb8c2ca9e034fa8b2d6ae26b3c7e598` |
| seq11 | 400 | `2e98d9e8ab1f6cccc887d3b8a03b82ba3621b761b5e0d45d46f6b28f038b851d` |

Independent RF/waveform seeds used for every sequence are `20260971`, `20260972`, `20260973`,
`20260974`, and `20260975`. Primary comparisons are paired on frame and waveform seed. The
sequence-cluster bootstrap uses seed `20260976` and exactly 10000 resamples.

## Frozen comparisons and criteria

- Primary baseline: RF-only full 3-D.
- Primary method: RF+Vision with one shared UAV pose perturbation.
- Ablation 1: optimistic fixed-pose RF+Vision.
- Ablation 2: range+azimuth+Vision, with candidate compatibility computed without elevation.
- Primary success target: at least `15%` 3-D RMSE reduction and positive paired 95% CI lower bound.
- Z target: at least `10%` Z-RMSE reduction.
- Preferred calibration range: nominal 95% coverage within `90%–98%`.
- Hard-association target: at least `90%` correct; the frozen F4 Hard result was already below this
  target and will not be redefined.

Configuration SHA-256 is
`f182c3d898ebccc594840597eadb776de42a2b3bad9e68a662b2fd08f1833729`. The read-only QUICK
configuration remains
`7be6cdd844a6f0c5694b7acf787934dbc7cd48449c93a77d83f89aedf67547f6`.

## Leakage audit

The pre-freeze audit passed with 63 tests. Its artifact SHA-256 is
`e35e36cae2fd2ee92d83babfe9deaadd2cb531c08453ae10106673790757fc08`.
AST and signature checks found no `user_gt_position`, `served_target_gt_id`, evaluation-only
record, or Sionna path field in the online waveform, association, or localization interfaces.
