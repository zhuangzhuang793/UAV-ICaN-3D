# FULL localization data freeze

Status: **PASS**

The formal dataset contains 12 disjoint 400-frame Cosys-AirSim sequences. Splits are made only at
complete-sequence boundaries. RGB, camera/UAV pose and evaluation-only target records are
synchronized in each manifest.

## Frozen sequences

- `seq00` (calibration, 400 frames): `6cbd1f95eb5baa99857fb6ea341f08c28f0cc9fc44da3bda0267c95faf41a7a3`
- `seq01` (calibration, 400 frames): `ec4647177b69a2d5a328a0c5cbbdac674e7d8204c948ac44631a6e293ad96677`
- `seq02` (validation, 400 frames): `ebc02a0196ce140cdad2e248943693d963052cc41dbaf6701f46cb8a73ceb7fe`
- `seq03` (validation, 400 frames): `cea307fc54812e13df8bec575d9b874049f12f83413ce9e2fc3b1e5531e4fe14`
- `seq04` (test, 400 frames): `cf968ae9716cbe4d0834fe00729b9487912ac97c9bc81ad30157a4d048587029`
- `seq05` (test, 400 frames): `a6338f6ebf9fedc1622bb6392cb8979cd8ee70fe2e0bb3900f4a0973b1cbae5c`
- `seq06` (test, 400 frames): `4e35f6433f3add66003c15228b801367003a0a5aa67b610ab217ad58bd8cf3fc`
- `seq07` (test, 400 frames): `dcadff9c930a0473f937c4256fb6ff8eddeb0f607de12bb7abfea5bd01e747fc`
- `seq08` (test, 400 frames): `b596f532ea0aefd53fc136245d1d1d4137b344032c15e95a87815e7ded38ac8c`
- `seq09` (test, 400 frames): `d9ec76e7d070bd052f237383bc2e540f09212e89caf2003dce30ee1874b61f09`
- `seq10` (test, 400 frames): `73c0ad0f6ac74fe2799dae8dd890e9584cb8c2ca9e034fa8b2d6ae26b3c7e598`
- `seq11` (test, 400 frames): `2e98d9e8ab1f6cccc887d3b8a03b82ba3621b761b5e0d45d46f6b28f038b851d`

## Frozen detector

- Checkpoint: `checkpoints/full_localization_lora.pt`
- SHA-256: `c1a8f9176df331cdf5f695b482322f8f60f3ad0933a4a9a1dca2ae143febb6d1`
- LoRA rank / alpha / dropout: `4 / 8 / 0.05`
- Training / validation images: `6471 / 548`
- Selection metric: `validation mAP50-95`
- Validation mAP50-95: `0.257586`

AirSim test sequences were not used for detector checkpoint selection.
