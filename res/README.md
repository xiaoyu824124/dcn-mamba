# VTMOT single-frame IR-visible registration

This is the complete runnable branch. It contains only:

```text
IR + visible_mis
  -> MIND structural descriptors
  -> shared encoder
  -> 1/8 all-pairs global matching
  -> confidence-weighted 6-DoF affine projection
  -> [dy, dx] flow and backward-warped IR
```

The flow convention is fixed throughout:

```text
aligned_ir(y, x) = ir(y + flow_y, x + flow_x)
flow = [dy, dx] pixels on the visible_mis grid
```

## Required VTMOT layout

```text
data/VTMOT_misaligned/
data_split/IVF/VTMOT/split.json
```

`gt_h` maps fixed `visible_mis` coordinates to moving IR coordinates. The
loader applies the same aspect-preserving resize to images and homography, then
materialises dense GT flow. VTMOT files are read-only.

## RTX 3090 training

The default config uses the full 480x640 field of view and batch one. Start
from the retained real-VTMOT affine checkpoint:

```powershell
python -B -m res.train_vtmot --device cuda --run pilot --init res_runs\vtmot_affine_stable_3060\best.pt --output-dir res_runs\vtmot_affine_3090
```

Continue only if validation relative EPE is below one:

```powershell
python -B -m res.train_vtmot --device cuda --run full --resume res_runs\vtmot_affine_3090\best.pt --output-dir res_runs\vtmot_affine_3090

Every validation print also includes `affine_corner_epe_px` and
`affine_gt_inverse_cycle_px`. These are diagnostics only: both should decrease
with EPE, but neither is an additional test-set tuning target.

For a Windows debugging run where worker processes cannot be created, append
`--num-workers 0`; retain the configured two workers for the normal 3090 run.
```

`relative EPE = predicted EPE / zero-flow EPE`; values below one beat no
registration. The held-out `test` split is not used during training.

## Evaluation

First verify GT direction without a model:

```powershell
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 50 --check-gt
```

Then evaluate a checkpoint on the development split:

```powershell
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 50 --checkpoint res_runs\vtmot_affine_3090\best.pt
```

After choosing settings, replace `--split eval` with `--split test` once for
the final held-out report.

## Tests

```powershell
python -B -m unittest -v res.tests.test_mind res.tests.test_encoder res.tests.test_global_matcher res.tests.test_warp res.tests.test_registration_net res.tests.test_losses res.tests.test_vtmot
```
