# VTMOT single-frame IR-visible registration

This is the complete runnable branch. It contains only:

```text
IR + visible_mis
  -> MIND structural descriptors
  -> shared multi-scale encoder (1/2, 1/4, 1/8)
  -> 1/8 all-pairs global matching (dual softmax)
  -> soft-argmax expectation -> raw correspondence field
  -> confidence-weighted 6-DoF affine projection -> coarse flow
  -> [dy, dx] flow and backward-warped IR
```

Only the 1/8 level is consumed today; the 1/4 and 1/2 encoder outputs are
computed and currently discarded. There is no fine/local stage yet, which is the
known accuracy ceiling (see "Current status").

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

## Supervision

`RegistrationLoss` combines five terms:

| weight | term | purpose |
|---|---|---|
| `match` | dual-softmax NLL at the sub-pixel GT cell | the only term that rewards putting mass on the correct key |
| `flow` | Charbonnier on the 6-DoF projected flow | dense sub-pixel shaping |
| `affine` | Charbonnier on the matcher's `[B,3,2]` parameters | direct 6-DoF supervision |
| `mind`, `edge` | descriptor / gradient magnitude after warping | structural agreement |
| `smooth` | edge-aware flow regularity | regularisation |

The dense `flow` term alone has a degenerate optimum (a constant field already
reaches the mean-displacement error), so `match` is what makes the matcher
learn. `match_focal_gamma > 0` switches the NLL to a focal variant.

## Diagnostics

`res.matching` reports how the ground-truth match ranks against competing keys.
Every `evaluate_vtmot` report includes them:

| field | reading |
|---|---|
| `match_frac_keys_beating_gt` | fraction of all 4800 keys that outrank the truth; 0 is perfect, ~0.5 is useless |
| `match_epe_argmax_px` | where the global argmax lands; ~200 px means random |
| `match_effective_keys_ratio` | exp(entropy)/N; 1.0 means a uniform distribution |
| `coarse_error_median_px`, `coarse_error_p90_px` | how far the produced coarse field is from the truth |
| `window{2,4}_coverage` | share of ground-truth keys inside a (2r+1)^2 window around the *predicted* key |
| `window{2,4}_argmax_correct` | share where the truth wins inside that window -- the fine stage's headroom |
| `affine_corner_epe_px` | predicted vs GT affine at the four image corners |

## RTX A4000 training

The default config uses the full 480x640 field of view and batch one on a
16 GB RTX A4000. First check one step on the target machine, then start a
fresh 3000-step run from the retained real-VTMOT affine checkpoint:

```powershell
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --init res_runs\vtmot_affine_stable_3060\best.pt --output-dir res_runs\_a4000_smoke
python -B -m res.train_vtmot --device cuda --run full --num-workers 0 --init res_runs\vtmot_affine_stable_3060\best.pt --output-dir res_runs\vtmot_match_dual_a4000
```

`--init` is a warm start: tensors whose shape still matches are kept and the
rest are reported, so width or architecture knobs can be changed without
throwing the checkpoint away. `--resume` stays strict. `--lr` and
`--batch-size` override the config.

Every validation print includes `ratio` and `match_frac_keys_beating_gt`; judge
progress by the latter (it aggregates ~4800 keys per query over every eval
image, whereas a 16-image `ratio` is noisy).

For a Windows run where worker processes cannot be created, append
`--num-workers 0`; otherwise retain the configured two workers. Check the
printed peak memory after the one-step run before starting the full run.

To continue an interrupted run, use `--resume` with `last.pt`, the same config
and output directory, and a larger **total** `--steps` value. The checkpoint
retains the best validation ratio, so restarting does not replace `best.pt`
with a worse model. For example, after completing 3000 steps:

```powershell
python -B -m res.train_vtmot --device cuda --run full --steps 6000 --num-workers 0 --resume res_runs\vtmot_match_dual_a4000\last.pt --output-dir res_runs\vtmot_match_dual_a4000
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
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs\vtmot_match_dual_a4000\best.pt --output res_runs\vtmot_match_dual_a4000\eval_stride10.json
```

After choosing settings, replace `--split eval` with `--split test` once for
the final held-out report.

The verdict line is deliberately strict:

```text
FUSION READY   EPE <= 2 px and pck@3px >= 0.90
COARSE ONLY    beats zero flow but far from fusion-ready
NOT READY      relative EPE >= 1
```

## Current status

| metric | value | source |
|---|---|---|
| `relative_epe` (best) | 0.754 at step 2700 | 3000-step full-FOV run |
| `match_frac_keys_beating_gt` | 0.065 | same |
| `match_epe_argmax_px` | ~150 (random is ~200) | same |
| `coarse_error_median_px` / `p90` | 8.8 / 12.0 | 500-step local run |
| `window2_coverage` | 0.996 | corrected 500-step wide-128 checkpoint, 16 eval frames |
| 1/4 radius-4 coverage | 0.986 | same checkpoint, independent GT-centred measurement |

The windowed rows come from a weaker 500-step checkpoint. The 3000-step
checkpoint underlying the reported 0.754/0.065 results is not present in this
checkout, so those values need re-evaluation on the new A4000 run. Earlier
`window2_coverage=1.00` reports were inflated by a denominator bug; do not
compare those historical coverage values to the corrected reports.

So the coarse field is good enough to centre a local search but not good enough
to be the answer: the global 1/8 matcher cannot localise, and the usable field
comes from projecting a diffuse correspondence field onto 6 DoF. At 1/8
resolution even 25 neighbouring cells keep beating the truth, and widening the
window makes it worse -- the fine stage therefore has to search on the 1/4
features with a radius of at least 16 px.

Two further known gaps: the confidence-weighted WLS affine fit is fragile on
individual images (occasional eval spikes with an unchanged ranking), and no
robust/outlier-rejecting fit is implemented.

## Tests

```powershell
python -B -m unittest discover -s res/tests -t .
```
