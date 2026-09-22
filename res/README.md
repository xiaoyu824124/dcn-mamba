# Independent MIND to Global Matching registration branch

This folder is deliberately separate from `src/`. It implements the requested
single-frame IR--visible registration research pipeline in validated stages; it
does not yet alter fusion training or the existing video registration code.

## Step 1 — complete: MIND descriptor

`mind.py` contains:

- `MINDDescriptor`: `[B,1,H,W] -> [B,8,H,W]`, patch self-similarity descriptors;
- `rgb_to_gray`: `[B,3,H,W] -> [B,1,H,W]` visible luminance conversion;
- `paired_mind`: paired IR/RGB convenience wrapper.

MIND descriptors retain the input spatial grid, are L2-normalised per pixel,
use no NumPy in `forward`, and support batched CPU/CUDA execution.

Run the first unit test from the repository root:

```powershell
python -B -m unittest res.tests.test_mind -v
```

Expected result: CPU shape/normalisation/autograd tests pass; the CUDA test
also passes when a CUDA device is available.

## Step 2 — complete: shared multi-scale encoder

`encoder.py` contains `MINDFeatureEncoder`. The same encoder instance handles
IR and visible MIND descriptors, returning L2-normalised structural features at
three scales:

```text
[B, 8, H, W] -> 1/2 [B, 24, H/2, W/2]
              -> 1/4 [B, 48, H/4, W/4]
              -> 1/8 [B, 64, H/8, W/8]
```

Run its unit test:

```powershell
python -B -m unittest res.tests.test_encoder -v
```

## Step 3 — complete: 1/8 global matcher

`global_matcher.py` computes a true all-pairs cosine correlation on the 1/8
feature grid. Visible features are queries and infrared features are keys; its
soft-argmax returns tentative correspondences. By default they are projected
through differentiable confidence-weighted least squares into one six-DoF
affine field, matching VTMOT's affine ground truth. The module also retains
the unprojected flow for diagnostics, full correlation, probability matrix and
maximum-probability confidence map.

```powershell
python -B -m unittest res.tests.test_global_matcher -v
```

The translation test uses unique synthetic features and exactly recovers a
feature-grid shift of `[dy, dx]=[5,-4]`, equivalent to `[40,-32]` pixels at
the input grid.

## Step 4 — complete: pixel warp convention

`warp.py` establishes the only flow convention used in this branch:

```text
flow = [dy, dx] in pixels
aligned(y, x) = moving(y + dy, x + dx)
```

It also provides `resize_flow`, which scales displacement magnitudes by
`(new_size - 1) / (old_size - 1)` for the fixed `align_corners=True` convention.

```powershell
python -B -m unittest res.tests.test_warp -v
```

## Step 5 — complete: coarse spatial registration chain

`registration_net.py` provides `MINDGlobalRegistration`:

```text
IR [B,1,H,W] + VI [B,3,H,W]
  -> MIND -> shared encoder -> 1/8 all-pairs matching
  -> coarse [dy,dx] flow -> coarse aligned IR [B,1,H,W]
```

The feature-grid flow is explicitly converted to image-pixel units using the
known encoder stride `(8,8)` before it enters `grid_sample`.  Its output retains
MIND maps, each pyramid feature, full correlation/probability, confidence,
1/8 flow, image flow and coarse aligned IR for later losses/visualisation.

```powershell
python -B -m unittest res.tests.test_registration_net -v
```

## Step 6 — complete: multi-scale DCN local refinement

`dcn_refiner.py` implements `MultiScaleDCNRefiner` at 1/8, 1/4 and 1/2.  Each
level first warps its IR feature with the coarse field, then predicts DCN's 3x3
sampling offsets and masks from warped IR, visible feature, confidence and the
coarser refined feature. DCN offsets remain feature-sampling offsets; they are
never mislabelled as optical flow.

The default mode reconstructs a refined IR image from DCN features and has no
final dense flow.  An optional, separate zero-initialised `ResidualFlowHead`
provides `[dy,dx]` residual and final flow for an explicit-flow ablation.

```powershell
python -B -m unittest res.tests.test_dcn_refiner -v
```

## Step 7 — complete: registration losses

`losses.py` provides `RegistrationLoss` with YAML-controlled weights for:

- robust Charbonnier supervised flow loss when `gt_flow` exists;
- MIND descriptor alignment between refined IR and grayscale visible image;
- contrast-reversal-invariant gradient-magnitude alignment;
- visible-edge-aware smoothness of the explicit coarse/final flow.

```powershell
python -B -m unittest res.tests.test_losses -v
```

## Step 8 — complete: synthetic geometry smoke training

`synthetic.py` produces deterministic cross-modal pairs with a known affine
fixed-to-moving mapping. The tensors retain the branch's sole convention:

```text
warp(ir, gt_flow) = fixed_ir          (on valid in-bounds pixels)
gt_flow = [dy, dx] on the visible/fixed grid
```

The visible frame is a nonlinear three-channel rendering of the latent thermal
structure, so it shares geometry without becoming a raw RGB copy.  It is only
for checking the pipeline and large-displacement global matcher; it is not a
claim of VTMOT performance.

`metrics.py` supplies masked endpoint error and `visualize.py` writes compact
four-panel previews.  `train_registration.py` is a standalone training command
that saves `last.pt`, `metrics.jsonl`, a config snapshot and previews without
touching the original fusion trainer.

Quick CUDA smoke run (about 20 updates):

```powershell
python -B -m res.train_registration --device cuda --steps 20 --no-use-dcn --output-dir res_runs\smoke
```

Full synthetic preflight:

```powershell
python -B -m res.train_registration --device cuda --steps 1000 --output-dir res_runs\synthetic
```

The default includes DCN reconstruction refinement. `--no-use-dcn` isolates
MIND + shared encoder + global matching and is the recommended first check.

Validate the synthetic geometry before training:

```powershell
python -B -m unittest -v res.tests.test_synthetic
```

## Next stage

## Step 9 — complete: read-only VTMOT single-frame validation

`vtmot.py` reads only `infrared`, `visible_mis` and `gt_h` for one frame.  It
recreates the legacy loader's centre-crop/rescale coordinate transform before
turning the stored homography into `[dy,dx]` flow; VTMOT itself is never
modified. `evaluate_vtmot.py` reports masked flow EPE, zero-flow EPE, relative
EPE and 1/3/5-pixel thresholds. Use the `eval` split while choosing settings;
the `test` split is reserved for the final report.

First prove the GT direction using aligned `visible_gt` (no learned model):

```powershell
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --check-gt
```

Then evaluate the synthetic checkpoint as a transfer baseline:

```powershell
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --no-use-dcn --checkpoint res_runs\synthetic_global_3060\last.pt --output res_runs\synthetic_global_3060\vtmot_eval.json
```

`relative_epe < 1` is the hard gate for fusion integration. A synthetic-only
checkpoint is expected to be a weak real-data baseline; if it fails this gate,
fine-tune only the standalone registration branch on VTMOT before touching
fusion.

Always state the field of view. The following reproduces the 160x160 pilot
validation rather than silently comparing it to full-frame EPE:

```powershell
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 50 --crop-hw 160 160 --no-use-dcn --checkpoint res_runs\vtmot_affine_stable_3060\best.pt
```

## Step 10 — real VTMOT coarse-registration fine-tuning

`train_vtmot.py` uses the `train` split, with a shared random 160x160 crop
after VTMOT's aspect-preserving 480x640 normalisation. Flow values remain in
physical pixels and its in-crop valid mask is recomputed after every crop.
The `eval` split is centre-cropped and checked every 100 updates; the held-out
`test` split is still untouched.

Start with a fresh real-data pilot. Do not initialise from the synthetic run
unless deliberately testing transfer as an ablation:

```powershell
python -B -m res.train_vtmot --device cuda --run pilot --output-dir res_runs\vtmot_global_3060
```

After the eval relative EPE is reliably below 1, continue the same run:

```powershell
python -B -m res.train_vtmot --device cuda --steps 3000 --resume res_runs\vtmot_affine_stable_3060\best.pt --output-dir res_runs\vtmot_affine_stable_3060
```

For an A4000 full-frame run, make both training and evaluation field of view
explicit and use batch one:

```powershell
python -B -m res.train_vtmot --device cuda --run pilot --crop-hw 480 640 --batch-size 1 --output-dir res_runs\vtmot_affine_a4000
```
