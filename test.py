# -*- coding: utf-8 -*-
"""Evaluate a VTMOT checkpoint and report registration accuracy.

WHICH SPLIT -- THEY ARE NOT INTERCHANGEABLE
-------------------------------------------
  This script always uses the six-sequence held-out ``test`` split.  It loads a
  checkpoint produced by ``train.py`` and must not be used to tune settings.

EPE is the single shared definition from ``src/util/flow_metric.py``: the mean
per-pixel L2 norm of ``pred - gt``, i.e. the standard end-point error, not a
component-wise L1.  It is reported twice:

  full frame   what the dataset stores
  centre crop  --crop pixels, the same field of view the training crops use

Those two are NOT comparable with each other; a ratio measured on a crop is a
different number from the same ratio on the full frame.

Examples::

    python test.py --check-data
    python test.py --exp-path <run> --checkpoint best --crop 288
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataset import get_ir_visible_dataset
from src.dataset.base_two_modal_dataset import DatasetMode
from src.model.net import IRVisibleFusion
from src.model.registration.common import SpatialTransformer
from src.util.flow_metric import flow_epe

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_CONFIG = "config/dataset/IRVisible/VTMOT/vtmot_5-frame-val.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VTMOT registration")
    parser.add_argument("--base-data-dir", default=str(ROOT / "data"),
                        help="parent directory of VTMOT_misaligned "
                             "(train.py uses the same argument name)")
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--exp-path", type=Path,
                        help="run directory, relative to output/ or absolute")
    parser.add_argument("--checkpoint", default="best",
                        help="checkpoint directory name, e.g. latest or best")
    parser.add_argument("--crop", type=int, default=0,
                        help="also report EPE on a centre crop of this size, to "
                             "match the training field of view (0 = off)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-data", action="store_true",
                        help="validate the stored GT flow without loading a model")
    return parser.parse_args()


def build_loader(args):
    config_path = Path(args.dataset_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = OmegaConf.load(config_path)
    dataset_dir = Path(args.base_data_dir) / "VTMOT_misaligned"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"VTMOT dataset not found at {dataset_dir}. The default is "
            f"<repo>/data/VTMOT_misaligned; otherwise pass --base-data-dir "
            f"pointing at the PARENT of VTMOT_misaligned. Note that csv_dir in "
            f"the dataset config is relative to the working directory, so run "
            f"this from the repository root.")
    dataset = get_ir_visible_dataset(
        cfg, base_data_dir=str(args.base_data_dir),
        mode=DatasetMode.TEST, augmentation_args=None)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0)
    return dataset, loader


def sequence_of(batch, index):
    """Sequence name of sample `index`, taken from the first raster column."""
    paths = batch["data_path_ls_dict"]["ir"][index]
    return Path(paths[0]).parts[0]


@torch.no_grad()
def check_data(loader, device):
    """Verify that the stored GT maps the aligned visible frame onto visible_mis."""
    transformer = SpatialTransformer().to(device)
    errors, baselines = [], []
    for batch_index, batch in enumerate(loader):
        rgb_gt = batch["rgb_gt"].to(device)
        rgb_mis = batch["rgb"].to(device)
        flow = batch["gt_flow"].to(device)
        batch_size, frames, channels, height, width = rgb_gt.shape
        warped = transformer(
            rgb_gt.reshape(-1, channels, height, width),
            flow.reshape(-1, 2, height, width))[0]
        target = rgb_mis.reshape(-1, channels, height, width)
        errors.append((warped - target).abs().mean().item())
        baselines.append((rgb_gt - rgb_mis).abs().mean().item())
        if batch_index >= 9:
            break
    print("GT warp MAE: %.5f (unwarped: %.5f)  -> lower warp MAE means the GT "
          "flow really aligns the pair" % (
              sum(errors) / len(errors), sum(baselines) / len(baselines)))


@torch.no_grad()
def evaluate(model, loader, device, crop=0):
    """Per-sequence and overall registration EPE, using the shared definition."""
    model.eval()
    # per sequence -> lists of per-window (epe, baseline)
    per_seq = OrderedDict()
    for batch in loader:
        infrared = batch["ir"].to(device, non_blocking=True)
        visible = batch["rgb"].to(device, non_blocking=True)
        gt_flow = batch["gt_flow"].to(device, non_blocking=True)
        _, registration = model(infrared, visible, stage="registration")
        predicted = registration["flows"]
        epe, baseline = flow_epe(predicted, gt_flow, crop=crop, per_sample=True)
        for index in range(epe.shape[0]):
            per_seq.setdefault(sequence_of(batch, index), []).append(
                (float(epe[index]), float(baseline[index])))

    print()
    print(f"  {'sequence':22s} {'windows':>7s} {'EPE(px)':>9s} "
          f"{'baseline':>9s} {'ratio':>7s}")
    all_epe, all_base = [], []
    for name, values in per_seq.items():
        epe = sum(v[0] for v in values) / len(values)
        base = sum(v[1] for v in values) / len(values)
        all_epe.append(epe)
        all_base.append(base)
        print(f"  {name:22s} {len(values):7d} {epe:9.4f} {base:9.4f} "
              f"{epe / max(base, 1e-6):7.4f}")
    epe = sum(all_epe) / max(len(all_epe), 1)
    base = sum(all_base) / max(len(all_base), 1)
    print(f"  {'OVERALL':22s} {len(all_epe):7d} {epe:9.4f} {base:9.4f} "
          f"{epe / max(base, 1e-6):7.4f}")
    print()
    print("  ratio < 1 means the predicted flow is better than predicting no "
          "motion at all.")
    return {"sequences": len(all_epe), "epe_px": epe, "baseline_px": base,
            "epe_ratio": epe / max(base, 1e-6)}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset, loader = build_loader(args)

    print("=" * 74)
    print(f"held-out TEST split  |  windows: {len(dataset)}  |  "
          f"sequences: {', '.join(dataset.scene_name_ls)}")
    print("  ** This is the held-out TEST split. Report it once and do not "
          "tune on it. **")
    print("=" * 74)

    if args.check_data:
        check_data(loader, device)
        return
    if args.exp_path is None:
        raise SystemExit("--exp-path is required unless --check-data is used")

    run_dir = (args.exp_path if args.exp_path.is_absolute()
               else ROOT / "output" / args.exp_path)
    checkpoint = run_dir / "checkpoint" / args.checkpoint / "model.pth"
    config = run_dir / "config.yaml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(f"missing checkpoint or config under {run_dir}")
    cfg = OmegaConf.load(config)
    model = IRVisibleFusion(model_config={"model": cfg.model}).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
    print(f"checkpoint: {checkpoint}")

    print("\n--- full frame ---")
    evaluate(model, loader, device, crop=0)
    if args.crop > 0:
        print(f"\n--- centre crop {args.crop} (matches the training FOV) ---")
        evaluate(model, loader, device, crop=args.crop)


if __name__ == "__main__":
    main()
