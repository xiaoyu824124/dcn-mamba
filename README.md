# VTMOT IR–Visible Registration and Fusion

This repository trains a video infrared-visible fusion model with supervised
dense registration on VTMOT. The dataset lives inside the repository at:

```text
data/VTMOT_misaligned/
```

The training configuration uses nine-frame clips, the VTMOT `gt_flow` target,
and a bounded trusted motion memory for long-term cross-modal propagation.

## Train

Activate the environment, then run a short validation job first:

```powershell
  conda activate vfbench-a4000
  python train.py --run pilot --no_wandb
```

The configuration is fixed for one RTX A4000 (16 GB): 288-pixel crops, batch
size one and effective batch size eight. When the pilot loss is stable, run the
full schedule:

```powershell
python train.py --run full --no_wandb
```

Print the resolved configuration without writing files:

```powershell
python train.py --run pilot --dry-run
```

## Test

Check the VTMOT ground-truth geometry:

```powershell
python test.py --check-data
```

Evaluate a completed run's checkpoint on the held-out VTMOT test split:

```powershell
python test.py --exp-path <run-folder-name> --checkpoint best --crop 288
```
