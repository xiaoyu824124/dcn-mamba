# -*- coding: utf-8 -*-
"""
train_registration.py -- 训练跨模态配准模块（MPDRNet 改造版）

数据：data\VTMOT_misaligned\<seq>\
        infrared/     移动图候选（IR）
        visible_mis/  错位后的可见光
        visible_gt/   对齐参考
        gt_h/         H_align(t) 3x3 真值
任务：把 visible_mis 对齐到 infrared（输出 aligned-vis）
监督：gt_h 转成稠密 flow，做 EPE 损失 + 平滑正则

用法：
  set PY=E:\Aconda\anaconda3\envs\vfbench\python.exe
  %PY% tools\train_registration.py --data_root data\VTMOT_misaligned --epochs 20 --batch 4
  %PY% tools\train_registration.py --data_root ... --device cuda --amp
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.registration import CrossModalRegistration, epe_loss, smooth_loss


# ----------------------------------------------------------------------
def homography_to_flow(H_align, H, W):
    """flow(x) = H_align^{-1} x - x ，满足 out(x)=src(x+flow(x))"""
    Hm = np.linalg.inv(H_align)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    P = np.stack([xs, ys, np.ones_like(xs)], 0).reshape(3, -1)
    Q = Hm @ P
    Q = Q[:2] / np.maximum(Q[2:3], 1e-9)
    return (Q.reshape(2, H, W) - np.stack([xs, ys], 0)).astype(np.float32)


def imread_u(path, flags=1):
    import cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flags) if buf.size else None
    except Exception:
        return None


class MisalignedRegDataset(Dataset):
    """读 make_misaligned.py 生成的数据。"""

    def __init__(self, root, seqs=None, crop=256, max_frames=None, augment=True):
        import cv2
        self.cv2 = cv2
        self.items = []
        self.crop = crop
        self.augment = augment
        all_seqs = sorted([d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d))
                           and os.path.isdir(os.path.join(root, d, "gt_h"))])
        if seqs:
            all_seqs = [s for s in all_seqs if s in seqs]
        for s in all_seqs:
            d = os.path.join(root, s)
            hs = sorted(glob.glob(os.path.join(d, "gt_h", "*.npy")))
            im = sorted(glob.glob(os.path.join(d, "infrared", "*.*")))
            vm = sorted(glob.glob(os.path.join(d, "visible_mis", "*.png")))
            n = min(len(hs), len(im), len(vm))
            if max_frames:
                n = min(n, max_frames)
            for i in range(n):
                self.items.append((s, i, im[i], vm[i], hs[i]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, i, pi, pv, ph = self.items[idx]
        ir = imread_u(pi, self.cv2.IMREAD_GRAYSCALE)
        vi = imread_u(pv, self.cv2.IMREAD_GRAYSCALE)
        H = np.load(ph)
        h, w = ir.shape
        if vi.shape != ir.shape:
            vi = self.cv2.resize(vi, (w, h), interpolation=self.cv2.INTER_AREA)
        flow = homography_to_flow(H, h, w)

        # 随机裁剪
        c = self.crop
        if c and (h > c or w > c):
            ch, cw = min(c, h), min(c, w)
            if self.augment:
                y0 = np.random.randint(0, h - ch + 1)
                x0 = np.random.randint(0, w - cw + 1)
            else:
                y0 = (h - ch) // 2
                x0 = (w - cw) // 2
            ir = ir[y0:y0+ch, x0:x0+cw]
            vi = vi[y0:y0+ch, x0:x0+cw]
            flow = flow[:, y0:y0+ch, x0:x0+cw]

        # 尺寸对齐到 16 的倍数（模型内部下采样 4 次）
        H2 = (ir.shape[0] // 16) * 16
        W2 = (ir.shape[1] // 16) * 16
        if H2 < 16 or W2 < 16:
            H2, W2 = 16, 16
        ir = ir[:H2, :W2]; vi = vi[:H2, :W2]; flow = flow[:, :H2, :W2]

        ir = torch.from_numpy(ir.astype(np.float32) / 255.).unsqueeze(0)
        vi = torch.from_numpy(vi.astype(np.float32) / 255.).unsqueeze(0)
        fl = torch.from_numpy(flow)

        # 边界 mask：warp 后超出范围的区域不计损失
        mask = torch.ones(1, H2, W2)
        b = 2
        mask[:, :b, :] = 0; mask[:, -b:, :] = 0
        mask[:, :, :b] = 0; mask[:, :, -b:] = 0

        if self.augment and np.random.rand() < 0.5:
            ir = torch.flip(ir, dims=[2]); vi = torch.flip(vi, dims=[2])
            fl = torch.flip(fl, dims=[1]).clone()
            fl[1] = -fl[1]
            mask = torch.flip(mask, dims=[1])

        return dict(mov=vi, fix=ir, flow=fl, mask=mask, seq=s)


# ----------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    tot_epe, tot_mag = 0.0, 0.0
    n = 0
    with torch.no_grad():
        for b in loader:
            mov = b["mov"].to(device); fix = b["fix"].to(device)
            gt = b["flow"].to(device)
            _, flow = model(mov, fix)
            # 与训练损失口径一致：端点距离 = |dx| + |dy|，逐像素平均
            e = (flow - gt).abs().sum(dim=1).mean().item()
            tot_epe += e * mov.shape[0]
            m = gt.abs().sum(dim=1).mean().item()
            tot_mag += m * mov.shape[0]
            n += mov.shape[0]
    model.train()
    return tot_epe / max(n, 1), tot_mag / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="output/registration")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--lambda_smooth", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    mf = args.max_frames if args.max_frames > 0 else None
    all_seqs = sorted([d for d in os.listdir(args.data_root)
                       if os.path.isdir(os.path.join(args.data_root, d))
                       and os.path.isdir(os.path.join(args.data_root, d, "gt_h"))])
    n_val = max(1, int(len(all_seqs) * args.val_ratio))
    val_seqs = all_seqs[-n_val:]
    tr_seqs = all_seqs[:-n_val]
    print("序列: 训练 %d / 验证 %d" % (len(tr_seqs), len(val_seqs)))

    tr = MisalignedRegDataset(args.data_root, tr_seqs, args.crop, mf, augment=True)
    va = MisalignedRegDataset(args.data_root, val_seqs, args.crop, mf, augment=False)
    print("样本: 训练 %d / 验证 %d" % (len(tr), len(va)))

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=args.workers)

    dev = torch.device(args.device)
    model = CrossModalRegistration(channels=args.channels).to(dev)
    print("参数量: %.3f M   设备: %s" % (sum(p.numel() for p in model.parameters())/1e6, dev))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and dev.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        run = 0.0
        for it, b in enumerate(tl):
            mov = b["mov"].to(dev); fix = b["fix"].to(dev)
            gt = b["flow"].to(dev); msk = b["mask"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and dev.type == "cuda"):
                _, flow = model(mov, fix)
                l_epe = epe_loss(flow, gt, msk)
                l_sm = smooth_loss(flow)
                loss = l_epe + args.lambda_smooth * l_sm
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            run += loss.item()
            if (it + 1) % 20 == 0:
                print("  ep%-3d it%-4d  loss=%.4f  epe=%.4f smooth=%.4f"
                      % (ep, it + 1, loss.item(), l_epe.item(), l_sm.item()))
        sched.step()
        epe_v, mag_v = evaluate(model, vl, dev)
        print("[epoch %d] train_loss=%.4f  val_EPE=%.4f  (真值平均位移=%.2f px)  %.0fs"
              % (ep, run / max(len(tl), 1), epe_v, mag_v, time.time() - t0))
        if epe_v < best:
            best = epe_v
            torch.save(dict(model=model.state_dict(), args=vars(args), epoch=ep),
                       os.path.join(args.out_dir, "best.pth"))
            print("   -> saved best (EPE=%.4f)" % best)
    print("完成。最优 val EPE = %.4f" % best)


if __name__ == "__main__":
    main()