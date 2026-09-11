# -*- coding: utf-8 -*-
"""
make_misaligned.py -- 在已对齐的红外/可见光视频上合成「时变错位」，生成配准训练/测试数据

背景：
  实测 E:\vtmot 的 90 个序列，IR<->VIS 全局错位中位数只有 4.5 px，
  且逐帧标准差 0.78 px —— 因为相机刚性固定，相对错位基本是静态的。
  也就是说：现有数据里「没有时序抖动可修」，直接拿它做时序配准模块实验是比不出差异的。
  因此需要合成「随时间平滑演变」的错位，并保存逐帧真值。

关键设计：错位是「有记忆」的，不是每帧独立随机
  dx(t) = amp * 平滑随机信号(t) + jitter * 白噪声(t)
  其中 平滑随机信号 由白噪声做高斯卷积得到，相关长度 corr 控制它变化多慢。
    - corr 大 -> 缓慢漂移（drift），逐帧独立配准会有误差累积
    - jitter 大 -> 逐帧高频抖动，逐帧独立配准会闪
  两者可分别调节，正好用来做消融实验：
    「有 drift / 有 jitter / 都没有」三档数据，比较你的时序方法与逐帧基线。

约定（重要）：
  vis_misaligned(t) = warpPerspective(vis_aligned(t), H_mis(t))
  H_mis  : 把「对齐的 VIS」变换到「错位后的 VIS」，即制造错位用的矩阵
  H_align: H_mis 的逆，是把「错位 VIS」拉回 IR 坐标系所需的矩阵 —— 这才是配准网络要预测的目标
  两个矩阵都会保存，避免歧义。

用法：
  set PY=E:\Aconda\anaconda3\envs\vfbench\python.exe
  %PY% tools\make_misaligned.py --data_root E:\vtmot --seq photo-0304-02 ^
        --out_root E:\vtmot_misaligned --amp 12 --corr 25 --jitter 0.5
  %PY% tools\make_misaligned.py --data_root E:\vtmot --all --max_frames 200 ^
        --out_root E:\vtmot_misaligned --amp 12 --corr 25 --jitter 0.5

输出（每个序列一个目录）：
  <out_root>/<seq>/infrared/         原始 IR（参考，不改）
  <out_root>/<seq>/visible_mis/      错位后的 VIS（网络输入）
  <out_root>/<seq>/visible_gt/       原始对齐 VIS（仅供对照，不要喂给网络）
  <out_root>/<seq>/gt_h/             H_align(t) 3x3 矩阵 (.npy)
  <out_root>/<seq>/gt_params.csv     逐帧 dx, dy, rot, scale, persp
  <out_root>/<seq>/meta.json         生成参数（可复现）
  <out_root>/<seq>/preview.png       可视化检查
"""
import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp_probe as wp

IR_DIR_NAMES = ["infrared", "ir", "IR"]
VIS_DIR_NAMES = ["visible", "rgb", "VIS", "vis"]
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """Unicode 安全的读图（cv2.imread 在中文路径下会失败）。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def imwrite_u(path, img, params=None):
    """Unicode 安全的写图（cv2.imwrite 在中文路径下会静默失败）。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True


def find_modality_dir(seq_dir, candidates):
    for name in candidates:
        p = os.path.join(seq_dir, name)
        if os.path.isdir(p):
            return p
    return None


def list_frames(d):
    out = []
    for e in IMG_EXTS:
        out += glob.glob(os.path.join(d, e))
        out += glob.glob(os.path.join(d, e.upper()))
    return sorted(set(out))


def smooth_noise(T, corr, rng):
    """白噪声 -> 高斯平滑 -> 归一化到 [-1,1]。corr 越大变化越慢。"""
    n = rng.randn(T)
    if T < 2:
        return np.zeros(T)
    if corr > 1.0:
        n = gaussian_filter1d(n, sigma=corr / 2.0, mode="reflect")
    m = float(np.abs(n).max())
    if m > 1e-9:
        n = n / m
    return n


def gen_trajectory(T, amp, corr, jitter, rot_amp, scale_amp, persp_amp, seed):
    """生成逐帧相似变换(+可选透视)参数序列。"""
    rng = np.random.RandomState(seed)
    dx = amp * smooth_noise(T, corr, rng) + jitter * rng.randn(T)
    dy = amp * smooth_noise(T, corr, rng) + jitter * rng.randn(T)
    rot = rot_amp * smooth_noise(T, corr, rng) + 0.15 * jitter * rng.randn(T)
    scale = 1.0 + scale_amp * smooth_noise(T, corr, rng)
    ph = persp_amp * smooth_noise(T, corr, rng)
    pv = persp_amp * smooth_noise(T, corr, rng)
    return dx, dy, rot, scale, ph, pv


def build_H(dx, dy, rot_deg, scale, ph, pv, w, h):
    """在图像中心做旋转/缩放的相似变换，叠加平移与轻微透视。"""
    cx, cy = w / 2.0, h / 2.0
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    a = scale * c
    b = scale * s
    # 先绕中心旋转缩放，再加平移
    tx = dx + cx - (a * cx - b * cy)
    ty = dy + cy - (b * cx + a * cy)
    H = np.array([[a, -b, tx],
                  [b,  a, ty],
                  [ph, pv, 1.0]], dtype=np.float64)
    return H / H[2, 2]

def link_or_copy(src, dst):
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def draw_preview(preview_frames, dx, dy, rot, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(preview_frames)
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, max(n, 1))

    for i, (ir, vis_mis, vis_gt, idx) in enumerate(preview_frames):
        h, w = ir.shape
        e_ir = cv2.Canny(cv2.GaussianBlur(ir, (5, 5), 0), 50, 150)
        e_mis = cv2.Canny(cv2.GaussianBlur(vis_mis, (5, 5), 0), 50, 150)
        e_gt = cv2.Canny(cv2.GaussianBlur(vis_gt, (5, 5), 0), 50, 150)
        c1 = np.zeros((h, w, 3), np.uint8); c1[..., 2] = e_ir; c1[..., 1] = e_mis
        c2 = np.zeros((h, w, 3), np.uint8); c2[..., 2] = e_ir; c2[..., 1] = e_gt
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.imshow(c1); ax.set_title("frame %d\nred=IR green=VIS(misaligned)" % idx, fontsize=8)
        ax = fig.add_subplot(gs[1, i]); ax.axis("off")
        ax.imshow(c2); ax.set_title("frame %d\nred=IR green=VIS(original)" % idx, fontsize=8)

    ax = fig.add_axes([0.08, 0.02, 0.88, 0.001]); ax.axis("off")
    fig.text(0.5, 0.965, title, ha="center", fontsize=11)
    fig.text(0.5, 0.94,
             "green edge should be visibly displaced in the TOP row, aligned in the BOTTOM row",
             ha="center", fontsize=8.5, color="gray")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_gt(dx, dy, rot, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(dx, lw=1.4, label="dx"); axes[0].plot(dy, lw=1.4, label="dy")
    axes[0].set_ylabel("translation (px)"); axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[1].plot(rot, color="tab:orange", lw=1.4); axes[1].set_ylabel("rotation (deg)")
    d1 = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)
    axes[2].plot(np.arange(1, len(dx)), d1, color="tab:red", lw=1.2)
    axes[2].set_ylabel("frame-to-frame |dT| (px)"); axes[2].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def process_sequence(root, seq, args):
    seq_dir = os.path.join(root, seq) if seq else root
    ir_dir = find_modality_dir(seq_dir, IR_DIR_NAMES)
    vis_dir = find_modality_dir(seq_dir, VIS_DIR_NAMES)
    name = seq if seq else os.path.basename(os.path.normpath(seq_dir))
    if ir_dir is None or vis_dir is None:
        print("[skip] %s : no infrared/visible subfolders" % name)
        return None

    pairs = list(zip(list_frames(ir_dir), list_frames(vis_dir)))[: args.max_frames]
    if not pairs:
        print("[skip] %s : empty" % name)
        return None

    T = len(pairs)
    seed = args.seed + (abs(hash(name)) % 100000)

    # ---- 预对齐：原始数据本身可能有几像素残余错位（实测中位数 4.5px），
    #      先把 VIS 对齐到 IR，否则合成的真值会带上这个系统误差 ----
    rdx = rdy = 0.0
    if args.prealign:
        step = max(1, T // 5)
        est = []
        for i in range(0, T, step)[:5]:
            a = imread_u(pairs[i][0], cv2.IMREAD_GRAYSCALE)
            b = imread_u(pairs[i][1], cv2.IMREAD_GRAYSCALE)
            if a is None or b is None:
                continue
            if b.shape != a.shape:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            r = wp.estimate_shift(a, b, radius=args.prealign_radius,
                                  metric="gncc", bins=32)
            if r["ok"]:
                est.append((r["dx"], r["dy"]))
        if est:
            rdx = float(np.median([e[0] for e in est]))
            rdy = float(np.median([e[1] for e in est]))
        print("[prealign] %-20s residual offset dx=%+.1f dy=%+.1f (removed)" % (name, rdx, rdy))
    dx, dy, rot, scale, ph, pv = gen_trajectory(
        T, args.amp, args.corr, args.jitter, args.rot_amp,
        args.scale_amp, args.persp_amp, seed)

    out_dir = os.path.join(args.out_root, name)
    d_ir = os.path.join(out_dir, "infrared")
    d_mis = os.path.join(out_dir, "visible_mis")
    d_gt = os.path.join(out_dir, "visible_gt")
    d_h = os.path.join(out_dir, "gt_h")
    for d in (d_ir, d_mis, d_gt, d_h):
        os.makedirs(d, exist_ok=True)

    rows = []
    preview = []
    pstep = max(1, T // 4)
    for i, (p_ir, p_vis) in enumerate(pairs):
        ir = imread_u(p_ir, cv2.IMREAD_GRAYSCALE)
        vis = imread_u(p_vis, cv2.IMREAD_COLOR)
        if ir is None or vis is None:
            continue
        h, w = ir.shape[:2]
        if vis.shape[1] != w or vis.shape[0] != h:
            vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)

        # 先去掉原始残余错位，再叠加合成错位
        T_pre = np.array([[1.0, 0.0, rdx], [0.0, 1.0, rdy], [0.0, 0.0, 1.0]])
        vis_pre = cv2.warpPerspective(vis, T_pre, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

        H_mis = build_H(dx[i], dy[i], rot[i], scale[i], ph[i], pv[i], w, h)
        H_align = np.linalg.inv(H_mis)

        vis_mis = cv2.warpPerspective(vis_pre, H_mis, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

        base = "%06d" % i
        link_or_copy(p_ir, os.path.join(d_ir, base + ".jpg"))
        imwrite_u(os.path.join(d_mis, base + ".png"), vis_mis)
        imwrite_u(os.path.join(d_gt, base + ".png"), vis_pre)
        np.save(os.path.join(d_h, base + ".npy"), H_align)

        rows.append(dict(frame=i, dx=dx[i], dy=dy[i], rot=rot[i], scale=scale[i],
                         ph=ph[i], pv=pv[i],
                         h_align=",".join("%.8f" % v for v in H_align.reshape(-1))))
        if i % pstep == 0 and len(preview) < 4:
            preview.append((ir, cv2.cvtColor(vis_mis, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), i))

    if not rows:
        return None

    with open(os.path.join(out_dir, "gt_params.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["frame", "dx", "dy", "rot", "scale",
                                             "ph", "pv", "h_align"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    meta = dict(seq=name, n_frames=len(rows), amp=args.amp, corr=args.corr,
                jitter=args.jitter, rot_amp=args.rot_amp,
                scale_amp=args.scale_amp, persp_amp=args.persp_amp, seed=seed,
                convention="vis_mis = warpPerspective(vis, H_mis); "
                           "H_align = inv(H_mis) is the target a registration net should predict")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    d1 = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)
    print("[done] %-22s frames=%4d  |drift|max=%.1fpx  jitter(1st diff) med=%.2f p95=%.2f  "
          "range dx=[%.0f,%.0f] dy=[%.0f,%.0f]"
          % (name, len(rows), float(np.abs(dx).max() + np.abs(dy).max()),
             float(np.median(d1)), float(np.percentile(d1, 95)),
             dx.min(), dx.max(), dy.min(), dy.max()))

    if args.preview:
        draw_preview(preview, dx, dy, rot,
                     os.path.join(out_dir, "preview.png"),
                     "%s  amp=%.0f corr=%.0f jitter=%.2f" % (name, args.amp, args.corr, args.jitter))
        plot_gt(dx, dy, rot, os.path.join(out_dir, "gt_curve.png"), name)
    return dict(seq=name, n_frames=len(rows), med_d1=float(np.median(d1)),
                p95_d1=float(np.percentile(d1, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="e.g. E:\\vtmot")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out_root", required=True, help="e.g. E:\\vtmot_misaligned")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--amp", type=float, default=12.0, help="max drift amplitude (px)")
    ap.add_argument("--corr", type=float, default=25.0,
                    help="correlation length in frames; larger = slower drift")
    ap.add_argument("--jitter", type=float, default=0.5, help="per-frame jitter std (px)")
    ap.add_argument("--rot_amp", type=float, default=2.0, help="max rotation (deg)")
    ap.add_argument("--scale_amp", type=float, default=0.02, help="max scale deviation")
    ap.add_argument("--persp_amp", type=float, default=0.0, help="perspective term amplitude")
    ap.add_argument("--prealign", dest="prealign", action="store_true",
                    help="estimate and remove the residual IR/VIS offset before synthesis")
    ap.add_argument("--no_prealign", dest="prealign", action="store_false")
    ap.add_argument("--prealign_radius", type=int, default=30)
    ap.set_defaults(prealign=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no_preview", dest="preview", action="store_false")
    ap.set_defaults(preview=True)
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        print("data_root not found: %s" % args.data_root)
        return 1

    if args.all:
        seqs = [d for d in sorted(os.listdir(args.data_root))
                if os.path.isdir(os.path.join(args.data_root, d))
                and find_modality_dir(os.path.join(args.data_root, d), IR_DIR_NAMES)]
    elif args.seq:
        seqs = [args.seq]
    else:
        seqs = [None]

    if not seqs:
        print("no sequence found under %s" % args.data_root)
        return 1

    os.makedirs(args.out_root, exist_ok=True)
    print("config: amp=%.1f corr=%.1f jitter=%.2f rot=%.1f scale=%.3f persp=%.4f seed=%d"
          % (args.amp, args.corr, args.jitter, args.rot_amp,
             args.scale_amp, args.persp_amp, args.seed))
    done = 0
    for s in seqs:
        if process_sequence(args.data_root, s, args):
            done += 1
    print("finished: %d sequences -> %s" % (done, os.path.abspath(args.out_root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())