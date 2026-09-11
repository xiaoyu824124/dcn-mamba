# -*- coding: utf-8 -*-
"""
warp_probe.py -- 量化红外与可见光之间的空间错位，以及逐帧错位的抖动程度

目的：在动手做配准模块之前，先用最便宜的办法回答两个问题
   1) IR 与 VIS 之间错位多少像素？是固定偏移还是随深度变化的视差？
   2) 逐帧错位抖不抖、抖多少？

方法选择（在 E:\\vtmot\\photo-0304-02 上的实测对比）：
   gncc  (默认) 梯度幅值图上的归一化互相关 —— 最稳，推荐
   nmi           原始灰度互信息 —— 方差较大，可能出假峰
   ecc           gncc 初值 + ECC 精修（可拿旋转/缩放）
   xcorr         边缘图互相关 —— 不可靠（分数仅 0.10~0.21）
   sift          SIFT+RANSAC —— 跨模态基本失效（每帧仅 2~21 个匹配）

实测结论：本数据集(标定双目)IR/VIS 的全局错位只有几个像素，
          之前 SIFT 报出的几百像素全部是假匹配造成的。

用法：
  set PY=E:\Aconda\anaconda3\envs\vfbench\python.exe
  %PY% tools\warp_probe.py --data_root E:\vtmot --seq photo-0304-02 --max_frames 40
  %PY% tools\warp_probe.py --data_root E:\vtmot --all --max_frames 25

输出（默认 output/warp_probe/）：
  <seq>_warp.csv     逐帧错位与置信度
  <seq>_warp.png     错位曲线 + 一阶/二阶差分（抖动）
  <seq>_samples.png  对齐样例：IR 边缘(红) vs 对齐后 VIS 边缘(绿)
  summary.csv        所有序列汇总
"""
import argparse
import csv
import glob
import math
import os
import sys

import cv2
import numpy as np

IR_DIR_NAMES = ["infrared", "ir", "IR"]
VIS_DIR_NAMES = ["visible", "rgb", "VIS", "vis", "visible_mis"]
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


def load_gray(path, size=None):
    img = imread_u(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if size is not None and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def edge_map(img):
    return cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150)


def gmag(img, blur=7):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    m = cv2.GaussianBlur(m, (blur, blur), 0)
    return m


def blank_result():
    return dict(ok=0, method="", dx=float("nan"), dy=float("nan"),
                score=float("nan"), base=float("nan"), gain=float("nan"),
                rot=float("nan"), scale=float("nan"),
                n_kp_ir=0, n_kp_vis=0, n_matches=0, n_inliers=0,
                rmse=float("nan"), M=None)


def _overlap(a, b, dx, dy, min_side=24):
    """a(y,x) ~= b(y-dy, x-dx) 的重叠区。"""
    h, w = a.shape
    x0 = max(0, dx); x1 = min(w, w + dx)
    y0 = max(0, dy); y1 = min(h, h + dy)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None, None
    return a[y0:y1, x0:x1], b[y0 - dy:y1 - dy, x0 - dx:x1 - dx]


def _down(img, ds):
    if ds <= 1:
        return img.astype(np.float32)
    return cv2.resize(img, None, fx=1.0 / ds, fy=1.0 / ds,
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def prep(img, metric, ds, blur=7):
    """按度量方式准备图像。gncc 先取梯度幅值并标准化。"""
    if metric == "nmi":
        return _down(img, ds)
    g = _down(gmag(img, blur), ds)
    g = g - float(g.mean())
    s = float(g.std())
    return g / (s + 1e-6)


def nmi_score(a, b, bins=32):
    a = a.ravel(); b = b.ravel()
    if a.size < 64:
        return 0.0
    la, ha = np.percentile(a, [1, 99]); lb, hb = np.percentile(b, [1, 99])
    if ha <= la or hb <= lb:
        return 0.0
    a = np.clip((a - la) / (ha - la), 0, 1)
    b = np.clip((b - lb) / (hb - lb), 0, 1)
    c = np.histogram2d(a, b, bins=bins, range=[[0, 1], [0, 1]])[0]
    tot = c.sum()
    if tot <= 0:
        return 0.0
    p = c / tot
    pa = p.sum(axis=1); pb = p.sum(axis=0)
    nz = p > 0
    den = np.outer(pa, pb)
    mi = float(np.sum(p[nz] * np.log(p[nz] / den[nz])))
    pa = pa[pa > 0]; pb = pb[pb > 0]
    ha_ = float(-np.sum(pa * np.log(pa))); hb_ = float(-np.sum(pb * np.log(pb)))
    if ha_ <= 1e-8 or hb_ <= 1e-8:
        return 0.0
    return float(2.0 * mi / (ha_ + hb_))


def score_pair(a, b, metric, bins=32):
    if metric == "nmi":
        return nmi_score(a, b, bins)
    return float((a * b).mean())


def search_shift(ir, vis, center=(0, 0), radius=40, step=4, ds=4,
                 metric="gncc", bins=32):
    a = prep(ir, metric, ds)
    b = prep(vis, metric, ds)
    r = int(round(radius / float(ds)))
    st = max(1, int(round(step / float(ds))))
    cx = int(round(center[0] / float(ds)))
    cy = int(round(center[1] / float(ds)))
    best = (-9.0, 0, 0)
    for dy in range(cy - r, cy + r + 1, st):
        for dx in range(cx - r, cx + r + 1, st):
            aa, bb = _overlap(a, b, dx, dy)
            if aa is None:
                continue
            s = score_pair(aa, bb, metric, bins)
            if s > best[0]:
                best = (s, dx, dy)
    return best[0], best[1] * ds, best[2] * ds


def estimate_shift(ir, vis, radius=40, metric="gncc", bins=32, coarse_step=4):
    """由粗到细搜索全局平移，并返回 (0,0) 处的基准分数用于判断增益。"""
    r = blank_result()
    r["method"] = metric
    # 注意：不同尺度的分数不可直接比较（下采样会平滑图像、抬高分数），
    # 所以必须逐级精修，最终只采用最细尺度的结果。
    s1, dx1, dy1 = search_shift(ir, vis, (0, 0), radius, coarse_step, 4, metric, bins)
    s2, dx2, dy2 = search_shift(ir, vis, (dx1, dy1), 6, 2, 2, metric, bins)
    s3, dx3, dy3 = search_shift(ir, vis, (dx2, dy2), 3, 1, 1, metric, bins)
    s4, dx4, dy4 = search_shift(ir, vis, (dx3, dy3), 1, 1, 1, metric, bins)
    score, dx, dy = s4, dx4, dy4
    a0 = prep(ir, metric, 1); b0 = prep(vis, metric, 1)
    aa, bb = _overlap(a0, b0, 0, 0)
    base = score_pair(aa, bb, metric, bins) if aa is not None else float("nan")
    r.update(dx=float(dx), dy=float(dy), score=float(score),
             base=float(base), gain=float(score - base), ok=1,
             M=np.array([[1., 0., float(dx)], [0., 1., float(dy)]], np.float32))
    return r

def refine_ecc(ir, vis, init_M, motion="euclidean", iters=60, eps=1e-5):
    """在梯度幅值图上用 ECC 精修（跨模态比原始灰度稳）。"""
    g_ir = gmag(ir).astype(np.float32)
    g_vis = gmag(vis).astype(np.float32)
    warp = np.array(init_M, np.float32).copy()
    if motion == "euclidean":
        warp[0, 0], warp[0, 1] = 1.0, 0.0
        warp[1, 0], warp[1, 1] = 0.0, 1.0
        mode = cv2.MOTION_EUCLIDEAN
    else:
        mode = cv2.MOTION_AFFINE
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        cc, warp = cv2.findTransformECC(g_ir, g_vis, warp, mode, crit, None, 5)
    except cv2.error:
        return None, None
    return warp, float(cc)


def estimate_xcorr(ir, vis, margin_ratio=0.20, blur=5):
    r = blank_result(); r["method"] = "xcorr"
    e_ir = edge_map(ir).astype(np.float32)
    e_vis = edge_map(vis).astype(np.float32)
    if blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        e_ir = cv2.GaussianBlur(e_ir, (k, k), 0)
        e_vis = cv2.GaussianBlur(e_vis, (k, k), 0)
    h, w = e_ir.shape
    my, mx = int(h * margin_ratio), int(w * margin_ratio)
    py0, py1, px0, px1 = my, h - my, mx, w - mx
    if py1 - py0 < 16 or px1 - px0 < 16:
        return r
    res = cv2.matchTemplate(e_vis, e_ir[py0:py1, px0:px1], cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res)
    dx = float(ml[0] - px0); dy = float(ml[1] - py0)
    r.update(dx=dx, dy=dy, score=float(mv), base=0.0, gain=float(mv), ok=1,
             M=np.array([[1., 0., dx], [0., 1., dy]], np.float32))
    return r


def estimate_sift(ir, vis, ratio=0.75, thr=3.0, min_inliers=25):
    r = blank_result(); r["method"] = "sift"
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(ir, None)
    k2, d2 = sift.detectAndCompute(vis, None)
    r["n_kp_ir"] = 0 if k1 is None else len(k1)
    r["n_kp_vis"] = 0 if k2 is None else len(k2)
    if d1 is None or d2 is None:
        return r
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    r["n_matches"] = len(good)
    if len(good) < 8:
        return r
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thr)
    if H is None or mask is None:
        return r
    mask = mask.ravel().astype(bool)
    r["n_inliers"] = int(mask.sum())
    if r["n_inliers"] < min_inliers:
        return r
    si = src[mask].reshape(-1, 2); di = dst[mask].reshape(-1, 2)
    proj = cv2.perspectiveTransform(si.reshape(-1, 1, 2), H).reshape(-1, 2)
    r["rmse"] = float(np.sqrt(((proj - di) ** 2).sum(axis=1).mean()))
    M, _ = cv2.estimateAffinePartial2D(si, di, method=cv2.RANSAC, ransacReprojThreshold=thr)
    if M is None:
        return r
    r["scale"] = float(math.hypot(M[0, 0], M[1, 0]))
    r["rot"] = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    r["dx"] = float(M[0, 2]); r["dy"] = float(M[1, 2])
    r.update(score=1.0, base=0.0, gain=1.0, ok=1, M=M)
    return r


def draw_samples(samples, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    if n == 0:
        return
    cols = min(4, n); rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i, (ir, vis, M, frame) in enumerate(samples):
        h, w = ir.shape
        e_ir = edge_map(ir); e_vis = edge_map(vis)
        if M is not None:
            e_vis = (cv2.warpPerspective(e_vis, M, (w, h)) if M.shape == (3, 3)
                     else cv2.warpAffine(e_vis, M, (w, h)))
        canvas = np.zeros((h, w, 3), np.uint8)
        canvas[..., 2] = e_ir
        canvas[..., 1] = e_vis
        axes[i].imshow(canvas)
        axes[i].set_title("frame %s  red=IR green=VIS aligned" % frame, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_seq(rows, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if r["ok"]]
    if len(ok) < 3:
        return
    f = np.array([r["frame"] for r in ok], float)
    dx = np.array([r["dx"] for r in ok], float)
    dy = np.array([r["dy"] for r in ok], float)
    sc = np.array([r["score"] for r in ok], float)
    gn = np.array([r["gain"] for r in ok], float)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(f, dx, lw=1.4, label="dx"); axes[0].plot(f, dy, lw=1.4, label="dy")
    axes[0].set_ylabel("misalignment (px)"); axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[1].plot(f, np.sqrt(dx ** 2 + dy ** 2), color="tab:brown", lw=1.4, label="|shift|")
    axes[1].set_ylabel("shift magnitude"); axes[1].legend(loc="upper right")
    d1 = np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)
    d2 = np.sqrt(np.diff(dx, 2) ** 2 + np.diff(dy, 2) ** 2)
    axes[2].plot(f[1:], d1, color="tab:red", lw=1.2, label="1st diff (px/frame)")
    axes[2].plot(f[2:], d2, color="tab:purple", lw=1.2, label="2nd diff (jitter)")
    axes[2].set_ylabel("frame-to-frame"); axes[2].legend(loc="upper right")
    axes[3].plot(f, sc, color="tab:blue", lw=1.2, label="peak score")
    axes[3].plot(f, gn, color="tab:green", lw=1.2, label="gain over (0,0)")
    axes[3].axhline(0.0, color="gray", ls="--", lw=1)
    axes[3].set_ylabel("confidence"); axes[3].set_xlabel("frame index")
    axes[3].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def summarize(rows, trust):
    ok = [r for r in rows if r["ok"] and r["gain"] >= trust]
    n = len(rows)
    s = dict(n_frames=n, n_ok=len(ok), ok_rate=(len(ok) / float(n) if n else 0.0),
             med_dx=float("nan"), med_dy=float("nan"), med_shift=float("nan"),
             std_shift=float("nan"), med_score=float("nan"), med_gain=float("nan"),
             jitter1=float("nan"), jitter2=float("nan"))
    if len(ok) < 3:
        return s
    dx = np.array([r["dx"] for r in ok], float)
    dy = np.array([r["dy"] for r in ok], float)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    s["med_dx"] = float(np.median(dx)); s["med_dy"] = float(np.median(dy))
    s["med_shift"] = float(np.median(mag)); s["std_shift"] = float(np.std(mag))
    s["med_score"] = float(np.median([r["score"] for r in ok]))
    s["med_gain"] = float(np.median([r["gain"] for r in ok]))
    s["jitter1"] = float(np.median(np.sqrt(np.diff(dx) ** 2 + np.diff(dy) ** 2)))
    if len(dx) > 2:
        s["jitter2"] = float(np.median(np.sqrt(np.diff(dx, 2) ** 2 + np.diff(dy, 2) ** 2)))
    return s

SEARCH_METHODS = ("gncc", "nmi", "ecc")
METRIC_OF = {"gncc": "gncc", "nmi": "nmi", "ecc": "gncc"}


def run_sequence(root, seq, args):
    seq_dir = os.path.join(root, seq) if seq else root
    ir_dir = find_modality_dir(seq_dir, IR_DIR_NAMES)
    vis_dir = find_modality_dir(seq_dir, VIS_DIR_NAMES)
    name = seq if seq else os.path.basename(os.path.normpath(seq_dir))
    if ir_dir is None or vis_dir is None:
        print("[skip] %s : no infrared/visible subfolders" % name)
        return None

    pair = list(zip(list_frames(ir_dir), list_frames(vis_dir)))[: args.max_frames]
    if len(pair) == 0:
        print("[skip] %s : empty" % name)
        return None

    metric = METRIC_OF.get(args.method, "gncc")
    searching = args.method in SEARCH_METHODS

    # ---- pass 1: 用若干帧估全局偏移 ----
    gdx = gdy = 0.0
    if searching:
        step = max(1, len(pair) // args.probe_frames)
        est = []
        for i in range(0, len(pair), step)[: args.probe_frames]:
            ir = load_gray(pair[i][0])
            if ir is None:
                continue
            vis = load_gray(pair[i][1], size=(ir.shape[1], ir.shape[0]))
            if vis is None:
                continue
            r = estimate_shift(ir, vis, args.radius, metric, args.bins, args.coarse_step)
            if r["ok"]:
                est.append((r["dx"], r["dy"], r["score"], r["gain"]))
        if est:
            gdx = float(np.median([e[0] for e in est]))
            gdy = float(np.median([e[1] for e in est]))
            print("[global] %-22s dx=%.1f dy=%.1f  (score=%.3f gain=%.3f from %d frames)"
                  % (name, gdx, gdy,
                     float(np.median([e[2] for e in est])),
                     float(np.median([e[3] for e in est])), len(est)))

    # ---- pass 2: 逐帧 ----
    rows = []
    samples = []
    every = max(1, len(pair) // 8)
    for i, (p_ir, p_vis) in enumerate(pair):
        ir = load_gray(p_ir)
        if ir is None:
            continue
        vis = load_gray(p_vis, size=(ir.shape[1], ir.shape[0]))
        if vis is None:
            continue

        if args.method == "sift":
            r = estimate_sift(ir, vis, args.ratio, args.ransac)
        elif args.method == "xcorr":
            r = estimate_xcorr(ir, vis, args.margin)
        else:
            r = estimate_shift(ir, vis, args.radius, metric, args.bins, args.coarse_step)
            if args.local_radius > 0:
                s, dx, dy = search_shift(ir, vis, (gdx, gdy), args.local_radius,
                                         1, 1, metric, args.bins)
                if s > -9.0:
                    a0 = prep(ir, metric, 1); b0 = prep(vis, metric, 1)
                    aa, bb = _overlap(a0, b0, 0, 0)
                    base = score_pair(aa, bb, metric, args.bins) if aa is not None else float("nan")
                    r.update(dx=float(dx), dy=float(dy), score=float(s),
                             base=float(base), gain=float(s - base),
                             M=np.array([[1., 0., float(dx)], [0., 1., float(dy)]], np.float32))
            if args.method == "ecc":
                W, cc = refine_ecc(ir, vis, r["M"], args.ecc_motion)
                if W is not None:
                    r["dx"] = float(W[0, 2]); r["dy"] = float(W[1, 2])
                    r["score"] = float(cc); r["M"] = W
                    if args.ecc_motion == "affine":
                        r["scale"] = float(math.hypot(W[0, 0], W[1, 0]))
                        r["rot"] = float(math.degrees(math.atan2(W[1, 0], W[0, 0])))

        r["frame"] = i
        rows.append(r)
        if r["ok"] and len(samples) < 8 and i % every == 0:
            samples.append((ir, vis, r["M"], i))
        if args.verbose and i % 10 == 0:
            print("    %s frame %3d  dx=%6.1f dy=%6.1f score=%.3f gain=%.3f"
                  % (name, i, r["dx"], r["dy"], r["score"], r["gain"]))

    if len(rows) == 0:
        return None

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "%s_warp.csv" % name),
              "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "ok", "method", "dx", "dy", "score", "base", "gain",
                    "rot_deg", "scale", "n_kp_ir", "n_kp_vis", "n_matches",
                    "n_inliers", "rmse"])
        for r in rows:
            w.writerow([r["frame"], r["ok"], r["method"], r["dx"], r["dy"],
                        r["score"], r["base"], r["gain"], r["rot"], r["scale"],
                        r["n_kp_ir"], r["n_kp_vis"], r["n_matches"],
                        r["n_inliers"], r["rmse"]])

    plot_seq(rows, os.path.join(args.out_dir, "%s_warp.png" % name),
             "%s  (method=%s, frames=%d, global dx=%.0f dy=%.0f)"
             % (name, args.method, len(rows), gdx, gdy))
    draw_samples(samples, os.path.join(args.out_dir, "%s_samples.png" % name))

    s = summarize(rows, args.trust)
    s.update(seq=name, global_dx=gdx, global_dy=gdy)
    print("[done] %-22s frames=%3d  shift dx=%+.1f dy=%+.1f |%.1f|  "
          "score=%.3f gain=%.3f  jitter1=%.2f jitter2=%.2f"
          % (name, s["n_frames"], s["med_dx"], s["med_dy"], s["med_shift"],
             s["med_score"], s["med_gain"], s["jitter1"], s["jitter2"]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="e.g. E:\\vtmot")
    ap.add_argument("--seq", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max_frames", type=int, default=40)
    ap.add_argument("--method", default="gncc",
                    choices=["gncc", "nmi", "ecc", "xcorr", "sift"])
    ap.add_argument("--radius", type=int, default=30, help="global search radius (px)")
    ap.add_argument("--coarse_step", type=int, default=4)
    ap.add_argument("--local_radius", type=int, default=6,
                    help="per-frame local search radius around the global offset")
    ap.add_argument("--probe_frames", type=int, default=10)
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--ecc_motion", default="euclidean", choices=["euclidean", "affine"])
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--trust", type=float, default=0.005,
                    help="min gain over the (0,0) baseline to trust a frame")
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--out_dir", default=os.path.join("output", "warp_probe"))
    ap.add_argument("--verbose", action="store_true")
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

    results = []
    for s in seqs:
        r = run_sequence(args.data_root, s, args)
        if r:
            results.append(r)

    if results:
        os.makedirs(args.out_dir, exist_ok=True)
        keys = ["seq", "n_frames", "n_ok", "ok_rate", "global_dx", "global_dy",
                "med_dx", "med_dy", "med_shift", "std_shift", "med_score",
                "med_gain", "jitter1", "jitter2"]
        with open(os.path.join(args.out_dir, "summary.csv"),
                  "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in keys})

        ms = np.array([r["med_shift"] for r in results], float)
        sd = np.array([r["std_shift"] for r in results], float)
        j1 = np.array([r["jitter1"] for r in results], float)
        j2 = np.array([r["jitter2"] for r in results], float)
        print("")
        print("=" * 72)
        print("OVERALL  (%d sequences, method=%s)" % (len(results), args.method))
        print("  median |shift| IR<->VIS : %.2f px" % np.nanmedian(ms))
        print("  median std(|shift|)     : %.2f px   <- 越大说明逐帧越不稳定" % np.nanmedian(sd))
        print("  median 1st-diff (jitter): %.3f px/frame" % np.nanmedian(j1))
        print("  median 2nd-diff (jitter): %.3f px/frame^2" % np.nanmedian(j2))
        print("  outputs -> %s" % os.path.abspath(args.out_dir))
        print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())