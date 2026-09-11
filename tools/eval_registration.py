# -*- coding: utf-8 -*-
"""
eval_registration.py -- 配准评测脚本

输入：make_misaligned.py 生成的数据集（含 gt_h/ 真值）
用途：给任何配准方法打分，并给出三个内置 baseline 作为参照。

内置 baseline（这三个就能回答"时序模块到底有没有用"）：
  identity  不做配准（直接用错位图）
  static    只做一次标定：用真值的中位数变换，对所有帧用同一个 H
            —— 这正是"刚性双目静态标定"能拿到的最好结果
  oracle    每帧用真值 —— 理论上限

关键判读：
  若 static 已接近 oracle，说明该数据集不需要逐帧/时序配准；
  若 static 明显差于 oracle，说明错位确实随时间变化 —— 逐帧与时序方法才有发挥空间。

指标：
  MEE      平均端点误差 (px)：把采样点用预测 H 与真值 H 分别变换，取平均欧氏距离 —— 主指标
  PSNR     配准后 VIS 与 visible_gt 的峰值信噪比 (dB)
  SSIM     同上，结构相似度
  chamfer  配准后 VIS 与 IR 的边缘距离 (px)，不需要参考图，反映真实对齐质量

用法：
  set PY=E:\Aconda\anaconda3\envs\vfbench\python.exe
  %PY% tools\eval_registration.py --data_root data\VTMOT_misaligned
  :: 评估自己方法的输出（<pred_root>/<seq>/pred_h/*.npy）
  %PY% tools\eval_registration.py --data_root data\VTMOT_misaligned --pred_root output\my_method

输出（默认 <data_root>/../eval_registration/）：
  per_sequence.csv / summary.csv / compare.png
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

IR_DIR_NAMES = ["infrared", "ir", "IR"]
VIS_DIR_NAMES = ["visible_mis", "visible", "rgb", "vis"]


def imread_u(path, flags=cv2.IMREAD_COLOR):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None


def list_frames(d):
    out = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        out += glob.glob(os.path.join(d, e))
    return sorted(set(out))


def load_gt(seq_dir):
    """读 gt_params.csv 里逐帧的 h_align（9 个数），返回 {frame: 3x3}。"""
    p = os.path.join(seq_dir, "gt_params.csv")
    if not os.path.isfile(p):
        return {}
    out = {}
    with open(p, "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            vals = [float(v) for v in row["h_align"].split(",")]
            out[int(row["frame"])] = np.array(vals, np.float64).reshape(3, 3)
    return out


def load_pred(pred_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.npy"))):
        try:
            idx = int(os.path.splitext(os.path.basename(f))[0])
        except ValueError:
            continue
        out[idx] = np.load(f).astype(np.float64).reshape(3, 3)
    return out


def sample_grid(w, h, n=7, margin=0.1):
    xs = np.linspace(w * margin, w * (1 - margin), n)
    ys = np.linspace(h * margin, h * (1 - margin), n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=0)


def apply_h(H, pts):
    q = H @ pts
    q = q[:2] / np.maximum(q[2:3], 1e-9)
    return q.T


def mee(H_pred, H_gt, pts):
    a = apply_h(H_pred, pts)
    b = apply_h(H_gt, pts)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean())


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def ssim(a, b, sigma=1.5):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (0, 0), sigma)
    mu_b = cv2.GaussianBlur(b, (0, 0), sigma)
    ma2, mb2, mab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, (0, 0), sigma) - ma2
    sb = cv2.GaussianBlur(b * b, (0, 0), sigma) - mb2
    sab = cv2.GaussianBlur(a * b, (0, 0), sigma) - mab
    s = ((2 * mab + C1) * (2 * sab + C2)) / ((ma2 + mb2 + C1) * (sa + sb + C2))
    return float(s.mean())


def grad_mag(img, blur=7):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(cv2.magnitude(gx, gy), (blur, blur), 0)


def gncc_score(a, b):
    """梯度幅值上的归一化互相关（越大越好）。

    跨模态下原始灰度不可直接比较，但结构边缘的梯度幅值高度相关，
    这也是本项目 warp_probe 里最稳的度量（实测比原始灰度 MI/NMI 区分度高得多）。
    """
    ga = grad_mag(a); gb = grad_mag(b)
    ga = ga - float(ga.mean()); gb = gb - float(gb.mean())
    sa = float(ga.std()); sb = float(gb.std())
    if sa < 1e-6 or sb < 1e-6:
        return float("nan")
    return float((ga / sa * (gb / sb)).mean())


def median_homography(Hs, w, h, n=7):
    """用采样点位置的中位数拟合一个"代表变换"，比直接对矩阵元素取中位数稳健。"""
    pts = sample_grid(w, h, n)
    allp = np.stack([apply_h(H, pts) for H in Hs], axis=0)   # [N, P, 2]
    med = np.median(allp, axis=0)                            # [P, 2]
    src = pts[:2].T.astype(np.float32).reshape(-1, 1, 2)
    dst = med.astype(np.float32).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src, dst, 0)
    return M if M is not None else np.eye(3)

def build_methods(args, seq_dir, gt, frames, ir_img, w, h):
    """返回 {method_name: {frame: H}}"""
    methods = {}
    eye = np.eye(3)

    if "identity" in args.method_set:
        methods["identity"] = {i: eye for i in frames}

    if "static" in args.method_set:
        calib = frames[:args.calib_frames]
        Hs = [gt[i] for i in calib if i in gt]
        H_med = median_homography(Hs, w, h) if Hs else eye
        methods["static"] = {i: H_med for i in frames}

    if "oracle" in args.method_set:
        methods["oracle"] = {i: gt[i] for i in frames if i in gt}

    if args.pred_root:
        d = os.path.join(args.pred_root, os.path.basename(os.path.normpath(seq_dir)), "pred_h")
        if not os.path.isdir(d):
            d = os.path.join(args.pred_root, os.path.basename(os.path.normpath(seq_dir)))
        if os.path.isdir(d):
            pr = load_pred(d)
            if pr:
                methods[args.pred_name] = pr
    return methods


def eval_sequence(seq_dir, args):
    name = os.path.basename(os.path.normpath(seq_dir))
    ir_dir = None
    for c in IR_DIR_NAMES:
        p = os.path.join(seq_dir, c)
        if os.path.isdir(p):
            ir_dir = p; break
    vis_dir = None
    for c in VIS_DIR_NAMES:
        p = os.path.join(seq_dir, c)
        if os.path.isdir(p):
            vis_dir = p; break
    gt_dir = os.path.join(seq_dir, "visible_gt")
    if ir_dir is None or vis_dir is None:
        return []

    ir_files = list_frames(ir_dir)
    vis_files = list_frames(vis_dir)
    gt_files = list_frames(gt_dir) if os.path.isdir(gt_dir) else []
    gt = load_gt(seq_dir)

    # IR 是硬链接，扩展名可能不同，按序号对齐
    n = min(len(ir_files), len(vis_files))
    if args.max_frames:
        n = min(n, args.max_frames)
    if n == 0:
        return []

    ir0 = imread_u(ir_files[0], cv2.IMREAD_GRAYSCALE)
    if ir0 is None:
        return []
    h, w = ir0.shape
    frames = [i for i in range(n) if i in gt]
    if not frames:
        return []

    methods = build_methods(args, seq_dir, gt, frames, ir0, w, h)
    if not methods:
        return []

    pts = sample_grid(w, h, args.sample_grid)
    rows = []
    for mname, Hmap in methods.items():
        mees, psnrs, ssims, nmis = [], [], [], []
        for i in frames:
            if i not in Hmap:
                continue
            mees.append(mee(Hmap[i], gt[i], pts))
            if i % args.metric_stride != 0:
                continue
            mis = imread_u(vis_files[i], cv2.IMREAD_COLOR)
            if mis is None:
                continue
            if mis.shape[1] != w or mis.shape[0] != h:
                mis = cv2.resize(mis, (w, h), interpolation=cv2.INTER_AREA)
            reg = cv2.warpPerspective(mis, Hmap[i], (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
            b = args.border
            g_reg = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
            if gr and gt_files and i < len(gt_files):
                ref = imread_u(gt_files[i], cv2.IMREAD_COLOR)
                if ref is not None:
                    if ref.shape[1] != w or ref.shape[0] != h:
                        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
                    a_c = g_reg[b:-b, b:-b].astype(np.uint8)
                    c_c = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)[b:-b, b:-b].astype(np.uint8)
                    psnrs.append(psnr(a_c, c_c))
                    ssims.append(ssim(a_c, c_c))
            if args.gt_img is not None and i < len(args.gt_img):
                pass
            irf = imread_u(ir_files[i], cv2.IMREAD_GRAYSCALE)
            if irf is not None:
                if irf.shape[1] != w or irf.shape[0] != h:
                    irf = cv2.resize(irf, (w, h), interpolation=cv2.INTER_AREA)
                nmis.append(gncc_score(g_reg[b:-b, b:-b], irf[b:-b, b:-b]))

        def m_(v):
            return float(np.mean(v)) if len(v) else float("nan")
        def md_(v):
            return float(np.median(v)) if len(v) else float("nan")

        rows.append(dict(seq=name, method=mname, n_frames=len(mees),
                         mee_mean=m_(mees), mee_median=md_(mees),
                         psnr_mean=m_(psnrs), ssim_mean=m_(ssims),
                         gncc_mean=m_(nmis)))
    return rows


def plot_compare(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["method"] for r in summary]
    mee = [r["mee_mean"] for r in summary]
    ps = [r["psnr_mean"] for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, mee, color="#4C72B0")
    axes[0].set_ylabel("MEE (px)  lower is better")
    axes[0].set_title("registration error")
    for i, v in enumerate(mee):
        axes[0].text(i, v, "%.1f" % v, ha="center", va="bottom", fontsize=9)
    axes[1].bar(names, ps, color="#55A868")
    axes[1].set_ylabel("PSNR (dB)  higher is better")
    axes[1].set_title("aligned image quality")
    for i, v in enumerate(ps):
        axes[1].text(i, v, "%.1f" % v, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_root", default=None,
                    help="你的方法输出：<pred_root>/<seq>/pred_h/*.npy")
    ap.add_argument("--pred_name", default="ours")
    ap.add_argument("--methods", default="identity,static,oracle")
    ap.add_argument("--calib_frames", type=int, default=5,
                    help="static baseline 用前几帧的真值做一次标定")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--metric_stride", type=int, default=5,
                    help="图像类指标每几帧算一次（MEE 每帧都算）")
    ap.add_argument("--border", type=int, default=40,
                    help="图像指标裁掉的边界像素（warp 会带来边界伪影）")
    ap.add_argument("--sample_grid", type=int, default=7)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    global gr
    gr = True

    if not os.path.isdir(args.data_root):
        print("data_root not found: %s" % args.data_root)
        return 1
    args.method_set = set(m.strip() for m in args.methods.split(",") if m.strip())
    if not args.method_set and not args.pred_root:
        print("nothing to evaluate")
        return 1
    args.gt_img = None

    seqs = [d for d in sorted(os.listdir(args.data_root))
            if os.path.isdir(os.path.join(args.data_root, d))
            and os.path.isfile(os.path.join(args.data_root, d, "gt_params.csv"))]
    if not seqs:
        print("no sequence with gt_params.csv under %s" % args.data_root)
        return 1

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.data_root)),
                                           "eval_registration")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for s in seqs:
        rows = eval_sequence(os.path.join(args.data_root, s), args)
        if rows:
            all_rows += rows
            print("[eval] %-22s %s" % (s, "  ".join(
                "%s:mee=%.2fpsnr=%.1f" % (r["method"], r["mee_mean"], r["psnr_mean"])
                for r in rows)))

    if not all_rows:
        print("no results")
        return 1

    per = os.path.join(out_dir, "per_sequence.csv")
    keys = ["seq", "method", "n_frames", "mee_mean", "mee_median",
            "psnr_mean", "ssim_mean", "gncc_mean"]
    with open(per, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    by = {}
    for r in all_rows:
        by.setdefault(r["method"], []).append(r)
    summary = []
    for m, rs in by.items():
        summary.append(dict(method=m,
                            n_seq=len(rs),
                            mee_mean=float(np.nanmean([r["mee_mean"] for r in rs])),
                            mee_median=float(np.nanmedian([r["mee_median"] for r in rs])),
                            psnr_mean=float(np.nanmean([r["psnr_mean"] for r in rs])),
                            ssim_mean=float(np.nanmean([r["ssim_mean"] for r in rs])),
                            gncc_mean=float(np.nanmean([r["gncc_mean"] for r in rs]))))
    order = {"identity": 0, "static": 1, "oracle": 2}
    summary.sort(key=lambda r: (order.get(r["method"], 9), r["method"]))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "n_seq", "mee_mean", "mee_median",
                                           "psnr_mean", "ssim_mean", "gncc_mean"])
        w.writeheader()
        for r in summary:
            w.writerow(r)

    plot_compare(summary, os.path.join(out_dir, "compare.png"))

    print("")
    print("=" * 78)
    print("%-10s %8s %8s %8s %8s %9s" % ("method", "MEE(px)", "MEEmed", "PSNR(dB)", "SSIM", "GNCC"))
    print("-" * 78)
    for r in summary:
        print("%-10s %8.2f %8.2f %8.2f %8.4f %9.3f"
              % (r["method"], r["mee_mean"], r["mee_median"],
                 r["psnr_mean"], r["ssim_mean"], r["gncc_mean"]))
    print("=" * 78)
    print("outputs -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())