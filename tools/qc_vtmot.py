# -*- coding: utf-8 -*-
"""
qc_vtmot.py -- 对 VTMOT 全部序列做可配准性质检

背景：实测发现有序列的原数据本身就无法配准（某个模态几乎全黑/无纹理），
      在这样的序列上合成的"错位真值"没有意义。必须先筛掉。

判据（默认值可通过参数调整）：
  1. 两个模态的边缘占比都 >= --min_edge      （有结构可匹配）
  2. 预对齐的 GNCC score      >= --min_score  （确实匹配上了）
  3. 预对齐的 gain            >= --min_gain   （比不对齐有实质提升）
  4. 预对齐后的残余错位       <= --max_resid  （对齐基准干净）
  5. 不同搜索半径结果一致      <= --max_drift  （解稳定，不是碰运气）

输出：<out_csv>，以及控制台汇总。
"""
import argparse, csv, glob, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warp_probe as wp

IR_NAMES = ["infrared", "ir", "IR"]
VIS_NAMES = ["visible", "rgb", "VIS", "vis", "visible_mis"]


def find_dir(seq_dir, cands):
    for c in cands:
        p = os.path.join(seq_dir, c)
        if os.path.isdir(p):
            return p
    return None


def edge_frac(img):
    e = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150)
    return float((e > 0).mean())


def qc_sequence(seq_dir, args):
    ir_dir = find_dir(seq_dir, IR_NAMES)
    vis_dir = find_dir(seq_dir, VIS_NAMES)
    if not ir_dir or not vis_dir:
        return None
    irf = wp.list_frames(ir_dir)
    vif = wp.list_frames(vis_dir)
    n = min(len(irf), len(vif))
    if n < 3:
        return None
    idx = list(range(0, n, max(1, n // args.probe)))[: args.probe]

    ed_ir, ed_vi = [], []
    est_small, est_big, sc, gn = [], [], [], []
    for i in idx:
        a = wp.imread_u(irf[i], cv2.IMREAD_GRAYSCALE)
        b = wp.imread_u(vif[i], cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        ed_ir.append(edge_frac(a)); ed_vi.append(edge_frac(b))

        r1 = wp.estimate_shift(a, b, radius=args.radius_small, metric="gncc", bins=32)
        if r1["ok"]:
            est_small.append((r1["dx"], r1["dy"]))
            sc.append(r1["score"]); gn.append(r1["gain"])
        r2 = wp.estimate_shift(a, b, radius=args.radius_big, metric="gncc", bins=32)
        if r2["ok"]:
            est_big.append((r2["dx"], r2["dy"]))

    if not est_small or not ed_ir:
        return None

    sx = float(np.median([e[0] for e in est_small]))
    sy = float(np.median([e[1] for e in est_small]))
    bx = float(np.median([e[0] for e in est_big])) if est_big else sx
    by = float(np.median([e[1] for e in est_big])) if est_big else sy
    drift = float(np.hypot(sx - bx, sy - by))

    # 预对齐后重测残余
    res = []
    for i in idx:
        a = wp.imread_u(irf[i], cv2.IMREAD_GRAYSCALE)
        b = wp.imread_u(vif[i], cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        T = np.float32([[1, 0, sx], [0, 1, sy]])
        b2 = cv2.warpAffine(b, T, (a.shape[1], a.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        r = wp.estimate_shift(a, b2, radius=args.resid_radius, metric="gncc", bins=32)
        if r["ok"]:
            res.append(np.hypot(r["dx"], r["dy"]))

    o_ir = float(np.median(ed_ir)); o_vi = float(np.median(ed_vi))
    o_sc = float(np.median(sc)); o_gn = float(np.median(gn))
    o_res = float(np.median(res)) if res else float("nan")

    ok = (min(o_ir, o_vi) >= args.min_edge and o_sc >= args.min_score
          and o_gn >= args.min_gain and o_res <= args.max_resid
          and drift <= args.max_drift)
    return dict(seq=os.path.basename(os.path.normpath(seq_dir)), n_frames=n,
                edge_ir=o_ir, edge_vis=o_vi, score=o_sc, gain=o_gn,
                offset_dx=sx, offset_dy=sy, residual=o_res, drift=drift,
                ok=int(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--probe", type=int, default=3)
    ap.add_argument("--radius_small", type=int, default=40)
    ap.add_argument("--radius_big", type=int, default=80)
    ap.add_argument("--resid_radius", type=int, default=12)
    ap.add_argument("--min_edge", type=float, default=0.010)
    ap.add_argument("--min_score", type=float, default=0.15)
    ap.add_argument("--min_gain", type=float, default=0.02)
    ap.add_argument("--max_resid", type=float, default=3.0)
    ap.add_argument("--max_drift", type=float, default=5.0)
    args = ap.parse_args()

    seqs = []
    for d in sorted(os.listdir(args.data_root)):
        p = os.path.join(args.data_root, d)
        if os.path.isdir(p) and find_dir(p, IR_NAMES) and find_dir(p, VIS_NAMES):
            seqs.append(p)

    rows = []
    for i, p in enumerate(seqs):
        r = qc_sequence(p, args)
        if r:
            rows.append(r)
            flag = "OK " if r["ok"] else "BAD"
            print("[%2d/%2d] %s %-20s edge=%.3f/%.3f score=%.3f gain=%.3f resid=%5.1f drift=%5.1f"
                  % (i + 1, len(seqs), flag, r["seq"], r["edge_ir"], r["edge_vis"],
                     r["score"], r["gain"], r["residual"], r["drift"]))

    keys = ["seq", "n_frames", "edge_ir", "edge_vis", "score", "gain",
            "offset_dx", "offset_dy", "residual", "drift", "ok"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    print("")
    print("=" * 70)
    print("质检完成：共 %d 个序列，合格 %d 个，淘汰 %d 个" % (len(rows), len(good), len(bad)))
    print("判据：edges>=%.3f  score>=%.2f  gain>=%.3f  resid<=%.1f  drift<=%.1f"
          % (args.min_edge, args.min_score, args.min_gain, args.max_resid, args.max_drift))
    if bad:
        print("")
        print("淘汰原因分布：")
        for r in bad[:20]:
            why = []
            if min(r["edge_ir"], r["edge_vis"]) < args.min_edge: why.append("边缘不足")
            if r["score"] < args.min_score: why.append("匹配分低")
            if r["gain"] < args.min_gain: why.append("无增益")
            if r["residual"] > args.max_resid: why.append("残余大")
            if r["drift"] > args.max_drift: why.append("解不稳")
            print("   %-20s %s" % (r["seq"], " / ".join(why)))
    print("报告 -> %s" % args.out_csv)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())