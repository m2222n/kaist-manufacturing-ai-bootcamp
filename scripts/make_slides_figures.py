#!/usr/bin/env python
"""
1차 중간발표 슬라이드용 시각 자료 생성 (PNG).
① 28부품 IoU/Chamfer 막대그래프 (잘됨/안됨 색 구분)
② 렌더 데이터셋 샘플 모자이크
출력: docs/*.png  (라벨은 영문 — 한글 폰트 없음, 학술발표엔 영문이 깔끔)
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)
DATASET = os.path.expanduser("~/kaist_render/KAIST_dataset_v1")


def short(name):
    return name.replace(".stl", "").replace("_hull", "")


# ---------- ① IoU 막대그래프 ----------
def fig_iou():
    rows = json.load(open(os.path.join(ROOT, "outputs/all_res64/summary.json")))
    rows = [r for r in rows if "iou" in r]
    rows.sort(key=lambda r: r["iou"], reverse=True)
    names = [short(r["part"]) for r in rows]
    ious = [r["iou"] for r in rows]
    mean_iou = np.mean(ious)
    # 색: IoU>=0.7 녹색(잘됨) / 0.45~0.7 주황 / <0.45 빨강(안됨)
    colors = ["#2e7d32" if v >= 0.7 else "#ef6c00" if v >= 0.45 else "#c62828" for v in ious]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(ious)), ious, color=colors)
    ax.axhline(mean_iou, color="#1565c0", ls="--", lw=1.5,
               label=f"mean IoU = {mean_iou:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("IoU (reconstruction vs GT)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Visual Hull Baseline — Per-part IoU (28 parts, res=64)",
                 fontsize=13, fontweight="bold")
    # 범례
    from matplotlib.patches import Patch
    legend = [Patch(color="#2e7d32", label="Good (IoU>=0.7): convex/simple"),
              Patch(color="#ef6c00", label="Mid (0.45-0.7)"),
              Patch(color="#c62828", label="Hard (<0.45): thin/concave/holes")]
    ax.legend(handles=legend + [plt.Line2D([], [], color="#1565c0", ls="--",
              label=f"mean = {mean_iou:.3f}")], fontsize=9, loc="upper right")
    plt.tight_layout()
    out = os.path.join(DOCS, "fig1_iou_per_part.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("[①] saved", out)


# ---------- ① 보조: IoU vs hull_inflation 산점도 (왜 안되는지) ----------
def fig_scatter():
    rows = json.load(open(os.path.join(ROOT, "outputs/all_res64/summary.json")))
    rows = [r for r in rows if "iou" in r]
    iou = np.array([r["iou"] for r in rows])
    infl = np.array([r["hull_inflation"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(infl, iou, s=60, c=iou, cmap="RdYlGn", vmin=0.2, vmax=0.85,
               edgecolors="k", linewidths=0.5)
    # 안되는 부품만 라벨 (좌상단 잘되는 부품은 겹쳐서 지저분 → 생략)
    for r in rows:
        if r["iou"] < 0.45:
            ax.annotate(short(r["part"]), (r["hull_inflation"], r["iou"]),
                        fontsize=8, xytext=(6, 0), textcoords="offset points",
                        va="center")
    ax.set_xlabel("Hull inflation (recon volume / GT volume)", fontsize=11)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title("Why IoU drops: more inflation = lower IoU\n(Visual Hull cannot carve concavities)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(DOCS, "fig1b_iou_vs_inflation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("[①b] saved", out)


# ---------- ② 데이터셋 샘플 모자이크 ----------
def fig_mosaic():
    # 대표 부품 4개 × 각도 4개 = 4x4 격자
    parts = ["02_sol_block_b", "11_sw_block", "17_mks_holder", "main_body"]
    parts = [p for p in parts if os.path.isdir(os.path.join(DATASET, p))]
    cell = 200
    cols = 4
    rows_n = len(parts)
    canvas = Image.new("RGB", (cell * cols, cell * rows_n), "white")
    for ri, p in enumerate(parts):
        pngs = sorted(glob.glob(os.path.join(DATASET, p, "*.png")))
        # 다양한 각도 4개 균등 추출
        idx = np.linspace(0, len(pngs) - 1, cols).astype(int)
        for ci, k in enumerate(idx):
            im = Image.open(pngs[k]).convert("RGB").resize((cell, cell))
            canvas.paste(im, (ci * cell, ri * cell))
    out = os.path.join(DOCS, "fig2_dataset_mosaic.png")
    canvas.save(out)
    print("[②] saved", out, f"({rows_n} parts x {cols} angles)")


if __name__ == "__main__":
    fig_iou()
    fig_scatter()
    fig_mosaic()
    print("done")
