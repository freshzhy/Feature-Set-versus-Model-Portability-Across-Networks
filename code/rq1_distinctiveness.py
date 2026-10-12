#!/usr/bin/env python3
"""
分析1：RQ1 攻击类别间 SHAP 签名区分度矩阵
补充分析 — 将 RQ1 从"定性展示"升级为"定量验证"

研究问题：同一数据集内，不同攻击类别的 SHAP 签名之间有多大差异？
          如果多数配对 Spearman ρ < 0.3，则定量证明签名具有攻击特异性。

方法：
  对每个数据集，取出所有攻击类别的 abs_mean_shap 向量，
  计算两两之间的 Spearman ρ，形成类间区分度矩阵（攻击类 × 攻击类）。
  ρ 越低 = 签名越正交 = 特异性越强。

直接复用 RQ1 输出的 *_rq1_shap_values.pkl，无需重跑任何模型。

输出（保存到 --out 目录）：
  {dataset}_distinctiveness_matrix.csv  — Spearman ρ 矩阵
  {dataset}_distinctiveness_matrix.pdf/png — 热力图
  summary.csv                           — 跨数据集汇总统计
  summary.json

用法：
  python code/rq1_distinctiveness.py --rq1-dir ./results/rq1_signatures/ --out ./results/rq1_distinctiveness/
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATASET_SHORT = {
    "NF-UNSW-NB15-v2":       "UNSW",
    "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2":         "ToN",
    "NF-BoT-IoT-v2":         "BoT",
}
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]

# Infilteration 低可信度类别（F1=0.44），在论文结果中标注
LOW_CONFIDENCE = {"Infilteration", "Infiltration"}

STYLE = {
    "full_width": 7.16,
    "fig_dpi":    300,
    "font_size":  8,
    "tick_size":  7,
}


def parse_args():
    p = argparse.ArgumentParser(description="分析1：类间 SHAP 签名区分度")
    p.add_argument("--rq1-dir", required=True)
    p.add_argument("--out", default="./results/rq1_distinctiveness/")
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size":   STYLE["font_size"],
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": STYLE["tick_size"],
        "ytick.labelsize": STYLE["tick_size"],
        "axes.linewidth": 0.5,
        "savefig.dpi": STYLE["fig_dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_fig(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", format="pdf")
    fig.savefig(str(path_stem) + ".png", format="png")
    plt.close(fig)


def load_rq1(rq1_dir: Path) -> dict:
    """加载所有数据集的 RQ1 SHAP 签名"""
    all_sigs = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        pkl = rq1_dir / f"{ds_full}_rq1_shap_values.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                all_sigs[ds_short] = pickle.load(f)
            print(f"  ✓ {ds_short}: {sorted(all_sigs[ds_short].keys())}")
        else:
            print(f"  ⚠ 未找到: {pkl.name}")
    return all_sigs


def compute_distinctiveness_matrix(sigs: dict) -> tuple:
    """
    计算单个数据集的类间 Spearman ρ 矩阵。

    输入：{class_name: shap_result_dict}
    返回：(classes, rho_matrix, pval_matrix)
      classes: 类别名列表
      rho_matrix: shape (n_classes, n_classes)，对角线=1.0
      pval_matrix: shape (n_classes, n_classes)
    """
    classes = sorted(sigs.keys())
    n = len(classes)
    rho_mat  = np.eye(n)
    pval_mat = np.zeros((n, n))

    # 提取每个类别的 abs_mean_shap 向量
    vectors = {cls: sigs[cls]["abs_mean_shap"] for cls in classes}

    for i in range(n):
        for j in range(i + 1, n):
            rho, pval = stats.spearmanr(vectors[classes[i]],
                                        vectors[classes[j]])
            rho_mat[i, j]  = rho
            rho_mat[j, i]  = rho
            pval_mat[i, j] = pval
            pval_mat[j, i] = pval

    return classes, rho_mat, pval_mat


def plot_distinctiveness_heatmap(classes: list, rho_mat: np.ndarray,
                                 pval_mat: np.ndarray,
                                 ds_short: str, out_stem: Path):
    """绘制单数据集的区分度热力图"""
    n = len(classes)
    size = max(2.8, n * 0.45)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))

    # 颜色：ρ 高（签名相似）→ 红；ρ 低（签名不同）→ 白
    # 用 RdYlGn_r：红=相似，绿=差异大
    im = ax.imshow(rho_mat, cmap="RdYlGn_r", vmin=-0.2, vmax=1.0,
                   aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       fontsize=STYLE["tick_size"])
    ax.set_yticklabels(classes, fontsize=STYLE["tick_size"])

    # 标注数值 + 低置信度类别
    for i in range(n):
        for j in range(n):
            v = rho_mat[i, j]
            if i == j:
                # 对角线：白色覆盖
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             color="white", zorder=2))
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#aaaaaa", zorder=3)
            else:
                # 显著性标记
                sig = ""
                if pval_mat[i, j] < 0.001:  sig = "***"
                elif pval_mat[i, j] < 0.01: sig = "**"
                elif pval_mat[i, j] < 0.05: sig = "*"
                tc = "white" if v > 0.75 else "#222222"
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center",
                        fontsize=6.5, color=tc, zorder=3)

    # 低置信度类别用虚线框标记
    for i, cls in enumerate(classes):
        if cls in LOW_CONFIDENCE:
            rect = plt.Rectangle((i - 0.5, -0.5), 1, n,
                                  fill=False, edgecolor="#888888",
                                  linestyle="--", linewidth=1.2, zorder=4)
            ax.add_patch(rect)
            rect2 = plt.Rectangle((-0.5, i - 0.5), n, 1,
                                   fill=False, edgecolor="#888888",
                                   linestyle="--", linewidth=1.2, zorder=4)
            ax.add_patch(rect2)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Spearman ρ (SHAP signature similarity)", fontsize=7)

    # 统计注释
    off_diag = rho_mat[np.triu_indices(n, k=1)]
    mean_rho = off_diag.mean()
    pct_low  = (off_diag < 0.3).mean() * 100
    ax.set_title(
        f"({ds_short}) Pairwise SHAP signature similarity\n"
        f"mean ρ = {mean_rho:.3f}, "
        f"{pct_low:.0f}% of pairs ρ < 0.30",
        loc="left", fontsize=8, fontweight="bold"
    )

    note = "* p<0.05  ** p<0.01  *** p<0.001   -- low-confidence class"
    fig.text(0.01, -0.02, note, fontsize=6, color="#666666")
    fig.tight_layout()
    save_fig(fig, out_stem)
    print(f"  ✓ 热力图: {out_stem.name}.pdf/.png")


def main():
    args = parse_args()
    rq1_dir = Path(args.rq1_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"\n{'#'*60}")
    print(f"#  分析1：类间 SHAP 签名区分度矩阵")
    print(f"{'#'*60}\n")

    # 加载 RQ1 签名
    print("加载 RQ1 SHAP 签名...")
    all_sigs = load_rq1(rq1_dir)

    summary_rows = []

    for ds in DS_ORDER:
        if ds not in all_sigs:
            continue

        print(f"\n[{ds}] 计算类间区分度矩阵")
        sigs  = all_sigs[ds]
        classes, rho_mat, pval_mat = compute_distinctiveness_matrix(sigs)
        n = len(classes)

        # 保存矩阵 CSV
        df_rho = pd.DataFrame(rho_mat, index=classes,
                              columns=classes).round(4)
        csv_path = out_dir / f"{ds}_distinctiveness_matrix.csv"
        df_rho.to_csv(csv_path)
        print(f"  ✓ 矩阵 CSV: {csv_path.name}")

        # 绘图
        out_stem = out_dir / f"{ds}_heatmap"
        plot_distinctiveness_heatmap(
            classes, rho_mat, pval_mat, ds, out_stem
        )

        # 汇总统计
        off_diag  = rho_mat[np.triu_indices(n, k=1)]
        n_pairs   = len(off_diag)
        mean_rho  = float(off_diag.mean())
        median_rho = float(np.median(off_diag))
        pct_lt03  = float((off_diag < 0.30).mean() * 100)
        pct_lt05  = float((off_diag < 0.50).mean() * 100)
        max_rho   = float(off_diag.max())
        min_rho   = float(off_diag.min())

        # 找最相似的配对（ρ 最高，签名最近）
        idx = np.triu_indices(n, k=1)
        max_pos = off_diag.argmax()
        most_similar_pair = (classes[idx[0][max_pos]],
                             classes[idx[1][max_pos]])
        min_pos = off_diag.argmin()
        most_distinct_pair = (classes[idx[0][min_pos]],
                              classes[idx[1][min_pos]])

        row = {
            "dataset":           ds,
            "n_classes":         n,
            "n_pairs":           n_pairs,
            "mean_rho":          round(mean_rho, 4),
            "median_rho":        round(median_rho, 4),
            "min_rho":           round(min_rho, 4),
            "max_rho":           round(max_rho, 4),
            "pct_pairs_lt_0.30": round(pct_lt03, 1),
            "pct_pairs_lt_0.50": round(pct_lt05, 1),
            "most_similar_pair": f"{most_similar_pair[0]} ↔ {most_similar_pair[1]}",
            "most_similar_rho":  round(max_rho, 4),
            "most_distinct_pair": f"{most_distinct_pair[0]} ↔ {most_distinct_pair[1]}",
            "most_distinct_rho": round(min_rho, 4),
        }
        summary_rows.append(row)

        print(f"  类别数: {n}  配对数: {n_pairs}")
        print(f"  mean ρ = {mean_rho:.4f}  median ρ = {median_rho:.4f}")
        print(f"  ρ < 0.30: {pct_lt03:.1f}%  ρ < 0.50: {pct_lt05:.1f}%")
        print(f"  最相似配对: {most_similar_pair[0]} ↔ {most_similar_pair[1]} "
              f"(ρ={max_rho:.4f})")
        print(f"  最差异配对: {most_distinct_pair[0]} ↔ {most_distinct_pair[1]} "
              f"(ρ={min_rho:.4f})")

    # 保存汇总
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ 汇总 CSV: {summary_csv}")

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"✓ 汇总 JSON: {summary_json}")

    # ── 绘制跨数据集汇总图（均值ρ + 低ρ比例）────────────────────────────────
    print("\n绘制跨数据集汇总图...")
    fig, axes = plt.subplots(1, 2, figsize=(STYLE["full_width"], 2.2))

    ds_labels = [r["dataset"] for r in summary_rows]
    mean_rhos = [r["mean_rho"] for r in summary_rows]
    pct_lt03s = [r["pct_pairs_lt_0.30"] for r in summary_rows]

    colors = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b"][:len(ds_labels)]

    # 左图：mean ρ（越低越好，说明签名越特异）
    bars = axes[0].barh(ds_labels, mean_rhos, height=0.5,
                        color=colors, alpha=0.80)
    axes[0].axvline(0.3, color="#888888", linestyle="--",
                    linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel("Mean pairwise Spearman ρ")
    axes[0].set_title("(a) Signature similarity (lower = more distinctive)",
                      loc="left", fontsize=8, fontweight="bold")
    axes[0].set_xlim(0, 0.8)
    for bar, v in zip(bars, mean_rhos):
        axes[0].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:.3f}", va="center", fontsize=7)
    axes[0].text(0.3, -0.6, "ρ=0.30\nthreshold",
                 ha="center", fontsize=6, color="#888888")

    # 右图：% 配对 ρ < 0.30（越高越好）
    bars2 = axes[1].barh(ds_labels, pct_lt03s, height=0.5,
                         color=colors, alpha=0.80)
    axes[1].set_xlabel("% of pairs with ρ < 0.30")
    axes[1].set_title("(b) Fraction of highly distinctive pairs",
                      loc="left", fontsize=8, fontweight="bold")
    axes[1].set_xlim(0, 110)
    axes[1].axvline(50, color="#888888", linestyle="--",
                    linewidth=0.8, alpha=0.7)
    for bar, v in zip(bars2, pct_lt03s):
        axes[1].text(v + 1, bar.get_y() + bar.get_height() / 2,
                     f"{v:.0f}%", va="center", fontsize=7)

    pass  # caption is given in the paper
    fig.tight_layout()
    out_stem = out_dir / "summary_figure"
    save_fig(fig, out_stem)
    print(f"✓ 汇总图: {out_stem.name}.pdf/.png")

    print(f"\n✓ 分析1全部完成，输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
