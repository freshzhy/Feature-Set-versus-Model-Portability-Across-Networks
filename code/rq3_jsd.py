#!/usr/bin/env python3
"""
rq3_jsd.py — 攻击流量分布的 Jensen-Shannon 散度 (JSD) 作为迁移增益的预测因子（Fig. 8）
=============================================================
步骤：
  1. 计算每个 source-target 配对的攻击流量分布 JSD
     （在共同 40 特征空间上，使用核密度估计或直方图近似）
  2. 计算 JSD 与 transfer F1 增益（SHAP-10 − Random-10）的 Spearman 相关
  ΔF1 取自 results/rq3_main/per_config_tests.csv（SHAP-10 vs Random-10，std 测试集）。

JSD 计算策略：
  - 对每个攻击配对 (src_cls, tgt_cls) 提取各自测试集正例的特征分布
  - 每个特征独立计算 JSD（基于等宽直方图），取所有特征的均值 JSD
  - JSD ∈ [0, 1]：0 = 分布完全相同，1 = 完全不重叠
  - 用 Top-10 SHAP 特征的 JSD 作为主指标（只在最重要特征上比较分布）

输出：
  jsd_matrix.csv         — 所有配对的 JSD 值
  jsd_transfer_corr.csv  — JSD vs Δ F1 相关分析
  jsd_summary.json           — 汇总结论

用法：
  python code/rq3_jsd.py \\
      --rq1-dir ./results/rq1_signatures/ \\
      --data-dir ./data/processed/ \\
      --out ./results/rq3_jsd/
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
DATASET_SHORT = {
    "NF-UNSW-NB15-v2":       "UNSW",
    "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2":         "ToN",
    "NF-BoT-IoT-v2":         "BoT",
}
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]
DS_CSV = {
    "UNSW": "NF-UNSW-NB15-v2_processed.csv",
    "CIC":  "NF-CSE-CIC-IDS2018-v2_processed.csv",
    "ToN":  "NF-ToN-IoT-v2_processed.csv",
    "BoT":  "NF-BoT-IoT-v2_processed.csv",
}
DS_ENV = {"UNSW": "IT", "CIC": "IT", "ToN": "IoT", "BoT": "IoT"}

SEMANTIC_MAPPING = {
    "DoS": {
        "UNSW": ["DoS"],
        "CIC":  ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                 "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
        "ToN":  ["dos"],
        "BoT":  ["DoS"],
    },
    "Reconnaissance": {
        "UNSW": ["Reconnaissance"],
        "ToN":  ["scanning"],
        "BoT":  ["Reconnaissance"],
    },
}

N_BINS       = 50      # 直方图 bin 数（用于 JSD 计算）
N_SAMPLES    = 5000    # 每个类别最多取多少样本（控制计算量）
TOP_K        = 10      # 与 RQ3 一致
RANDOM_SEED  = 42

STYLE = {"full_width": 7.16, "fig_dpi": 300, "font_size": 8}

# 语义类别标记（用于图表）
CLS_MARKERS  = {"DoS": "o", "Reconnaissance": "s"}
CLS_COLORS   = {"DoS": "#2166ac", "Reconnaissance": "#d6604d"}


def parse_args():
    p = argparse.ArgumentParser(description="分析4：JSD 行为相似度")
    p.add_argument("--rq1-dir",      required=True)
    p.add_argument("--data-dir",     required=True)
    p.add_argument("--rq3-dir", default="./results/rq3_main/",
                   help="RQ3 main results (per_config_tests.csv) providing ΔF1 = SHAP-10 − Random-10")
    p.add_argument("--out", default="./results/rq3_jsd/")
    p.add_argument("--classes", nargs="+",
                   default=list(SEMANTIC_MAPPING.keys()))
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":     STYLE["font_size"],
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi":   STYLE["fig_dpi"],
        "savefig.bbox":  "tight",
        "savefig.pad_inches": 0.05,
    })


def save_fig(fig, stem: Path):
    fig.savefig(str(stem) + ".pdf", format="pdf")
    fig.savefig(str(stem) + ".png", format="png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_shap_sigs(rq1_dir: Path) -> dict:
    sigs = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        pkl = rq1_dir / f"{ds_full}_rq1_shap_values.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                sigs[ds_short] = pickle.load(f)
    return sigs


def get_topk(sigs: dict, raw_labels: list,
             feature_cols: list, k: int) -> list:
    matched = {l: sigs[l] for l in raw_labels if l in sigs}
    if not matched:
        return []
    total_w, weighted = 0, np.zeros(len(feature_cols))
    for sig in matched.values():
        w = sig.get("n_pos_shap", 1)
        weighted += sig["abs_mean_shap"] * w
        total_w  += w
    merged = weighted / total_w
    idx = np.argsort(merged)[::-1][:k]
    return [feature_cols[i] for i in idx]


def load_attack_samples(data_dir: Path, ds: str,
                        attack_labels: list,
                        feature_cols: list,
                        n_max: int = N_SAMPLES) -> np.ndarray | None:
    """
    从预处理后的 CSV 中提取攻击类别样本的特征矩阵。
    返回 shape (n_samples, n_features)，值为 float64。
    """
    csv = data_dir / DS_CSV[ds]
    if not csv.exists():
        return None

    df = pd.read_csv(csv, low_memory=False)
    mask = df["Attack"].isin(attack_labels)
    pos  = df[mask]

    if len(pos) == 0:
        return None

    rng = np.random.default_rng(RANDOM_SEED)
    if len(pos) > n_max:
        idx = rng.choice(len(pos), n_max, replace=False)
        pos = pos.iloc[idx]

    use_feats = [c for c in feature_cols if c in pos.columns]
    X = pos[use_feats].to_numpy(dtype=np.float64)

    # 清理极值
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) > 0 else 0.0

    return X


# ══════════════════════════════════════════════════════════════════════════════
# JSD 计算
# ══════════════════════════════════════════════════════════════════════════════
def feature_jsd(x: np.ndarray, y: np.ndarray,
                n_bins: int = N_BINS) -> float:
    """
    计算单个特征两个样本集之间的 Jensen-Shannon 散度。

    策略：等宽直方图（联合范围），加 epsilon 平滑避免零概率。
    返回 JSD ∈ [0, 1]（取 jensenshannon 的平方以满足散度定义）。
    """
    # 联合范围
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if hi <= lo:
        return 0.0  # 零方差特征，视为相同分布

    bins = np.linspace(lo, hi, n_bins + 1)
    eps  = 1e-10

    p, _ = np.histogram(x, bins=bins)
    q, _ = np.histogram(y, bins=bins)

    p = p.astype(float) + eps
    q = q.astype(float) + eps
    p /= p.sum()
    q /= q.sum()

    # jensenshannon() 返回 sqrt(JSD)，平方得到真正的 JSD
    return float(jensenshannon(p, q) ** 2)


def compute_pairwise_jsd(samples_a: np.ndarray,
                         samples_b: np.ndarray,
                         feat_names: list[str],
                         topk_feats: list[str] | None = None) -> dict:
    """
    计算两组攻击样本之间的多特征 JSD。

    返回：
      "mean_jsd_all"  : 所有特征的均值 JSD（全局行为相似度）
      "mean_jsd_topk" : Top-K 特征的均值 JSD（主特征空间相似度）
      "per_feature"   : {feature: jsd} 字典
    """
    n_feats = min(samples_a.shape[1], samples_b.shape[1], len(feat_names))
    per_feat = {}

    for fi in range(n_feats):
        fname = feat_names[fi]
        jsd   = feature_jsd(samples_a[:, fi], samples_b[:, fi])
        per_feat[fname] = jsd

    mean_all  = float(np.mean(list(per_feat.values())))

    if topk_feats:
        topk_jsds = [per_feat[f] for f in topk_feats if f in per_feat]
        mean_topk = float(np.mean(topk_jsds)) if topk_jsds else mean_all
    else:
        mean_topk = mean_all

    return {
        "mean_jsd_all":  mean_all,
        "mean_jsd_topk": mean_topk,
        "per_feature":   per_feat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主分析
# ══════════════════════════════════════════════════════════════════════════════
def run_jsd_analysis(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    主流程：计算所有配对的 JSD，与分析3的 Δ F1 合并，计算 Spearman 相关。
    返回 (jsd_df, corr_df)
    """
    rq1_dir  = Path(args.rq1_dir)
    data_dir = Path(args.data_dir)
    rq3_dir  = Path(args.rq3_dir)

    print("加载 RQ1 SHAP 签名...")
    all_sigs = load_shap_sigs(rq1_dir)

    # ΔF1 = SHAP-10 − Random-10 (1:10 test set, configuration mean) from the RQ3 main experiment
    per = pd.read_csv(rq3_dir / "per_config_tests.csv")
    per = per[(per.comparison == "SHAP-10 vs Random-10") & (per.testset == "std") & (per.metric == "f1")]
    a3_df = pd.DataFrame({"semantic_class": per.semantic_class, "pair": per.pair, "delta_mean": per.delta,
                          "cohen_d": per.d_z, "wilcoxon_p_corrected": per.p_nb_holm,
                          "significant_corrected": per.p_nb_holm < 0.05})
    print(f"Loaded RQ3 ΔF1 for {len(a3_df)} configurations")

    # 获取特征列
    first_ds  = list(all_sigs.values())[0]
    first_cls = list(first_ds.values())[0]
    feature_cols = first_cls["feature_cols"]
    print(f"特征维度: {len(feature_cols)}")

    jsd_rows = []

    for sem_cls in args.classes:
        if sem_cls not in SEMANTIC_MAPPING:
            continue
        mapping  = SEMANTIC_MAPPING[sem_cls]
        available = [ds for ds in DS_ORDER if ds in mapping and ds in all_sigs]

        print(f"\n语义类别: {sem_cls}  数据集: {available}")

        # 为每个数据集预加载攻击样本
        print("  加载攻击样本...")
        samples = {}
        topk_per_ds = {}

        for ds in available:
            labels   = mapping[ds]
            X = load_attack_samples(data_dir, ds, labels,
                                    feature_cols, N_SAMPLES)
            if X is not None:
                samples[ds] = X
                print(f"    {ds}: {X.shape[0]} 样本 × {X.shape[1]} 特征")

            # 该数据集的 Top-K SHAP 特征
            topk = get_topk(all_sigs[ds], labels, feature_cols, TOP_K)
            topk_per_ds[ds] = topk

        # 计算所有有序配对的 JSD（src → tgt，有方向）
        print("  计算 JSD...")
        for src_ds in available:
            for tgt_ds in available:
                if src_ds == tgt_ds:
                    continue
                if src_ds not in samples or tgt_ds not in samples:
                    continue

                X_src = samples[src_ds]
                X_tgt = samples[tgt_ds]

                # 使用源数据集的 Top-K 特征子集计算 JSD
                # （与 SHAP-10 迁移实验一致：用 src 的 Top-10 在 tgt 上测试）
                src_topk = topk_per_ds[src_ds]
                src_topk_idx = [feature_cols.index(f)
                                for f in src_topk if f in feature_cols]

                X_src_topk = X_src[:, src_topk_idx]
                X_tgt_topk = X_tgt[:, src_topk_idx]  # 用相同特征子集

                # 计算全特征 JSD
                jsd_all = compute_pairwise_jsd(
                    X_src, X_tgt, feature_cols, src_topk
                )

                # 计算 Top-K 特征子集 JSD
                jsd_topk_val = float(np.mean([
                    feature_jsd(X_src[:, i], X_tgt[:, i])
                    for i in src_topk_idx
                ]))

                env_same = DS_ENV.get(src_ds) == DS_ENV.get(tgt_ds)
                pair = f"{src_ds}→{tgt_ds}"

                jsd_rows.append({
                    "semantic_class":   sem_cls,
                    "source":           src_ds,
                    "target":           tgt_ds,
                    "pair":             pair,
                    "env_same":         env_same,
                    "env_src":          DS_ENV.get(src_ds, "?"),
                    "env_tgt":          DS_ENV.get(tgt_ds, "?"),
                    "jsd_all_feats":    round(jsd_all["mean_jsd_all"],  6),
                    "jsd_topk_feats":   round(jsd_topk_val,             6),
                    "n_src_samples":    X_src.shape[0],
                    "n_tgt_samples":    X_tgt.shape[0],
                    "src_topk":         str(src_topk),
                })

                print(f"    {pair}: JSD_all={jsd_all['mean_jsd_all']:.4f}  "
                      f"JSD_topk={jsd_topk_val:.4f}  "
                      f"env={'Same' if env_same else 'Cross'}")

    jsd_df = pd.DataFrame(jsd_rows)

    # ── 合并分析3的 Δ F1 ─────────────────────────────────────────────────────
    # a3_df 的 "pair" 格式也是 "src→tgt"
    merged = jsd_df.merge(
        a3_df[["semantic_class", "pair", "delta_mean",
               "cohen_d", "wilcoxon_p_corrected",
               "significant_corrected"]],
        on=["semantic_class", "pair"],
        how="left",
    )

    print(f"\n合并后: {len(merged)} 行  "
          f"（含 Δ F1: {merged['delta_mean'].notna().sum()} 行）")

    # ── Spearman 相关分析 ─────────────────────────────────────────────────────
    print("\n计算 JSD vs Δ F1 Spearman 相关...")
    corr_rows = []

    # 全类别合并
    valid = merged.dropna(subset=["delta_mean", "jsd_topk_feats"])

    for jsd_col in ["jsd_all_feats", "jsd_topk_feats"]:
        for cls_filter in [None] + args.classes:
            sub = valid if cls_filter is None \
                  else valid[valid["semantic_class"] == cls_filter]
            if len(sub) < 4:
                continue
            rho, pval = spearmanr(sub[jsd_col], sub["delta_mean"])
            corr_rows.append({
                "subset":       cls_filter or "All",
                "jsd_metric":   jsd_col,
                "n_pairs":      len(sub),
                "spearman_rho": round(float(rho),  4),
                "p_value":      round(float(pval), 6),
                "significant":  pval < 0.05,
                "interpretation": (
                    "Higher JSD → lower Δ F1 (more distant = worse transfer)"
                    if rho < -0.3 else
                    "Higher JSD → higher Δ F1 (unexpected)"
                    if rho > 0.3 else
                    "No clear monotonic relationship"
                ),
            })
            print(f"  [{cls_filter or 'All':15s}] {jsd_col}: "
                  f"ρ = {rho:+.4f}  p = {pval:.4f}  n = {len(sub)}")

    corr_df = pd.DataFrame(corr_rows)
    return merged, corr_df


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"\n{'#'*65}")
    print(f"#  分析4：JSD 行为相似度 + 迁移性相关分析（问题3修复）")
    print(f"#  分析类别: {args.classes}")
    print(f"{'#'*65}\n")

    # ── 主分析 ───────────────────────────────────────────────────────────────
    merged_df, corr_df = run_jsd_analysis(args)

    # ── 保存 CSV ─────────────────────────────────────────────────────────────
    jsd_csv = out_dir / "jsd_matrix.csv"
    merged_df.to_csv(jsd_csv, index=False)
    print(f"\n✓ JSD 矩阵: {jsd_csv}")

    corr_csv = out_dir / "jsd_transfer_corr.csv"
    corr_df.to_csv(corr_csv, index=False)
    print(f"✓ 相关分析: {corr_csv}")

    # ── 汇总 JSON（论文用）───────────────────────────────────────────────────
    summary = {
        "n_pairs":        len(merged_df),
        "jsd_range_all":  [
            round(float(merged_df["jsd_all_feats"].min()), 4),
            round(float(merged_df["jsd_all_feats"].max()), 4),
        ],
        "jsd_range_topk": [
            round(float(merged_df["jsd_topk_feats"].min()), 4),
            round(float(merged_df["jsd_topk_feats"].max()), 4),
        ],
        "spearman_results": corr_df.to_dict(orient="records"),
        "key_pairs": [],
    }

    # 找最高/最低 JSD 的显著配对
    sig = merged_df.dropna(subset=["delta_mean"])
    for _, row in sig.iterrows():
        summary["key_pairs"].append({
            "pair":          row["pair"],
            "class":         row["semantic_class"],
            "jsd_topk":      row["jsd_topk_feats"],
            "delta_f1":      row.get("delta_mean"),
            "significant":   row.get("significant_corrected"),
            "env_same":      row["env_same"],
        })

    with open(out_dir / "jsd_summary.json",
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if pd.isna(x) else x)
    print(f"✓ 汇总 JSON: jsd_summary.json")

    # ── 控制台结论 ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  JSD vs Δ F1 Spearman 相关汇总")
    print(f"{'='*65}")
    print(f"\n  {'子集':15s}  {'JSD指标':20s}  {'ρ':>8}  "
          f"{'p':>10}  {'n':>4}  {'显著':>5}")
    print(f"  {'─'*15}  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*4}  {'─'*5}")
    for _, r in corr_df.iterrows():
        print(f"  {r['subset']:15s}  {r['jsd_metric']:20s}  "
              f"{r['spearman_rho']:>+8.4f}  {r['p_value']:>10.4f}  "
              f"{r['n_pairs']:>4d}  {'✓' if r['significant'] else ''}")

    print(f"\n  输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
