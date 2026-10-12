#!/usr/bin/env python3
"""
RQ2：SHAP 攻击特征签名的跨数据集迁移性分析
第二篇NIDS论文 — 核心实验

研究问题：在网络环境 A 上学到的攻击特征签名，在环境 B 上是否仍然成立？

方法：
  对在多个数据集中共同存在的攻击类别（DoS/DDoS/Reconnaissance），
  比较各数据集 SHAP 签名之间的相似性：
    - Jaccard 相似度：Top-K 特征集合的交集/并集
    - Spearman ρ：全特征空间上的排名相关性

  主实验：DoS（四数据集全覆盖，跨越 IT/IoT 两种网络环境）
  补充分析：DDoS、Reconnaissance（三数据集）

输入：
  RQ1 输出的 *_rq1_shap_values.pkl 文件（每个数据集一个）

输出（保存到 --out 目录）：
  rq2_jaccard_matrix_{class}.csv     — Jaccard 矩阵（数据集 × 数据集）
  rq2_spearman_matrix_{class}.csv    — Spearman ρ 矩阵
  rq2_topk_feature_sets_{class}.csv  — 各数据集的 Top-K 特征列表
  rq2_summary.json                   — 汇总结果（所有类别）
  rq2_transferability_report.csv     — 全量结果表（供论文直接引用）

用法：
  python code/rq2_similarity.py --rq1-dir ./results/rq1_signatures/ --out ./results/rq2_similarity/

  # 只分析 DoS（调试）
  python code/rq2_similarity.py --rq1-dir ./results/rq1_signatures/ --classes DoS

  # 调整 Top-K（需与 RQ1 保持一致）
  python code/rq2_similarity.py --rq1-dir ./results/rq1_signatures/ --topk 10
"""

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 跨数据集类别映射表
#
# 不同数据集对同类攻击的命名不一致，需要统一映射到语义标签。
# key   = 语义标签（论文中使用）
# value = {数据集名: 该数据集中对应的原始标签}
#
# 设计原则：
#   - DoS 主实验：四数据集全覆盖，语义无歧义
#   - DDoS/Reconnaissance：三数据集补充分析
#   - CIC-IDS2018 的 DoS 有多种变体，合并为 DoS 大类
# ══════════════════════════════════════════════════════════════════════════════
CROSS_DATASET_MAPPING = {
    # ── 主实验锚点（四数据集） ────────────────────────────────────────────────
    "DoS": {
        "NF-UNSW-NB15-v2":        ["DoS"],
        "NF-CSE-CIC-IDS2018-v2":  ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                                    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
        "NF-ToN-IoT-v2":          ["dos"],
        "NF-BoT-IoT-v2":          ["DoS"],
    },

    # ── 补充分析（三数据集） ─────────────────────────────────────────────────
    "DDoS": {
        "NF-CSE-CIC-IDS2018-v2":  ["DDOS attack-HOIC", "DDoS attacks-LOIC-HTTP"],
        "NF-ToN-IoT-v2":          ["ddos"],
        "NF-BoT-IoT-v2":          ["DDoS"],
    },
    "Reconnaissance": {
        "NF-UNSW-NB15-v2":        ["Reconnaissance"],
        "NF-ToN-IoT-v2":          ["scanning"],
        "NF-BoT-IoT-v2":          ["Reconnaissance"],
    },
}

# 数据集的网络环境标注（用于论文中的分组描述）
DATASET_ENV = {
    "NF-UNSW-NB15-v2":       "IT network (lab)",
    "NF-CSE-CIC-IDS2018-v2": "IT network (enterprise)",
    "NF-ToN-IoT-v2":         "IoT network",
    "NF-BoT-IoT-v2":         "IoT network (botnet)",
}

# 数据集的简称（图表标签用）
DATASET_SHORT = {
    "NF-UNSW-NB15-v2":       "UNSW",
    "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2":         "ToN",
    "NF-BoT-IoT-v2":         "BoT",
}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="RQ2 跨数据集 SHAP 签名迁移性分析")
    parser.add_argument("--rq1-dir", required=True,
                        help="RQ1 输出目录（含 *_rq1_shap_values.pkl 文件）")
    parser.add_argument("--out", default="./results/rq2_similarity/",
                        help="输出目录（默认 ./results/rq2_similarity/）")
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-K 特征数（默认 10，需与 RQ1 一致）")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="只分析指定语义类别，如 --classes DoS DDoS")
    return parser.parse_args()


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_rq1_results(rq1_dir: Path) -> dict:
    """
    加载所有数据集的 RQ1 SHAP 签名。
    返回：{dataset_name: {class_name: shap_result_dict}}
    """
    section("加载 RQ1 SHAP 签名")

    pkl_files = sorted(rq1_dir.glob("*_rq1_shap_values.pkl"))
    if not pkl_files:
        print(f"错误：未找到 *_rq1_shap_values.pkl 文件于 {rq1_dir}")
        sys.exit(1)

    all_sigs = {}
    for pkl_path in pkl_files:
        dataset_name = pkl_path.stem.replace("_rq1_shap_values", "")
        with open(pkl_path, "rb") as f:
            sigs = pickle.load(f)
        all_sigs[dataset_name] = sigs
        short = DATASET_SHORT.get(dataset_name, dataset_name)
        print(f"  ✓ {short:6s}  {len(sigs)} 类别: {sorted(sigs.keys())}")

    print(f"\n  共加载 {len(all_sigs)} 个数据集")
    return all_sigs


# ══════════════════════════════════════════════════════════════════════════════
# 签名合并（处理 CIC-IDS2018 的 DoS 多变体）
# ══════════════════════════════════════════════════════════════════════════════
def merge_signatures(sigs_by_class: dict, feature_cols: list[str]) -> np.ndarray:
    """
    当一个语义类别对应多个原始标签时（如 DoS 的四种变体），
    取各变体 abs_mean_shap 的加权平均（按 n_pos_shap 加权）。

    返回合并后的 abs_mean_shap 向量（shape: n_features）
    """
    if len(sigs_by_class) == 1:
        return list(sigs_by_class.values())[0]["abs_mean_shap"]

    total_weight = 0
    weighted_sum = np.zeros(len(feature_cols))

    for cls_name, sig in sigs_by_class.items():
        w = sig.get("n_pos_shap", 1)
        weighted_sum += sig["abs_mean_shap"] * w
        total_weight += w

    return weighted_sum / total_weight


# ══════════════════════════════════════════════════════════════════════════════
# 相似度计算
# ══════════════════════════════════════════════════════════════════════════════
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard 相似度：|A∩B| / |A∪B|"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


def spearman_correlation(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float, float]:
    """
    Spearman 秩相关系数。
    输入：abs_mean_shap 向量（全特征空间，形状相同）
    返回：(rho, p_value)
    """
    rho, pval = stats.spearmanr(vec_a, vec_b)
    return float(rho), float(pval)


def get_topk_features(abs_shap: np.ndarray, feature_cols: list[str],
                      k: int) -> tuple[set, list[str]]:
    """返回 Top-K 特征的集合和有序列表"""
    top_idx = np.argsort(abs_shap)[::-1][:k]
    top_feats = [feature_cols[i] for i in top_idx]
    return set(top_feats), top_feats


# ══════════════════════════════════════════════════════════════════════════════
# 单类别跨数据集分析
# ══════════════════════════════════════════════════════════════════════════════
def analyze_class(semantic_class: str,
                  all_sigs: dict,
                  mapping: dict,
                  topk: int,
                  feature_cols: list[str]) -> dict:
    """
    对一个语义类别做跨数据集分析。
    返回分析结果字典。
    """
    print(f"\n  语义类别: {semantic_class}")
    print(f"  {'─'*60}")

    # 找出哪些数据集包含此类别
    available_datasets = []
    dataset_abs_shap   = {}   # {dataset_name: merged_abs_shap_vector}
    dataset_topk_feats = {}   # {dataset_name: (set, list)}

    for ds_name, ds_sigs in all_sigs.items():
        raw_labels = mapping.get(ds_name, [])
        if not raw_labels:
            continue

        # 收集该数据集中所有匹配标签的签名
        matched = {lbl: ds_sigs[lbl] for lbl in raw_labels if lbl in ds_sigs}
        if not matched:
            print(f"    ⚠ {DATASET_SHORT.get(ds_name, ds_name):6s}: "
                  f"标签 {raw_labels} 不在已加载签名中，跳过")
            continue

        # 特征列表对齐检查
        ref_feats = matched[list(matched.keys())[0]]["feature_cols"]
        if ref_feats != feature_cols:
            print(f"    ⚠ {ds_name} 特征列与参考不一致，跳过")
            continue

        abs_shap = merge_signatures(matched, feature_cols)
        top_set, top_list = get_topk_features(abs_shap, feature_cols, topk)

        available_datasets.append(ds_name)
        dataset_abs_shap[ds_name]   = abs_shap
        dataset_topk_feats[ds_name] = (top_set, top_list)

        matched_labels = list(matched.keys())
        short = DATASET_SHORT.get(ds_name, ds_name)
        print(f"    ✓ {short:6s}: {matched_labels}")
        print(f"      Top-{topk}: {top_list}")

    if len(available_datasets) < 2:
        print(f"    ⚠ 可用数据集 < 2，跳过此类别")
        return None

    # ── Jaccard 矩阵 ─────────────────────────────────────────────────────────
    n = len(available_datasets)
    jaccard_matrix  = np.eye(n)
    spearman_matrix = np.eye(n)
    pval_matrix     = np.zeros((n, n))

    pairs_result = []

    for i, j in combinations(range(n), 2):
        ds_a = available_datasets[i]
        ds_b = available_datasets[j]
        set_a, list_a = dataset_topk_feats[ds_a]
        set_b, list_b = dataset_topk_feats[ds_b]

        jac  = jaccard_similarity(set_a, set_b)
        rho, pval = spearman_correlation(
            dataset_abs_shap[ds_a], dataset_abs_shap[ds_b]
        )

        jaccard_matrix[i, j]  = jac
        jaccard_matrix[j, i]  = jac
        spearman_matrix[i, j] = rho
        spearman_matrix[j, i] = rho
        pval_matrix[i, j]     = pval
        pval_matrix[j, i]     = pval

        short_a = DATASET_SHORT.get(ds_a, ds_a)
        short_b = DATASET_SHORT.get(ds_b, ds_b)
        intersection = set_a & set_b

        print(f"\n    {short_a} ↔ {short_b}:")
        print(f"      Jaccard Top-{topk}:  {jac:.4f}  "
              f"(交集 {len(intersection)}/{topk}: {sorted(intersection)})")
        print(f"      Spearman ρ:     {rho:+.4f}  (p={pval:.4f})")

        pairs_result.append({
            "semantic_class": semantic_class,
            "dataset_A":      ds_a,
            "dataset_B":      ds_b,
            "dataset_A_short": short_a,
            "dataset_B_short": short_b,
            "env_A":          DATASET_ENV.get(ds_a, "unknown"),
            "env_B":          DATASET_ENV.get(ds_b, "unknown"),
            "jaccard":        round(jac,  4),
            "spearman_rho":   round(rho,  4),
            "spearman_pval":  round(pval, 6),
            "n_intersection": len(intersection),
            "intersection_features": sorted(intersection),
            "topk_A":         list_a,
            "topk_B":         list_b,
        })

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    jac_vals = [r["jaccard"]      for r in pairs_result]
    rho_vals = [r["spearman_rho"] for r in pairs_result]

    print(f"\n    汇总 ({semantic_class}, {len(available_datasets)} 个数据集):")
    print(f"      Jaccard:  mean={np.mean(jac_vals):.4f}  "
          f"min={np.min(jac_vals):.4f}  max={np.max(jac_vals):.4f}")
    print(f"      Spearman: mean={np.mean(rho_vals):+.4f}  "
          f"min={np.min(rho_vals):+.4f}  max={np.max(rho_vals):+.4f}")

    # 解释性结论
    mean_jac = np.mean(jac_vals)
    mean_rho = np.mean(rho_vals)
    if mean_jac >= 0.5 and mean_rho >= 0.7:
        conclusion = "HIGH transferability — core signatures stable across networks"
    elif mean_jac >= 0.3 or mean_rho >= 0.5:
        conclusion = "MODERATE transferability — partial signature overlap"
    else:
        conclusion = "LOW transferability — signatures network-specific"
    print(f"      → {conclusion}")

    ds_labels = [DATASET_SHORT.get(d, d) for d in available_datasets]

    return {
        "semantic_class":    semantic_class,
        "available_datasets": available_datasets,
        "dataset_labels":    ds_labels,
        "dataset_topk_feats": {
            DATASET_SHORT.get(k, k): v[1]
            for k, v in dataset_topk_feats.items()
        },
        "jaccard_matrix":   jaccard_matrix.tolist(),
        "spearman_matrix":  spearman_matrix.tolist(),
        "pval_matrix":      pval_matrix.tolist(),
        "pairs":            pairs_result,
        "summary": {
            "jaccard_mean":  round(float(np.mean(jac_vals)),  4),
            "jaccard_min":   round(float(np.min(jac_vals)),   4),
            "jaccard_max":   round(float(np.max(jac_vals)),   4),
            "spearman_mean": round(float(np.mean(rho_vals)),  4),
            "spearman_min":  round(float(np.min(rho_vals)),   4),
            "spearman_max":  round(float(np.max(rho_vals)),   4),
            "n_datasets":    len(available_datasets),
            "n_pairs":       len(pairs_result),
            "conclusion":    conclusion,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ══════════════════════════════════════════════════════════════════════════════
def save_results(results: dict, out_dir: Path, topk: int):
    section("保存分析结果")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    for sem_cls, res in results.items():
        if res is None:
            continue

        ds_labels = res["dataset_labels"]

        # ── Jaccard 矩阵 CSV ────────────────────────────────────────────────
        jac_df = pd.DataFrame(
            res["jaccard_matrix"],
            index=ds_labels, columns=ds_labels
        ).round(4)
        jac_path = out_dir / f"rq2_jaccard_{sem_cls}.csv"
        jac_df.to_csv(jac_path)
        print(f"  ✓ Jaccard 矩阵 ({sem_cls}): {jac_path.name}")

        # ── Spearman 矩阵 CSV ───────────────────────────────────────────────
        spe_df = pd.DataFrame(
            res["spearman_matrix"],
            index=ds_labels, columns=ds_labels
        ).round(4)
        spe_path = out_dir / f"rq2_spearman_{sem_cls}.csv"
        spe_df.to_csv(spe_path)
        print(f"  ✓ Spearman 矩阵 ({sem_cls}): {spe_path.name}")

        # ── Top-K 特征集合 CSV ──────────────────────────────────────────────
        topk_rows = []
        for ds_short, feat_list in res["dataset_topk_feats"].items():
            for rank, feat in enumerate(feat_list, 1):
                topk_rows.append({
                    "semantic_class": sem_cls,
                    "dataset":        ds_short,
                    "rank":           rank,
                    "feature":        feat,
                })
        topk_df = pd.DataFrame(topk_rows)
        topk_path = out_dir / f"rq2_topk_{sem_cls}.csv"
        topk_df.to_csv(topk_path, index=False)
        print(f"  ✓ Top-{topk} 特征集合 ({sem_cls}): {topk_path.name}")

        # 收集全量 pairs
        all_pairs.extend(res["pairs"])

    # ── 全量结果表（论文直接引用） ──────────────────────────────────────────
    if all_pairs:
        report_rows = []
        for p in all_pairs:
            report_rows.append({
                "semantic_class":      p["semantic_class"],
                "dataset_pair":        f"{p['dataset_A_short']}↔{p['dataset_B_short']}",
                "env_pair":            f"{p['env_A']} / {p['env_B']}",
                "jaccard":             p["jaccard"],
                "spearman_rho":        p["spearman_rho"],
                "spearman_pval":       p["spearman_pval"],
                "n_shared_features":   p["n_intersection"],
                "shared_features":     ", ".join(p["intersection_features"]),
            })
        report_df = pd.DataFrame(report_rows)
        report_path = out_dir / "rq2_transferability_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"  ✓ 全量结果表: {report_path.name}")

    # ── 汇总 JSON ────────────────────────────────────────────────────────────
    summary = {
        cls: res["summary"] if res else None
        for cls, res in results.items()
    }
    summary_path = out_dir / "rq2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 汇总 JSON: {summary_path.name}")

    return report_df if all_pairs else None


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rq1_dir = Path(args.rq1_dir)
    out_dir  = Path(args.out)

    if not rq1_dir.exists():
        print(f"错误：RQ1 目录不存在 → {rq1_dir}")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print(f"#  RQ2 跨数据集 SHAP 签名迁移性分析")
    print(f"#  RQ1 目录: {rq1_dir}")
    print(f"#  Top-K:   {args.topk}")
    print(f"{'#'*70}")

    # ── 加载 RQ1 结果 ─────────────────────────────────────────────────────────
    all_sigs = load_rq1_results(rq1_dir)

    # 取特征列（从第一个数据集第一个类别中提取，全部数据集特征一致）
    first_ds   = list(all_sigs.values())[0]
    first_cls  = list(first_ds.values())[0]
    feature_cols = first_cls["feature_cols"]
    print(f"\n  特征维度: {len(feature_cols)} 个")

    # ── 确定要分析的语义类别 ─────────────────────────────────────────────────
    target_classes = args.classes if args.classes else list(CROSS_DATASET_MAPPING.keys())
    print(f"  分析类别: {target_classes}")

    # ── 逐类别分析 ───────────────────────────────────────────────────────────
    section(f"逐类别跨数据集相似度分析（Top-{args.topk}）")

    results = {}
    for sem_cls in target_classes:
        if sem_cls not in CROSS_DATASET_MAPPING:
            print(f"\n  ⚠ '{sem_cls}' 不在映射表中，跳过")
            continue
        mapping = CROSS_DATASET_MAPPING[sem_cls]
        results[sem_cls] = analyze_class(
            sem_cls, all_sigs, mapping, args.topk, feature_cols
        )

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    report_df = save_results(results, out_dir, args.topk)

    # ── 控制台总览 ────────────────────────────────────────────────────────────
    section("RQ2 分析完成 — 迁移性总览")
    print(f"\n  {'语义类别':>15s}  {'数据集数':>6s}  {'Jaccard均值':>11s}  "
          f"{'Spearman均值':>12s}  结论")
    print(f"  {'─'*15}  {'─'*6}  {'─'*11}  {'─'*12}  {'─'*40}")

    for sem_cls, res in results.items():
        if res is None:
            print(f"  {sem_cls:>15s}  （数据不足，跳过）")
            continue
        s = res["summary"]
        # 结论简化
        level = s["conclusion"].split(" ")[0]   # HIGH / MODERATE / LOW
        level_zh = {"HIGH": "高迁移性", "MODERATE": "中等迁移性",
                    "LOW": "低迁移性"}.get(level, level)
        print(f"  {sem_cls:>15s}  {s['n_datasets']:>6d}  "
              f"{s['jaccard_mean']:>11.4f}  "
              f"{s['spearman_mean']:>+12.4f}  {level_zh}")

    print(f"\n  输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
