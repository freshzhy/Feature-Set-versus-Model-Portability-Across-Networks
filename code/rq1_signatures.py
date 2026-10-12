#!/usr/bin/env python3
"""
RQ1：攻击特异性 SHAP 特征签名提取
第二篇NIDS论文 — 核心实验

研究问题：不同攻击类型是否具有各自独特的 SHAP 特征签名？

方法：
  对每种攻击类别，训练 one-vs-rest XGBoost 分类器，
  在测试集上计算 SHAP 值，提取 Top-K 特征签名。
  记录每个类别的特征重要性排序向量，供 RQ2 跨数据集比较使用。

输出（保存到 --out 目录）：
  {dataset}_rq1_shap_values.pkl     — 每个类别的原始 SHAP 均值向量
  {dataset}_rq1_top{K}_signatures.csv — Top-K 特征签名表（类别 × 特征排名）
  {dataset}_rq1_model_metrics.csv   — 每个类别的分类性能指标
  {dataset}_rq1_summary.json        — 实验元信息（可复现性记录）

用法：
  # 先在 UNSW-NB15 上验证
  python code/rq1_signatures.py --data ./data/processed/NF-UNSW-NB15-v2_processed.csv

  # 指定输出目录和 Top-K
  python code/rq1_signatures.py --data ./data/processed/NF-UNSW-NB15-v2_processed.csv \\
      --out ./results/rq1_signatures/ --topk 10

  # 跳过 Benign（只分析攻击类别的签名）
  python code/rq1_signatures.py --data ./data/processed/NF-UNSW-NB15-v2_processed.csv \\
      --skip-benign

  # 限制 SHAP 背景样本数（内存不足时）
  python code/rq1_signatures.py --data ./data/processed/NF-UNSW-NB15-v2_processed.csv \\
      --shap-background 500
"""

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# 实验配置
# ══════════════════════════════════════════════════════════════════════════════
EXP_CONFIG = {
    # XGBoost 超参数（与第一篇保持一致，确保可比性）
    "xgb_params": {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
        "tree_method":      "hist",   # Apple Silicon 推荐
    },

    # 训练/测试划分
    "test_size":  0.2,
    "random_seed": 42,

    # one-vs-rest 正例采样上限（正例过多时对 XGBoost 无益，且拖慢 SHAP 计算）
    # None = 不限制
    "max_positive_train": 50_000,

    # SHAP TreeExplainer 背景样本数
    # 背景样本用于近似 SHAP 期望值，100-1000 之间通常足够
    "shap_background_size": 1_000,

    # Top-K 特征签名（默认值，可被 --topk 覆盖）
    "default_topk": 10,

    # one-vs-rest 中，负例（other）的采样比例上限（相对正例数量的倍数）
    # 避免极端类别不平衡导致模型退化
    "neg_ratio_cap": 10,

    # 标签列名
    "multi_label_col":  "Attack",
    "binary_label_col": "Label",

    # 跳过的类别（分析攻击签名，Benign 不做 one-vs-rest）
    "skip_classes": ["Benign"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="RQ1 SHAP 攻击特征签名提取")
    parser.add_argument("--data", required=True,
                        help="预处理后的 CSV 路径（来自 preprocess.py 输出）")
    parser.add_argument("--out", default="./results/rq1_signatures/",
                        help="输出目录（默认 ./results/rq1_signatures/）")
    parser.add_argument("--topk", type=int, default=EXP_CONFIG["default_topk"],
                        help=f"Top-K 特征签名（默认 {EXP_CONFIG['default_topk']}）")
    parser.add_argument("--skip-benign", action="store_true",
                        help="跳过 Benign 类别（默认已跳过，此参数为显式声明）")
    parser.add_argument("--shap-background", type=int,
                        default=EXP_CONFIG["shap_background_size"],
                        help=f"SHAP 背景样本数（默认 {EXP_CONFIG['shap_background_size']}，"
                             "内存不足时可降至 200-500）")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="只处理指定类别（调试用，例：--classes DoS Backdoor）")
    return parser.parse_args()


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_data(data_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    读取预处理后的 CSV，返回 (df, feature_cols, attack_classes)。
    feature_cols: 40 个特征列名
    attack_classes: 所有攻击类别（不含 Benign）
    """
    section("数据加载")
    print(f"读取: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"完成: {df.shape[0]:,d} 行 × {df.shape[1]} 列")

    attack_col   = EXP_CONFIG["multi_label_col"]
    label_cols   = [EXP_CONFIG["binary_label_col"], attack_col]
    feature_cols = [c for c in df.columns if c not in label_cols]

    all_classes    = sorted(df[attack_col].unique())
    attack_classes = [c for c in all_classes
                      if c not in EXP_CONFIG["skip_classes"]]

    print(f"\n特征列: {len(feature_cols)} 个")
    print(f"全部类别: {all_classes}")
    print(f"攻击类别（待分析）: {attack_classes}")

    # 各类别样本量
    vc = df[attack_col].value_counts().sort_values(ascending=False)
    print(f"\n类别分布:")
    for cls, cnt in vc.items():
        print(f"  {str(cls):>30s}: {cnt:>10,d}")

    return df, feature_cols, attack_classes


# ══════════════════════════════════════════════════════════════════════════════
# One-vs-Rest 数据准备
# ══════════════════════════════════════════════════════════════════════════════
def build_ovr_dataset(df: pd.DataFrame, feature_cols: list[str],
                      target_class: str) -> tuple:
    """
    为目标攻击类别构建 one-vs-rest 二分类数据集。

    正例：目标攻击类别样本
    负例：其余所有类别样本（按 neg_ratio_cap 上限采样）

    返回：X_train, X_test, y_train, y_test, n_pos, n_neg
    """
    cfg = EXP_CONFIG
    attack_col = cfg["multi_label_col"]

    pos_df = df[df[attack_col] == target_class]
    neg_df = df[df[attack_col] != target_class]

    n_pos_orig = len(pos_df)
    n_neg_orig = len(neg_df)

    # 正例上限采样（保留分层随机性）
    if cfg["max_positive_train"] and n_pos_orig > cfg["max_positive_train"]:
        pos_df = pos_df.sample(n=cfg["max_positive_train"],
                               random_state=cfg["random_seed"])

    # 负例上限：不超过正例的 neg_ratio_cap 倍
    n_pos = len(pos_df)
    neg_cap = n_pos * cfg["neg_ratio_cap"]
    if n_neg_orig > neg_cap:
        neg_df = neg_df.sample(n=int(neg_cap), random_state=cfg["random_seed"])

    # 合并并标记标签
    combined = pd.concat([
        pos_df[feature_cols].assign(_y=1),
        neg_df[feature_cols].assign(_y=0),
    ], ignore_index=True)

    X = combined[feature_cols].values.astype(np.float64)
    y = combined["_y"].values

    # 清理异常值：inf/-inf 以及超大整数转 float64 后的溢出值
    # SRC_TO_DST_SECOND_BYTES 等字段在零时长流量时产生极大整数，
    # 转 float64 后变成 inf 或接近 float64 上限，XGBoost QuantileDMatrix 会拒绝
    # 策略：先处理 inf，再将绝对值超过 1e15 的值裁剪到列中位数
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        n_inf = int(inf_mask.sum())
        col_inf = np.where(inf_mask.any(axis=0))[0]
        print(f"  [Inf修复] 发现 {n_inf} 个 inf/-inf，涉及列 {col_inf}，替换为 0")
        X[inf_mask] = 0.0

    # 裁剪超大值（超过 float32 最大值 ~3.4e38 的 1/1000，即 3.4e35 的值）
    # 用列中位数替换，保留分布形状
    CLIP_THRESHOLD = 1e15
    large_mask = np.abs(X) > CLIP_THRESHOLD
    if large_mask.any():
        n_large = int(large_mask.sum())
        col_large = np.where(large_mask.any(axis=0))[0]
        print(f"  [极值修复] 发现 {n_large} 个 |值| > 1e15，涉及列 {col_large}，替换为列中位数")
        for col_idx in col_large:
            col_data = X[:, col_idx]
            col_mask = large_mask[:, col_idx]
            median_val = float(np.median(col_data[~col_mask])) if (~col_mask).any() else 0.0
            col_data[col_mask] = median_val
            X[:, col_idx] = col_data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, n_pos_orig, len(neg_df)


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost 训练
# ══════════════════════════════════════════════════════════════════════════════
def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    params = EXP_CONFIG["xgb_params"].copy()
    # scale_pos_weight：处理类别不平衡
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    params["missing"] = np.nan   # 显式声明缺失值标记，避免 inf 触发报错
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 模型评估
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model: xgb.XGBClassifier,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   target_class: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "class":        target_class,
        "f1":           f1_score(y_test, y_pred, zero_division=0),
        "precision":    precision_score(y_test, y_pred, zero_division=0),
        "recall":       recall_score(y_test, y_pred, zero_division=0),
        "auc_roc":      roc_auc_score(y_test, y_proba),
        "support_pos":  int((y_test == 1).sum()),
        "support_neg":  int((y_test == 0).sum()),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SHAP 签名提取
# ══════════════════════════════════════════════════════════════════════════════
def extract_shap_signature(model: xgb.XGBClassifier,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           feature_cols: list[str],
                           background_size: int) -> dict:
    """
    计算测试集上的 SHAP 值，提取攻击特征签名。

    签名定义：仅在正例（目标攻击类别）样本上计算 SHAP 均值，
    体现"该类攻击相对于所有其他流量的特征贡献方向与大小"。

    返回：
      "mean_shap"    : 每个特征的 SHAP 均值（正 = 推向攻击判定，负 = 推向正常判定）
      "abs_mean_shap": SHAP 绝对值均值（特征重要性，用于 Top-K 排序）
      "feature_cols" : 对应的特征名列表
    """
    rng = np.random.default_rng(42)

    # 只在正例上计算 SHAP（攻击样本的特征签名）
    pos_indices = np.where(y_test == 1)[0]
    if len(pos_indices) > 5_000:
        pos_indices = rng.choice(pos_indices, size=5_000, replace=False)

    # tree_path_dependent 模式：不需要背景数据集，使用训练时的树结构
    # 避免 interventional 模式的 categorical split 限制和背景样本上限问题
    # 对于纯数值特征（本数据集全为数值）两种模式结果高度一致
    explainer = shap.TreeExplainer(model,
                                   feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test[pos_indices])

    # shap 新版对二分类返回 shape (n, features)，旧版返回 (n, features, 2)
    # 统一取正类（index 1）的 SHAP 值
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    n_bg = 0  # tree_path_dependent 模式不使用背景样本

    mean_shap     = shap_values.mean(axis=0)        # 带符号均值
    abs_mean_shap = np.abs(shap_values).mean(axis=0)  # 绝对值均值（重要性）

    return {
        "mean_shap":     mean_shap,
        "abs_mean_shap": abs_mean_shap,
        "feature_cols":  feature_cols,
        "n_pos_shap":    len(pos_indices),
        "n_bg":          n_bg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"错误: 文件不存在 → {data_path}")
        sys.exit(1)

    dataset_name = data_path.stem.replace("_processed", "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"#  RQ1 SHAP 签名提取")
    print(f"#  数据集: {dataset_name}")
    print(f"#  Top-K:  {args.topk}")
    print(f"#  输出:   {out_dir}")
    print(f"{'#'*70}")

    exp_start = time.time()

    # ── 1. 加载数据 ───────────────────────────────────────────────────────────
    df, feature_cols, attack_classes = load_data(data_path)

    # 若指定 --classes，只处理指定子集（调试用）
    if args.classes:
        attack_classes = [c for c in attack_classes if c in args.classes]
        print(f"\n[调试模式] 只处理: {attack_classes}")

    # ── 2. 逐类别提取 SHAP 签名 ───────────────────────────────────────────────
    section(f"逐类别 One-vs-Rest 训练 + SHAP 提取（共 {len(attack_classes)} 类）")

    all_signatures = {}   # {class_name: shap_result_dict}
    all_metrics    = []   # [metrics_dict, ...]

    for i, cls in enumerate(attack_classes, 1):
        cls_start = time.time()
        print(f"\n[{i}/{len(attack_classes)}] {cls}")
        print(f"  {'─'*60}")

        # 2a. 构建 OvR 数据集
        X_train, X_test, y_train, y_test, n_pos, n_neg = build_ovr_dataset(
            df, feature_cols, cls
        )
        print(f"  OvR 数据集: 正例(原) {n_pos:,d} | 负例(采样后) {n_neg:,d}")
        print(f"  训练集: {len(X_train):,d}  测试集: {len(X_test):,d}")
        print(f"  测试集正例: {int((y_test==1).sum()):,d}  "
              f"负例: {int((y_test==0).sum()):,d}")

        # 2b. 训练 XGBoost
        t0 = time.time()
        model = train_xgb(X_train, y_train)
        print(f"  XGBoost 训练: {fmt_time(time.time()-t0)}")

        # 2c. 评估
        metrics = evaluate_model(model, X_test, y_test, cls)
        all_metrics.append(metrics)
        print(f"  性能: F1={metrics['f1']:.4f}  "
              f"Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  "
              f"AUC-ROC={metrics['auc_roc']:.4f}")

        # 2d. SHAP 签名提取
        t0 = time.time()
        sig = extract_shap_signature(
            model, X_test, y_test, feature_cols, args.shap_background
        )
        all_signatures[cls] = sig
        print(f"  SHAP 计算: {fmt_time(time.time()-t0)}  "
              f"（正例样本 {sig['n_pos_shap']:,d}，背景 {sig['n_bg']:,d}）")

        # 打印 Top-K
        top_idx = np.argsort(sig["abs_mean_shap"])[::-1][:args.topk]
        print(f"\n  Top-{args.topk} 特征签名:")
        print(f"  {'Rank':>4}  {'Feature':35}  {'|SHAP|':>10}  {'SHAP':>10}  {'方向'}")
        print(f"  {'─'*4}  {'─'*35}  {'─'*10}  {'─'*10}  {'─'*4}")
        for rank, idx in enumerate(top_idx, 1):
            feat  = feature_cols[idx]
            abs_v = sig["abs_mean_shap"][idx]
            raw_v = sig["mean_shap"][idx]
            direc = "↑攻击" if raw_v > 0 else "↓正常"
            print(f"  {rank:>4}  {feat:35}  {abs_v:>10.6f}  {raw_v:>10.6f}  {direc}")

        print(f"\n  类别总耗时: {fmt_time(time.time()-cls_start)}")

    # ── 3. 汇总输出 ───────────────────────────────────────────────────────────
    section("保存实验结果")

    # 3a. 原始 SHAP 签名（pickle，供 RQ2 直接加载）
    sig_pkl = out_dir / f"{dataset_name}_rq1_shap_values.pkl"
    with open(sig_pkl, "wb") as f:
        pickle.dump(all_signatures, f)
    print(f"  ✓ SHAP 原始签名: {sig_pkl}")

    # 3b. Top-K 签名表（CSV，方便查阅和论文绘图）
    rows = []
    for cls, sig in all_signatures.items():
        top_idx = np.argsort(sig["abs_mean_shap"])[::-1][:args.topk]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "class":       cls,
                "rank":        rank,
                "feature":     feature_cols[idx],
                "abs_shap":    round(float(sig["abs_mean_shap"][idx]), 8),
                "mean_shap":   round(float(sig["mean_shap"][idx]), 8),
                "direction":   "attack" if sig["mean_shap"][idx] > 0 else "normal",
            })
    sig_csv = out_dir / f"{dataset_name}_rq1_top{args.topk}_signatures.csv"
    pd.DataFrame(rows).to_csv(sig_csv, index=False)
    print(f"  ✓ Top-{args.topk} 签名表: {sig_csv}")

    # 3c. 完整特征重要性矩阵（类别 × 特征，用于热力图）
    matrix_data = {}
    for cls, sig in all_signatures.items():
        matrix_data[cls] = sig["abs_mean_shap"]
    matrix_df = pd.DataFrame(matrix_data, index=feature_cols).T
    matrix_df.index.name = "class"
    matrix_csv = out_dir / f"{dataset_name}_rq1_shap_matrix.csv"
    matrix_df.to_csv(matrix_csv)
    print(f"  ✓ SHAP 矩阵 ({matrix_df.shape[0]}类 × {matrix_df.shape[1]}特征): {matrix_csv}")

    # 3d. 分类性能指标
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = out_dir / f"{dataset_name}_rq1_model_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  ✓ 模型性能指标: {metrics_csv}")

    # 3e. 实验元信息（可复现性记录）
    total_time = time.time() - exp_start
    summary = {
        "dataset":          dataset_name,
        "data_path":        str(data_path),
        "n_samples":        len(df),
        "n_features":       len(feature_cols),
        "feature_cols":     feature_cols,
        "attack_classes":   attack_classes,
        "topk":             args.topk,
        "shap_background":  args.shap_background,
        "exp_config":       {k: v for k, v in EXP_CONFIG.items()
                             if k != "xgb_params"},
        "xgb_params":       EXP_CONFIG["xgb_params"],
        "total_time_sec":   round(total_time, 1),
        "metrics_summary": {
            "mean_f1":    round(float(metrics_df["f1"].mean()), 4),
            "min_f1":     round(float(metrics_df["f1"].min()), 4),
            "max_f1":     round(float(metrics_df["f1"].max()), 4),
        },
    }
    summary_json = out_dir / f"{dataset_name}_rq1_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 实验摘要: {summary_json}")

    # ── 4. 控制台汇总 ─────────────────────────────────────────────────────────
    section("实验完成 — 性能汇总")
    print(f"\n  {'类别':>25s}  {'F1':>8}  {'AUC-ROC':>8}  {'正例数':>8}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*8}")
    for m in sorted(all_metrics, key=lambda x: -x["f1"]):
        print(f"  {m['class']:>25s}  {m['f1']:>8.4f}  "
              f"{m['auc_roc']:>8.4f}  {m['support_pos']:>8,d}")
    print(f"\n  平均 F1: {metrics_df['f1'].mean():.4f}  "
          f"(min {metrics_df['f1'].min():.4f} / max {metrics_df['f1'].max():.4f})")
    print(f"  总耗时: {fmt_time(total_time)}")
    print(f"\n  输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
