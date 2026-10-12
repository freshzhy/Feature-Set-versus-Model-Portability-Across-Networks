#!/usr/bin/env python3
"""
rq1_stability.py — RQ1 签名在 30 个随机划分（seed 42–71）上的稳定性（Sections 3.2.4, 4.1.4）
======================================================
对 7 个 (语义类别, 数据集) 配置，每个 seed 重新划分并训练 OvR XGBoost，按与
rq1_signatures.py 完全相同的口径（测试集正例上的 mean |SHAP|）提取签名，然后统计
跨 seed 的 Top-10 Jaccard、40 维 Spearman 以及第 1 名特征的出现频率。

用法：
  python code/rq1_stability.py --data-dir ./data/processed/ --out ./results/rq1_stability/ --n-seeds 30
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── settings and helpers shared with the nested Target-10 protocol ──
DATASET_SHORT = {
    "NF-UNSW-NB15-v2": "UNSW", "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2": "ToN", "NF-BoT-IoT-v2": "BoT",
}
DS_CSV = {
    "UNSW": "NF-UNSW-NB15-v2_processed.csv",
    "CIC":  "NF-CSE-CIC-IDS2018-v2_processed.csv",
    "ToN":  "NF-ToN-IoT-v2_processed.csv",
    "BoT":  "NF-BoT-IoT-v2_processed.csv",
}
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
# Table VI-a 顺序
TARGET_ORDER = [("DoS", "UNSW"), ("DoS", "CIC"), ("DoS", "ToN"), ("DoS", "BoT"),
                ("Reconnaissance", "UNSW"), ("Reconnaissance", "ToN"), ("Reconnaissance", "BoT")]

NON_FEATURE_COLS = {"Label", "Attack"}

XGB_PARAMS = {
    "objective": "binary:logistic", "n_estimators": 300, "max_depth": 6,
    "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
    "eval_metric": "logloss", "n_jobs": -1, "tree_method": "hist",
    "missing": np.nan,
}
TOP_K = 10
TEST_SIZE = 0.20
MAX_TRAIN_POS = 30_000
NEG_RATIO_CAP = 10
SHAP_POS_CAP = 3_000   # SHAP 只在 train 正例上算，采样上限（控制耗时）


def load_dataset(data_dir: Path, ds: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / DS_CSV[ds], low_memory=False)


def clean_X(X: np.ndarray) -> np.ndarray:
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) > 0 else 0.0
    return X


def build_full40(df: pd.DataFrame, feature_cols: list, target_labels: list, seed: int):
    pos_idx = df.index[df["Attack"].isin(target_labels)].to_numpy()
    neg_idx = df.index[~df["Attack"].isin(target_labels)].to_numpy()
    if len(pos_idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, MAX_TRAIN_POS, replace=False)
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, int(neg_cap), replace=False)
    X = clean_X(np.vstack([df.loc[pos_idx, feature_cols].to_numpy(dtype=np.float64),
                            df.loc[neg_idx, feature_cols].to_numpy(dtype=np.float64)]))
    y = np.concatenate([np.ones(len(pos_idx), dtype=np.int32),
                         np.zeros(len(neg_idx), dtype=np.int32)])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def fit_xgb(X_tr, y_tr, seed):
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    params["random_state"] = seed
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, verbose=False)
    return m




def nested_round_test_positives(X_train, X_test, y_train, y_test, feature_cols, seed):
    """与 v1 相同的 nested 分类器训练设计，但 SHAP 向量改为在
    test-set positive samples 上计算，与正式 signature 定义（3.2.1 Eq.1）
    protocol 完全一致。"""
    import shap

    m_full = fit_xgb(X_train, y_train, seed)

    rng = np.random.default_rng(seed + 800000)  # 与 v1 的 +700000 区分
    pos_idx = np.where(y_test == 1)[0]
    if len(pos_idx) > SHAP_POS_CAP:
        pos_idx = rng.choice(pos_idx, SHAP_POS_CAP, replace=False)

    explainer = shap.TreeExplainer(m_full, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test[pos_idx])
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    abs_mean_shap = np.abs(shap_values).mean(axis=0)  # full 40-dim vector

    top10_idx = np.argsort(abs_mean_shap)[::-1][: TOP_K]
    top10_feats = [feature_cols[i] for i in top10_idx]

    metrics = {}
    metrics["top10_features"] = ";".join(top10_feats)
    metrics["shap_vector"] = abs_mean_shap
    metrics["n_test_pos_used"] = int(len(pos_idx))
    return metrics


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="./results/rq1_stability/")
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = args.n_seeds

    all_feature_cols = None
    df_cache = {}
    per_config_vectors = {}
    per_config_top10 = {}
    raw_rows = []

    for sem_cls, tgt_ds in TARGET_ORDER:
        if tgt_ds not in df_cache:
            df_cache[tgt_ds] = load_dataset(data_dir, tgt_ds)
        df = df_cache[tgt_ds]
        if all_feature_cols is None:
            all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        tgt_labels = SEMANTIC_MAPPING[sem_cls][tgt_ds]

        print(f"\n== {sem_cls} / {tgt_ds} ({len(df):,d} rows) ==")
        vecs, tops = {}, {}
        for seed_off in range(n_seeds):
            seed = 42 + seed_off
            split = build_full40(df, all_feature_cols, tgt_labels, seed)
            if split is None:
                continue
            X_train, X_test, y_train, y_test = split
            m = nested_round_test_positives(X_train, X_test, y_train, y_test, all_feature_cols, seed)
            vecs[seed] = m["shap_vector"]
            tops[seed] = m["top10_features"].split(";")
            raw_rows.append(dict(semantic_class=sem_cls, target=tgt_ds, seed=seed,
                                  n_test_pos_used=m["n_test_pos_used"],
                                  top10_features=m["top10_features"],
                                  dominant_feature=tops[seed][0]))
            print(f"  seed {seed}: dominant={tops[seed][0]} (n_test_pos={m['n_test_pos_used']})")
        per_config_vectors[(sem_cls, tgt_ds)] = vecs
        per_config_top10[(sem_cls, tgt_ds)] = tops

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = out_dir / "signature_stability_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n✓ 原始结果: {raw_csv} ({len(raw_df)} 行)")

    summary_rows = []
    for sem_cls, tgt_ds in TARGET_ORDER:
        vecs = per_config_vectors[(sem_cls, tgt_ds)]
        tops = per_config_top10[(sem_cls, tgt_ds)]
        seeds = sorted(vecs.keys())
        if len(seeds) < 2:
            continue
        jaccs, rhos = [], []
        for s1, s2 in itertools.combinations(seeds, 2):
            jaccs.append(jaccard(tops[s1], tops[s2]))
            rho, _ = spearmanr(vecs[s1], vecs[s2])
            rhos.append(rho)
        dominants = [tops[s][0] for s in seeds]
        mode_feat = pd.Series(dominants).value_counts().idxmax()
        mode_freq = pd.Series(dominants).value_counts().max() / len(dominants)

        summary_rows.append(dict(
            semantic_class=sem_cls, target=tgt_ds, n_seeds=len(seeds), n_pairs=len(jaccs),
            jaccard_mean=float(np.mean(jaccs)), jaccard_median=float(np.median(jaccs)),
            jaccard_min=float(np.min(jaccs)), jaccard_max=float(np.max(jaccs)),
            spearman_mean=float(np.mean(rhos)), spearman_median=float(np.median(rhos)),
            spearman_min=float(np.min(rhos)), spearman_max=float(np.max(rhos)),
            dominant_feature_mode=mode_feat, dominant_feature_freq=float(mode_freq),
        ))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "signature_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ 稳定性汇总: {summary_csv}")
    print(summary_df.to_string(index=False))

    overall = dict(
        jaccard_mean=summary_df["jaccard_mean"].mean(),
        jaccard_median=summary_df["jaccard_median"].median(),
        spearman_mean=summary_df["spearman_mean"].mean(),
        spearman_median=summary_df["spearman_median"].median(),
        dominant_feature_freq_mean=summary_df["dominant_feature_freq"].mean(),
        dominant_feature_freq_min=summary_df["dominant_feature_freq"].min(),
    )
    print("\n=== OVERALL (mean across 7 configs, test-positive protocol) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    pd.DataFrame([overall]).to_csv(out_dir / "overall.csv", index=False)


if __name__ == "__main__":
    main()
