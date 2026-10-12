#!/usr/bin/env python3
"""
rq3_transfer.py — RQ3：特征集可移植性 vs 模型可移植性
====================================================================================
条件：
  1. 在目标域重训练：SHAP-10、MI-10（互信息）、Gain-10（XGBoost gain）、SHAP-10-NoPort
     （去掉 L4_SRC_PORT / L4_DST_PORT）、Random-10（每个 seed 取 R 组随机子集的平均，默认 R=20；
     第 0 组另存为 Random-10-single）、Target-10、Full-40。源特征集均在源数据集 seed=42 训练集上
     计算一次并冻结。
  2. 零样本（模型可移植性）：SHAP/MI/Gain/NoPort-10-Direct 与 Random-10-Direct（R_d 组平均），
     在同一目标测试集上评估。
  3. 指标：F1、MCC（0.5 阈值）、AUC-ROC、PR-AUC（average precision）、pAUC@1%FPR（McClish 标准化）。
  4. 两种测试集：std = 1:10 标准测试集；nat = prevalence-matched 测试集（按原始数据集的类别流行率
     与负类构成重新抽取，只用未参与训练的行，正例为 std 测试正例的子集）。
  5. 统计：配对（按 seed）Wilcoxon + Nadeau–Bengio 校正重采样 t 检验 + Holm；效应量 d_z 与
     配对秩二列相关；配置级（n=18）Wilcoxon 与簇级（9 簇）精确符号翻转检验。
  稳健性：--dedup（40 特征精确去重 + 删除标签冲突行）；--variant trainshap | reselect | noscenario。

运行方式（项目根目录）：
  python code/rq3_transfer.py prep                 # ① 准备特征集与数据缓存
  python code/rq3_transfer.py run                  # ② 主实验（可随时中断，重跑自动续跑）
  python code/rq3_transfer.py status               # 查看进度（另开终端）
  python code/rq3_transfer.py aggregate            # ③ 汇总与统计检验
  推荐直接用 ./run.sh rq3 / ./run.sh bg rq3。

依赖：results/rq3_diagnostics/raw_class_counts.json（code/rq3_diagnostics.py）、results/rq1_signatures/*.pkl、data/processed/*.csv
"""
import argparse
import itertools
import multiprocessing as _mp
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ── 线程控制必须在 import numpy/xgboost 之前（防止多进程超额订阅导致假死）──
_T = os.environ.get("A7_THREADS")
if _T:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = _T

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
MAIN_OUT = ROOT / "results" / "rq3_main"
OUT = MAIN_OUT
CACHE = OUT / "cache"
PARTS = OUT / "parts"
LOGS = OUT / "logs"
PROGRESS = OUT / "progress.log"
DEDUP = False
VARIANT = None
VARIANTS = {
    # 名称: (是否用去重数据, 输出目录, 复用的缓存目录)
    "trainshap": (False, "rq3_trainshap", "rq3_main"),
    "reselect": (True, "rq3_dedup_reselect", "rq3_dedup"),
    # 场景特征消融：原始数据；所有源特征集与随机子集都从去掉端口/TTL/TCP 窗口后的 34 个特征中选
    "noscenario": (False, "rq3_noscenario", "rq3_main"),
}
SCENARIO_FEATURES = ["L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]


def set_mode(dedup, variant=None):
    """--dedup 时所有输出改写到 results/rq3_dedup/，与主实验完全隔离。
    --variant trainshap：原始数据；SHAP-10 改为在 seed=42 源端【训练集】正例上计算（与 MI/Gain 同一信息预算）
    --variant reselect ：去重数据；SHAP（训练集）/MI/Gain/NoPort 全部在去重后的源端训练集上重新选择
    两个变体都复用已有数组缓存，输出到各自目录。"""
    global OUT, CACHE, PARTS, LOGS, PROGRESS, DEDUP, VARIANT
    VARIANT = variant or None
    if VARIANT:
        dd, out, cache = VARIANTS[VARIANT]
        DEDUP = dd
        OUT = ROOT / "results" / out
        CACHE = ROOT / "results" / cache / "cache"
    else:
        DEDUP = bool(dedup)
        OUT = ROOT / "results" / ("rq3_dedup" if DEDUP else "rq3_main")
        CACHE = OUT / "cache"
    PARTS, LOGS, PROGRESS = OUT / "parts", OUT / "logs", OUT / "progress.log"


def mode_args():
    """传给子进程的模式参数（spawn/subprocess 不继承全局变量）。"""
    if VARIANT:
        return ["--variant", VARIANT]
    return ["--dedup"] if DEDUP else []

DATASETS = {"UNSW": "NF-UNSW-NB15-v2", "CIC": "NF-CSE-CIC-IDS2018-v2",
            "ToN": "NF-ToN-IoT-v2", "BoT": "NF-BoT-IoT-v2"}
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]
DS_ENV = {"UNSW": "IT", "CIC": "IT", "ToN": "IoT", "BoT": "IoT"}
SEMANTIC_MAPPING = {
    "DoS": {"UNSW": ["DoS"],
            "CIC": ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
            "ToN": ["dos"], "BoT": ["DoS"]},
    "Reconnaissance": {"UNSW": ["Reconnaissance"], "ToN": ["scanning"],
                       "BoT": ["Reconnaissance"]},
}
PORT_FEATURES = ["L4_SRC_PORT", "L4_DST_PORT"]
XGB_PARAMS = dict(objective="binary:logistic", n_estimators=300, max_depth=6,
                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", tree_method="hist", missing=np.nan)
TOP_K, TEST_SIZE, MAX_TRAIN_POS, NEG_RATIO_CAP = 10, 0.20, 30_000, 10
SEED_BASE = 42
NAT_NEG_CAP = 400_000          # 自然流行率测试集负例上限（超过则按比例缩小正例与负例）
NAT_MIN_POS = 50
MI_MAX_ROWS = 60_000           # MI 估计的行数上限（分层子抽样）
TRAIN_FRAC = 1 - TEST_SIZE     # Nadeau–Bengio 校正：n_test/n_train = 0.25

RETRAINED = ["SHAP-10", "MI-10", "Gain-10", "SHAP-10-NoPort"]
DIRECT = [m + "-Direct" for m in RETRAINED]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def plog(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════════════════
def load_cache(ds):
    X = np.load(CACHE / f"{ds}_X.npy", mmap_mode="r")
    codes = np.load(CACHE / f"{ds}_y.npy", mmap_mode="r")
    meta = json.loads((CACHE / f"{ds}_meta.json").read_text())
    return X, codes, meta


def clean_subset(X):
    """非有限值→0；|x|>1e15 → 该子集该列有效值中位数。"""
    X = np.array(X, dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) else 0.0
    return X


def split_indices(codes, pos_codes, seed):
    """RQ3 抽样（rng(seed)）与 80/20 分层划分，返回行号。"""
    from sklearn.model_selection import train_test_split
    pos_mask = np.isin(codes, pos_codes)
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(~pos_mask)
    rng = np.random.default_rng(seed)
    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, MAX_TRAIN_POS, replace=False)
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, int(neg_cap), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx), np.int32), np.zeros(len(neg_idx), np.int32)])
    tr, te, ytr, yte = train_test_split(idx, y, test_size=TEST_SIZE,
                                        random_state=seed, stratify=y)
    return tr, te, ytr.astype(np.int32), yte.astype(np.int32)


def fit_xgb(X, y, seed, threads):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    p["scale_pos_weight"] = int((y == 0).sum()) / max(int((y == 1).sum()), 1)
    p["random_state"] = seed
    p["n_jobs"] = threads
    m = xgb.XGBClassifier(**p)
    m.fit(X, np.asarray(y, dtype=np.int32), verbose=False)
    return m


def metrics(y, proba):
    from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
                                 average_precision_score)
    y = np.asarray(y, dtype=np.int32)
    pred = (proba >= 0.5).astype(np.int32)
    out = {"f1": f1_score(y, pred, zero_division=0),
           "mcc": matthews_corrcoef(y, pred)}
    try:
        out["auc"] = roc_auc_score(y, proba)
        out["pr_auc"] = average_precision_score(y, proba)
        out["pauc01"] = roc_auc_score(y, proba, max_fpr=0.01)
    except ValueError:
        out["auc"] = out["pr_auc"] = out["pauc01"] = np.nan
    return {k: round(float(v), 6) for k, v in out.items()}


def rank_vector(vec, feature_cols, exclude=()):
    order = [feature_cols[i] for i in np.argsort(np.asarray(vec))[::-1]]
    return [f for f in order if f not in exclude][:TOP_K]


# ══════════════════════════════════════════════════════════════════════════
# ① prep：缓存数组 + 冻结源特征集
# ══════════════════════════════════════════════════════════════════════════
def prep_cache(ds_short, dedup=False):
    import pickle
    set_mode(dedup)          # 子进程（spawn）不继承主进程的全局设置，必须显式传入
    ds = DATASETS[ds_short]
    out_x = CACHE / f"{ds_short}_X.npy"
    if out_x.exists() and (CACHE / f"{ds_short}_meta.json").exists():
        return f"{ds_short}: 缓存已存在，跳过"
    sigs = pickle.load(open(ROOT / "results" / "rq1_signatures" / f"{ds}_rq1_shap_values.pkl", "rb"))
    feature_cols = list(sigs.values())[0]["feature_cols"]
    df = pd.read_csv(ROOT / "data" / "processed" / f"{ds}_processed.csv", low_memory=False)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ds_short}: processed 缺少列 {missing}")
    dedup_info = None
    if DEDUP:
        # 精确去重：40 特征完全相同的行
        #   ① 同一特征向量对应多个标签（标签冲突，无法判定真值）→ 全部删除
        #   ② 同一特征向量、同一标签的重复行 → 只保留第一条
        h = pd.util.hash_pandas_object(df[feature_cols], index=False)
        att = df["Attack"].astype(str)
        before = att.value_counts().to_dict()
        n_lab = pd.DataFrame({"h": h.values, "a": att.values}).groupby("h")["a"].nunique()
        conflict = h.map(n_lab).to_numpy() > 1
        keep = ~conflict & ~pd.Series(h.values).duplicated(keep="first").to_numpy()
        df = df[keep].reset_index(drop=True)
        after = df["Attack"].astype(str).value_counts().to_dict()
        dedup_info = {"n_before": int(len(keep)), "n_after": int(keep.sum()),
                      "n_conflict_rows_dropped": int(conflict.sum()),
                      "per_class": {k: [int(before[k]), int(after.get(k, 0))] for k in before}}
    X = df[feature_cols].to_numpy(dtype=np.float64)
    labels = sorted(df["Attack"].astype(str).unique())
    lab2code = {l: i for i, l in enumerate(labels)}
    codes = df["Attack"].astype(str).map(lab2code).to_numpy(dtype=np.int16)
    tmp = CACHE / f"{ds_short}_X.tmp.npy"
    np.save(tmp, X)
    np.save(CACHE / f"{ds_short}_y.npy", codes)
    (CACHE / f"{ds_short}_meta.json").write_text(json.dumps(
        {"feature_cols": feature_cols, "labels": labels,
         "counts": {l: int((codes == c).sum()) for l, c in lab2code.items()},
         "dedup": dedup_info},
        ensure_ascii=False, indent=1))
    os.replace(tmp, out_x)
    if dedup_info:
        return (f"{ds_short}: 去重缓存完成 {dedup_info['n_before']:,} → {dedup_info['n_after']:,} 行"
                f"（其中标签冲突删除 {dedup_info['n_conflict_rows_dropped']:,} 行）")
    return f"{ds_short}: 缓存完成 {X.shape}"


def prep_variant_importance(ds_short, label, threads, dedup=False, variant=None):
    """单个原始标签（变体）在 seed=42 训练集上的 MI、XGB gain 与训练集正例 TreeSHAP。"""
    from sklearn.feature_selection import mutual_info_classif
    import xgboost as xgb
    set_mode(dedup, variant)          # spawn 子进程必须显式设置模式（决定读哪个缓存）
    X, codes, meta = load_cache(ds_short)
    fc = meta["feature_cols"]
    code = meta["labels"].index(label)
    tr, te, ytr, yte = split_indices(codes, [code], SEED_BASE)
    Xtr = clean_subset(X[np.sort(tr)])
    ytr = ytr[np.argsort(tr)]
    # MI：分层子抽样（MI 计算 O(n log n)，6 万行足够稳定）
    rng = np.random.default_rng(SEED_BASE)
    if len(ytr) > MI_MAX_ROWS:
        pos = np.flatnonzero(ytr == 1)
        neg = np.flatnonzero(ytr == 0)
        n_pos = min(len(pos), int(MI_MAX_ROWS * len(pos) / len(ytr)) + 1)
        sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, MI_MAX_ROWS - n_pos, replace=False)])
    else:
        sel = np.arange(len(ytr))
    mi = mutual_info_classif(Xtr[sel], ytr[sel], random_state=SEED_BASE, n_jobs=threads)
    m = fit_xgb(Xtr, ytr, SEED_BASE, threads)
    score = m.get_booster().get_score(importance_type="gain")
    gain = np.array([score.get(f"f{i}", 0.0) for i in range(len(fc))])
    # 训练集正例上的 TreeSHAP（pred_contribs = tree_path_dependent；与 RQ1 一样最多 5,000 个正例）
    pos = np.flatnonzero(ytr == 1)
    if len(pos) > 5_000:
        pos = np.sort(np.random.default_rng(SEED_BASE).choice(pos, 5_000, replace=False))
    contrib = m.get_booster().predict(xgb.DMatrix(Xtr[pos]), pred_contribs=True)[:, :-1]
    shap_tr = np.abs(contrib).mean(axis=0)
    return (ds_short, label, mi.tolist(), gain.tolist(), int((ytr == 1).sum()),
            shap_tr.tolist(), int(len(pos)))


def cmd_prep(a):
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed
    for d in (OUT, CACHE, PARTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    cj = ROOT / "results" / "rq3_diagnostics" / "raw_class_counts.json"
    if not cj.exists():
        sys.exit("✗ 缺少 results/rq3_diagnostics/raw_class_counts.json —— 请先运行 code/rq3_diagnostics.py")
    if VARIANT:
        miss = [d for d in DS_ORDER if not (CACHE / f"{d}_X.npy").exists()]
        if miss:
            base = VARIANTS[VARIANT][2]
            sys.exit(f"✗ 变体 {VARIANT} 复用 results/{base}/cache，但缺少 {miss}；"
                     f"请先完成{'去重实验' if DEDUP else '主实验'}的 prep")
        if not (MAIN_OUT / "feature_sets.json").exists():
            sys.exit("✗ 缺少主实验 feature_sets.json（用于对照），请先完成主实验 prep")
    else:
      plog("prep ① 构建数组缓存（4 个数据集并行）")
      with ProcessPoolExecutor(max_workers=min(4, a.workers), mp_context=_mp.get_context("spawn")) as ex:
        for f in as_completed([ex.submit(prep_cache, d, DEDUP) for d in DS_ORDER]):
            plog("   " + f.result())

    if DEDUP and not VARIANT:
        src = MAIN_OUT / "feature_sets.json"
        if not src.exists():
            sys.exit("✗ 去重模式沿用主实验冻结的特征集，请先完成主实验的 prep")
        import shutil
        missing = [d for d in DS_ORDER if not (CACHE / f"{d}_meta.json").exists()]
        if missing:
            sys.exit(f"✗ 去重缓存未生成：{missing}")
        rows = []
        for d in DS_ORDER:
            info = json.loads((CACHE / f"{d}_meta.json").read_text())["dedup"]
            for cls, (b, a_) in info["per_class"].items():
                rows.append({"dataset": d, "class": cls, "rows_before": b, "rows_after": a_,
                             "kept_share": round(a_ / b, 4) if b else np.nan})
        pd.DataFrame(rows).to_csv(OUT / "dedup_report.csv", index=False)
        shutil.copy(MAIN_OUT / "feature_sets.csv", OUT / "feature_sets.csv")
        shutil.copy(src, OUT / "feature_sets.json")      # 最后写入：作为“prep 已完成”的标志
        plog("prep（去重模式）完成：特征集沿用主实验（SHAP/MI/Gain/NoPort 冻结不变），"
             "去重明细见 results/rq3_dedup/dedup_report.csv")
        return

    plog(f"prep ② 计算 MI / Gain / 训练集 SHAP（seed=42 训练集，按变体并行；模式={VARIANT or ('dedup' if DEDUP else 'main')}）")
    jobs = {(d, l) for mp in SEMANTIC_MAPPING.values() for d, ls in mp.items() for l in ls}
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=_mp.get_context("spawn")) as ex:
        futs = [ex.submit(prep_variant_importance, d, l, a.threads, DEDUP, VARIANT) for d, l in sorted(jobs)]
        for f in as_completed(futs):
            d, l, mi, gain, npos, shap_tr, nshap = f.result()
            res[(d, l)] = (np.array(mi), np.array(gain), npos, np.array(shap_tr), nshap)
            plog(f"   ✓ {d}/{l}  (训练正例 {npos:,}；SHAP 用 {nshap:,})")
    main_fs = json.loads((MAIN_OUT / "feature_sets.json").read_text()) if VARIANT else None

    fsets, rows = {}, []
    EXCL = SCENARIO_FEATURES if VARIANT == "noscenario" else []
    for sem, mp in SEMANTIC_MAPPING.items():
        for d, ls in mp.items():
            sigs = pickle.load(open(ROOT / "results" / "rq1_signatures" / f"{DATASETS[d]}_rq1_shap_values.pkl", "rb"))
            fc = json.loads((CACHE / f"{d}_meta.json").read_text())["feature_cols"]
            m = {l: sigs[l] for l in ls if l in sigs}
            w = sum(s.get("n_pos_shap", 1) for s in m.values())
            shap_vec = sum(s["abs_mean_shap"] * s.get("n_pos_shap", 1) for s in m.values()) / w
            if VARIANT in ("trainshap", "reselect"):   # 变体：SHAP 改用 seed=42 源端训练集正例（与 MI/Gain 相同的数据划分），同样按样本数加权合成
                ws = sum(res[(d, l)][4] for l in ls)
                shap_vec = sum(res[(d, l)][3] * res[(d, l)][4] for l in ls) / ws
            # MI / Gain：变体内先归一化（和为1）再按训练正例数加权，避免尺度差异
            tot = sum(res[(d, l)][2] for l in ls)
            mi_vec = sum(res[(d, l)][0] / max(res[(d, l)][0].sum(), 1e-12) * res[(d, l)][2] for l in ls) / tot
            gain_vec = sum(res[(d, l)][1] / max(res[(d, l)][1].sum(), 1e-12) * res[(d, l)][2] for l in ls) / tot
            fsets.setdefault(sem, {})[d] = {
                "SHAP-10": rank_vector(shap_vec, fc, exclude=EXCL),
                "MI-10": rank_vector(mi_vec, fc, exclude=EXCL),
                "Gain-10": rank_vector(gain_vec, fc, exclude=EXCL),
                "SHAP-10-NoPort": rank_vector(shap_vec, fc, exclude=list(EXCL) + PORT_FEATURES),
            }
            if VARIANT == "trainshap":   # 原始数据：MI/Gain 与主实验完全相同（直接沿用，保证只有 SHAP 一处变化）
                for k in ("MI-10", "Gain-10"):
                    fsets[sem][d][k] = main_fs[sem][d][k]
            jac = lambda x, y: len(set(x) & set(y)) / len(set(x) | set(y))
            for meth, feats in fsets[sem][d].items():
                r = {"class": sem, "dataset": d, "method": meth, "top10": ";".join(feats),
                     "jaccard_vs_SHAP": jac(feats, fsets[sem][d]["SHAP-10"])}
                if VARIANT:   # 与主实验冻结特征集（原 SHAP=测试集正例；原 MI/Gain=未去重训练集）的重合度
                    r["main_top10"] = ";".join(main_fs[sem][d][meth])
                    r["jaccard_vs_main_same_method"] = jac(feats, main_fs[sem][d][meth])
                rows.append(r)
    (OUT / "feature_sets.json").write_text(json.dumps(fsets, indent=1, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(OUT / "feature_sets.csv", index=False)
    plog(f"prep 完成 → {OUT.relative_to(ROOT)}/feature_sets.csv（各方法 Top-10 及与 SHAP 的 Jaccard）")


# ══════════════════════════════════════════════════════════════════════════
# ② worker：一个 (类别, seed) 任务
# ══════════════════════════════════════════════════════════════════════════
class Heartbeat:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        self.f.write(f"[{now()}] {msg}\n")
        self.f.flush()


def random_subsets(feature_cols, seed, R):
    """第 0 组用 rng(seed+10000)；其余组用独立种子。"""
    subs = [list(np.random.default_rng(seed + 10000).choice(feature_cols, size=TOP_K, replace=False))]
    rng = np.random.default_rng(seed + 30000)
    while len(subs) < R:
        subs.append(list(rng.choice(feature_cols, size=TOP_K, replace=False)))
    return subs


def natural_test(codes, meta, raw_counts, pos_codes, train_idx, test_pos_idx, seed):
    """从未参与训练的行中，按原始数据集流行率与负类构成抽取测试集。"""
    labels = meta["labels"]
    present = {l: raw_counts.get(l, 0) for l in labels}
    pos_labels = [labels[c] for c in pos_codes]
    raw_pos = sum(present[l] for l in pos_labels)
    raw_neg = {l: v for l, v in present.items() if l not in pos_labels and v > 0}
    raw_neg_tot = sum(raw_neg.values())
    if raw_pos == 0 or raw_neg_tot == 0:
        return None
    pi = raw_pos / (raw_pos + raw_neg_tot)
    used = np.zeros(len(codes), dtype=bool)
    used[train_idx] = True
    n_pos = len(test_pos_idx)
    want_neg_tot = n_pos * (1 - pi) / pi
    quota = {l: want_neg_tot * v / raw_neg_tot for l, v in raw_neg.items()}
    avail = {}
    for l in quota:
        c = labels.index(l)
        avail[l] = np.flatnonzero((codes == c) & ~used)
    s = min(1.0, NAT_NEG_CAP / max(want_neg_tot, 1))
    for l, q in quota.items():
        if q * s > len(avail[l]):
            s = min(s, len(avail[l]) / q)
    n_pos_s = int(np.floor(n_pos * s))
    if n_pos_s < NAT_MIN_POS:
        return None
    rng = np.random.default_rng(seed + 20000)
    pos_sel = np.sort(rng.choice(test_pos_idx, n_pos_s, replace=False))
    negs = []
    for l, q in quota.items():
        k = int(round(q * s))
        if k > 0:
            negs.append(rng.choice(avail[l], min(k, len(avail[l])), replace=False))
    neg_sel = np.sort(np.concatenate(negs)) if negs else np.array([], dtype=np.int64)
    idx = np.concatenate([pos_sel, neg_sel])
    y = np.concatenate([np.ones(len(pos_sel), np.int32), np.zeros(len(neg_sel), np.int32)])
    return idx, y, pi


def cmd_worker(a):
    sem, seed, threads = a.cls, a.seed, a.threads
    tag = f"{sem}__seed{seed}"
    hb = Heartbeat(LOGS / f"{tag}.log")
    hb(f"开始 {tag} threads={threads} R={a.n_random} R_direct={a.n_random_direct}")
    if os.environ.get("A7_DEBUG_HANG"):          # 仅用于测试调度器的防卡死机制
        time.sleep(10 ** 6)
    t0 = time.time()
    fsets = json.loads((OUT / "feature_sets.json").read_text())[sem]
    raw_counts = json.loads((ROOT / "results" / "rq3_diagnostics" / "raw_class_counts.json").read_text())
    mp = SEMANTIC_MAPPING[sem]
    dss = [d for d in DS_ORDER if d in mp]
    data = {d: load_cache(d) for d in dss}
    fc = data[dss[0]][2]["feature_cols"]
    col = {f: i for i, f in enumerate(fc)}
    pool = [f for f in fc if f not in SCENARIO_FEATURES] if VARIANT == "noscenario" else fc
    subsets = random_subsets(pool, seed, a.n_random)
    rows, rand_rows = [], []

    def add(target, source, cond, testset, met, extra=None):
        r = dict(semantic_class=sem, target=target, source=source, condition=cond,
                 env_same=(DS_ENV[source] == DS_ENV[target]) if source in DS_ENV else np.nan,
                 seed=seed, testset=testset, **met)
        if extra:
            r.update(extra)
        rows.append(r)

    # ── 每个 target：构建 std / nat 测试集 ──
    tsets = {}
    for tgt in dss:
        X, codes, meta = data[tgt]
        pos_codes = [meta["labels"].index(l) for l in mp[tgt] if l in meta["labels"]]
        tr, te, ytr, yte = split_indices(codes, pos_codes, seed)
        # 在“训练+测试”同一批抽样行上计算清洗用中位数，再切分（保持行顺序）
        Xall = clean_subset(X[np.concatenate([tr, te])])
        Xtr, Xte = Xall[:len(tr)], Xall[len(tr):]
        del Xall
        nat = natural_test(codes, meta, raw_counts.get(tgt, {}), pos_codes, tr, te[yte == 1], seed)
        if nat is not None:
            Xn, yn, pi = clean_subset(X[nat[0]]), nat[1], nat[2]
        else:
            Xn, yn, pi = None, None, np.nan
        tsets[tgt] = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xn=Xn, yn=yn, pi=pi)
        hb(f"{tgt}: train={len(ytr):,} std_test={len(yte):,} nat_test="
           f"{0 if yn is None else len(yn):,} (π={pi:.4f})")

    def evaluate(model, cols, tgt):
        T = tsets[tgt]
        out = {"std": metrics(T["yte"], model.predict_proba(T["Xte"][:, cols])[:, 1])}
        if T["Xn"] is not None:
            m = metrics(T["yn"], model.predict_proba(T["Xn"][:, cols])[:, 1])
            m["prevalence"] = round(float(T["yn"].mean()), 6)
            out["nat"] = m
        return out

    # ── 目标端重训练条件 ──
    for tgt in dss:
        T = tsets[tgt]
        all_cols = list(range(len(fc)))
        m40 = fit_xgb(T["Xtr"], T["ytr"], seed, threads)
        for ts, met in evaluate(m40, all_cols, tgt).items():
            add(tgt, "—", "Full-40", ts, met)
        if VARIANT == "noscenario":            # 参照：去掉场景特征后的全部 34 个特征
            c34 = [col[f] for f in pool]
            m34 = fit_xgb(T["Xtr"][:, c34], T["ytr"], seed, threads)
            for ts, met in evaluate(m34, c34, tgt).items():
                add(tgt, "—", "Full-34", ts, met)
        # 嵌套 Target-10：Full-40 模型在训练正例上的 TreeSHAP（pred_contribs，等价 tree_path_dependent）
        import xgboost as xgb
        pos_tr = T["Xtr"][T["ytr"] == 1]
        contrib = m40.get_booster().predict(xgb.DMatrix(pos_tr), pred_contribs=True)[:, :-1]
        t10 = list(np.argsort(np.abs(contrib).mean(axis=0))[::-1][:TOP_K])
        m = fit_xgb(T["Xtr"][:, t10], T["ytr"], seed, threads)
        for ts, met in evaluate(m, t10, tgt).items():
            add(tgt, tgt, "Target-10", ts, met, {"features": ";".join(fc[i] for i in t10)})
        hb(f"{tgt}: Full-40 / Target-10 完成")

        # 随机子集（R 组）
        acc = {}
        for ri, sub in enumerate(subsets):
            if ri % 5 == 0:
                hb(f"{tgt}: Random 子集 {ri}/{len(subsets)}")
            cols = [col[f] for f in sub]
            m = fit_xgb(T["Xtr"][:, cols], T["ytr"], seed, threads)
            for ts, met in evaluate(m, cols, tgt).items():
                acc.setdefault(ts, []).append(met)
                rand_rows.append(dict(semantic_class=sem, target=tgt, seed=seed, testset=ts,
                                      subset=ri, features=";".join(sub), **met))
                if ri == 0:
                    add(tgt, "Random", "Random-10-single", ts, met)
        for ts, lst in acc.items():
            dfm = pd.DataFrame(lst)
            mean = dfm.mean().round(6).to_dict()
            add(tgt, "Random", "Random-10", ts, mean,
                {"rand_sd_f1": round(float(dfm["f1"].std(ddof=1)), 6), "rand_n": len(lst)})
        hb(f"{tgt}: Random-10 × {len(subsets)} 完成")

        # 源特征集（目标端重训练）
        for src in dss:
            if src == tgt:
                continue
            done = {}
            hb(f"{tgt}: 源={src} 特征集重训练")
            for meth in RETRAINED:
                feats = fsets[src][meth]
                key = tuple(feats)
                cols = [col[f] for f in feats]
                if key in done:                       # 特征集相同则复用结果（如 NoPort 与 SHAP 相同）
                    res = done[key]
                else:
                    m = fit_xgb(T["Xtr"][:, cols], T["ytr"], seed, threads)
                    res = done[key] = evaluate(m, cols, tgt)
                for ts, met in res.items():
                    add(tgt, src, meth, ts, met)
        hb(f"{tgt}: 源特征集（重训练）完成  累计 {time.time()-t0:.0f}s")

    # ── 零样本：源端训练，直接在 target 的 std / nat 测试集上评估 ──
    for src in dss:
        S = tsets[src]                         # 源端训练集 = 该数据集同一 seed 的训练划分
        models = {}
        for meth in RETRAINED:
            feats = fsets[src][meth]
            key = tuple(feats)
            if key not in models:
                cols = [col[f] for f in feats]
                models[key] = (fit_xgb(S["Xtr"][:, cols], S["ytr"], seed, threads), cols)
        hb(f"零样本 源={src}：训练源端模型")
        rmodels = []
        for sub in subsets[:a.n_random_direct]:
            cols = [col[f] for f in sub]
            rmodels.append((fit_xgb(S["Xtr"][:, cols], S["ytr"], seed, threads), cols))
        for tgt in dss:
            if tgt == src:
                continue
            for meth in RETRAINED:
                m, cols = models[tuple(fsets[src][meth])]
                for ts, met in evaluate(m, cols, tgt).items():
                    add(tgt, src, meth + "-Direct", ts, met)
            acc = {}
            for m, cols in rmodels:
                for ts, met in evaluate(m, cols, tgt).items():
                    acc.setdefault(ts, []).append(met)
            for ts, lst in acc.items():
                add(tgt, src, "Random-10-Direct", ts, pd.DataFrame(lst).mean().round(6).to_dict(),
                    {"rand_n": len(lst)})
        hb(f"零样本 源={src} 完成  累计 {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    tmp = PARTS / f"{tag}.csv.tmp"
    out.to_csv(tmp, index=False)
    pd.DataFrame(rand_rows).to_csv(PARTS / f"{tag}_random.csv", index=False)
    os.replace(tmp, PARTS / f"{tag}.csv")       # 原子写入：只有完整结果才算完成
    hb(f"完成 {tag}，{len(out)} 行，用时 {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# ③ run：调度器（子进程 + 超时 + 心跳检测 + 断点续跑）
# ══════════════════════════════════════════════════════════════════════════
def all_tasks(seeds, classes):
    return [(c, s) for s in seeds for c in classes]


def cmd_run(a):
    for d in (PARTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    if not (OUT / "feature_sets.json").exists():
        sys.exit("✗ 请先运行 prep")
    miss = [d for d in DS_ORDER if not (CACHE / f"{d}_X.npy").exists()]
    if miss:
        sys.exit(f"✗ 缓存缺失 {miss}，请重新运行 prep {' '.join(mode_args())}")
    seeds = list(range(SEED_BASE, SEED_BASE + a.n_seeds))
    tasks = [t for t in all_tasks(seeds, a.classes)
             if not (PARTS / f"{t[0]}__seed{t[1]}.csv").exists()]
    total = len(all_tasks(seeds, a.classes))
    plog(f"run 开始：总任务 {total}，待完成 {len(tasks)}，并行 {a.workers} × 线程 {a.threads}，"
         f"单任务超时 {a.timeout_min} 分钟，心跳停滞判定 {a.stall_min} 分钟")
    (OUT / "run_config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    env = dict(os.environ, A7_THREADS=str(a.threads), PYTHONUNBUFFERED="1")
    running, attempts, failed, durations = {}, {}, [], []
    queue = list(tasks)
    last_report = 0

    def launch(t):
        cls, seed = t
        tag = f"{cls}__seed{seed}"
        errf = open(LOGS / f"{tag}.stderr", "a")
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker",
                              "--cls", cls, "--seed", str(seed), "--threads", str(a.threads),
                              "--n-random", str(a.n_random),
                              "--n-random-direct", str(a.n_random_direct)]
                             + mode_args(),
                             env=env, stdout=errf, stderr=errf, start_new_session=True)
        running[t] = (p, time.time(), errf)
        attempts[t] = attempts.get(t, 0) + 1

    def kill(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            p.kill()

    try:
        while queue or running:
            while queue and len(running) < a.workers:
                launch(queue.pop(0))
            time.sleep(5)
            for t, (p, st, errf) in list(running.items()):
                tag = f"{t[0]}__seed{t[1]}"
                lg = LOGS / f"{tag}.log"
                rc = p.poll()
                last_beat = max(st, lg.stat().st_mtime) if lg.exists() else st
                stalled = (time.time() - last_beat) > a.stall_min * 60
                timeout = (time.time() - st) > a.timeout_min * 60
                if rc is None and not (stalled or timeout):
                    continue
                if rc is None:
                    kill(p)
                    reason = "超时" if timeout else "心跳停滞"
                else:
                    reason = f"退出码 {rc}"
                errf.close()
                del running[t]
                ok = (PARTS / f"{tag}.csv").exists()
                if ok:
                    durations.append(time.time() - st)
                    plog(f"✓ {tag} 完成（{(time.time()-st)/60:.1f} 分钟）")
                elif attempts[t] < 2:
                    plog(f"⚠ {tag} 失败（{reason}），重试一次；详情见 logs/{tag}.stderr")
                    queue.append(t)
                else:
                    failed.append(tag)
                    plog(f"✗ {tag} 两次失败（{reason}），跳过；详情见 logs/{tag}.stderr")
            if time.time() - last_report > 60:
                last_report = time.time()
                done = len(list(PARTS.glob("*__seed*[0-9].csv")))
                eta = ""
                if durations:
                    left = total - done
                    eta = f"，预计剩余 {np.mean(durations) * left / a.workers / 60:.0f} 分钟"
                plog(f"进度 {done}/{total}，运行中 {len(running)}，失败 {len(failed)}{eta}")
    except KeyboardInterrupt:
        plog("收到中断，正在终止子进程……（已完成的任务会保留，重跑 run 自动续跑）")
        for p, _, _ in running.values():
            kill(p)
        sys.exit(1)
    plog(f"run 结束：失败 {len(failed)} 个 {failed if failed else ''}")
    if not failed:
        plog("下一步：python code/rq3_transfer.py aggregate")


def cmd_status(a):
    cfg = OUT / "run_config.json"
    n_seeds = json.loads(cfg.read_text())["n_seeds"] if cfg.exists() else 30
    classes = json.loads(cfg.read_text())["classes"] if cfg.exists() else list(SEMANTIC_MAPPING)
    total = len(all_tasks(range(n_seeds), classes))
    done = sorted(PARTS.glob("*__seed*[0-9].csv")) if PARTS.exists() else []
    print(f"已完成 {len(done)}/{total}")
    if LOGS.exists():
        active = []
        for lg in LOGS.glob("*.log"):
            tag = lg.stem
            if (PARTS / f"{tag}.csv").exists():
                continue
            age = time.time() - lg.stat().st_mtime
            last = lg.read_text(encoding="utf-8").strip().splitlines()[-1:] or [""]
            active.append((age, tag, last[0]))
        for age, tag, last in sorted(active):
            flag = "  ⚠ 超过10分钟无心跳" if age > 600 else ""
            print(f"  {tag:<28} {age/60:5.1f} 分钟前：{last}{flag}")
    if PROGRESS.exists():
        print("\n最近进度：")
        print("\n".join(PROGRESS.read_text(encoding="utf-8").strip().splitlines()[-5:]))


# ══════════════════════════════════════════════════════════════════════════
# ④ aggregate：统计
# ══════════════════════════════════════════════════════════════════════════
def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(np.where(ok, p, np.inf))
    m, run = ok.sum(), 0.0
    for k, i in enumerate(idx[:m]):
        run = max(run, (m - k) * p[i])
        out[i] = min(1.0, run)
    return out


def paired_stats(a_, b_, alternative):
    from scipy.stats import wilcoxon, t as tdist
    D = np.asarray(a_) - np.asarray(b_)
    D = D[~np.isnan(D)]
    J = len(D)
    res = dict(n=J, delta=np.nan, d_z=np.nan, rank_biserial=np.nan, p_wilcoxon=np.nan,
               p_nb=np.nan, nb_ci_lo=np.nan, nb_ci_hi=np.nan)
    if J < 3:
        return res
    res["delta"] = D.mean()
    sd = D.std(ddof=1)
    res["d_z"] = D.mean() / sd if sd > 0 else np.nan
    nz = D[D != 0]
    if len(nz):
        rk = pd.Series(np.abs(nz)).rank().to_numpy()
        res["rank_biserial"] = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
        res["p_wilcoxon"] = wilcoxon(D, alternative=alternative).pvalue if len(nz) >= 1 else np.nan
    if sd > 0:
        se = np.sqrt((1 / J + TEST_SIZE / TRAIN_FRAC) * sd ** 2)
        t = D.mean() / se
        res["p_nb"] = (1 - tdist.cdf(t, J - 1)) if alternative == "greater" else \
            2 * (1 - tdist.cdf(abs(t), J - 1))
        q = tdist.ppf(0.975, J - 1)
        res["nb_ci_lo"], res["nb_ci_hi"] = D.mean() - q * se, D.mean() + q * se
    else:
        res["p_nb"] = 1.0 if D.mean() <= 0 else 0.0
    return res


COMPARISONS = [  # (A, B, alternative, 家族)
    ("SHAP-10", "Random-10", "greater", "retrained"),
    ("MI-10", "Random-10", "greater", "retrained"),
    ("Gain-10", "Random-10", "greater", "retrained"),
    ("SHAP-10-NoPort", "Random-10", "greater", "retrained"),
    ("SHAP-10", "MI-10", "two-sided", "retrained"),
    ("SHAP-10", "Gain-10", "two-sided", "retrained"),
    ("SHAP-10", "SHAP-10-NoPort", "two-sided", "retrained"),
    ("SHAP-10", "Random-10-single", "greater", "retrained"),
    ("SHAP-10-Direct", "Random-10-Direct", "greater", "direct"),
    ("SHAP-10-Direct", "MI-10-Direct", "two-sided", "direct"),
    ("SHAP-10-Direct", "Gain-10-Direct", "two-sided", "direct"),
    ("SHAP-10-NoPort-Direct", "SHAP-10-Direct", "two-sided", "direct"),
]
METRICS = ["f1", "mcc", "auc", "pr_auc", "pauc01"]


def cmd_aggregate(a):
    from scipy.stats import wilcoxon
    files = sorted(PARTS.glob("*__seed*[0-9].csv"))
    if not files:
        sys.exit("✗ 没有结果文件")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw.to_csv(OUT / "raw.csv", index=False)
    plog(f"aggregate：读入 {len(files)} 个任务，{len(raw):,} 行")

    summ = (raw.groupby(["semantic_class", "target", "source", "condition", "testset"])[METRICS]
            .agg(["mean", "std"]).round(4))
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ.reset_index().to_csv(OUT / "config_summary.csv", index=False)

    rows = []
    cfgs = raw[raw.condition == "SHAP-10"][["semantic_class", "target", "source"]].drop_duplicates()
    for A, B, alt, fam in COMPARISONS:
        for ts in ("std", "nat"):
            for met in METRICS:
                block = []
                for _, c in cfgs.iterrows():
                    sub = raw[(raw.semantic_class == c.semantic_class) & (raw.target == c.target)
                              & (raw.testset == ts)]
                    srcA = "Random" if A in ("Random-10", "Random-10-single") else c.source
                    srcB = "Random" if B in ("Random-10", "Random-10-single") else c.source
                    sa = sub[(sub.condition == A) & (sub.source == srcA)].set_index("seed")[met]
                    sb = sub[(sub.condition == B) & (sub.source == srcB)].set_index("seed")[met]
                    j = sa.index.intersection(sb.index)
                    if len(j) < 3:
                        continue
                    st = paired_stats(sa.loc[j].values, sb.loc[j].values, alt)
                    block.append(dict(comparison=f"{A} vs {B}", family=fam, testset=ts, metric=met,
                                      semantic_class=c.semantic_class,
                                      pair=f"{c.source}→{c.target}",
                                      env_same=DS_ENV[c.source] == DS_ENV[c.target],
                                      mean_A=sa.loc[j].mean(), mean_B=sb.loc[j].mean(), **st))
                if not block:
                    continue
                pw = holm([b["p_wilcoxon"] for b in block])
                pn = holm([b["p_nb"] for b in block])
                for b, x, y in zip(block, pw, pn):
                    b["p_wilcoxon_holm"], b["p_nb_holm"] = x, y
                rows += block
    per = pd.DataFrame(rows)
    n_seeds = raw["seed"].nunique()
    if per.empty:
        plog(f"⚠ 只有 {n_seeds} 个 seed（配对检验至少需要 3 个），跳过统计检验，只输出一致性核对和条件均值")
    else:
        per.round(6).to_csv(OUT / "per_config_tests.csv", index=False)

    # 配置级 + 簇级
    lev = []
    for (comp, ts, met), g in (per.groupby(["comparison", "testset", "metric"]) if not per.empty else []):
        D = g["delta"].to_numpy()
        alt = "greater" if "Random" in comp.split(" vs ")[1] else "two-sided"
        r = dict(comparison=comp, testset=ts, metric=met, n_configs=len(D),
                 n_positive=int((D > 0).sum()), mean_delta=D.mean(),
                 n_sig_wilcoxon_holm=int((g["p_wilcoxon_holm"] < 0.05).sum()),
                 n_sig_nb_holm=int((g["p_nb_holm"] < 0.05).sum()))
        r["p_config_wilcoxon"] = wilcoxon(D, alternative=alt).pvalue if np.any(D != 0) else np.nan
        g = g.assign(cl=g.semantic_class + "|" + g.pair.map(lambda s: "-".join(sorted(s.split("→")))))
        cm = g.groupby("cl")["delta"].mean().to_numpy()
        obs, cnt, N = cm.mean(), 0, 0
        for sg in itertools.product([1, -1], repeat=len(cm)):
            N += 1
            v = np.mean(np.array(sg) * cm)
            cnt += (v >= obs - 1e-12) if alt == "greater" else (abs(v) >= abs(obs) - 1e-12)
        r["n_clusters"], r["p_cluster_signflip"] = len(cm), cnt / N
        lev.append(r)
    lv = pd.DataFrame(lev)
    if not lv.empty:
        lv.round(6).to_csv(OUT / "config_level_tests.csv", index=False)

    # 稳健性实验：与主实验（相同 seed）对比
    chk = []
    if DEDUP or VARIANT:
        lab = {"trainshap": "训练集SHAP", "reselect": "去重+重选", "noscenario": "去场景特征"}.get(VARIANT, "去重")
        mr = MAIN_OUT / "raw.csv"
        if mr.exists():
            main_raw = pd.read_csv(mr)
            main_raw = main_raw[main_raw.seed.isin(raw.seed.unique())]
            cmp_ = []
            for (sem, cond), g in raw[raw.testset == "std"].groupby(["semantic_class", "condition"]):
                m = main_raw[(main_raw.semantic_class == sem) & (main_raw.condition == cond)
                             & (main_raw.testset == "std")]
                if len(m):
                    cmp_.append(f"  {sem:<15}{cond:<24} F1 主实验 {m.f1.mean():.4f} → {lab} {g.f1.mean():.4f}"
                                f"（Δ {g.f1.mean() - m.f1.mean():+.4f}）  AUC {m.auc.mean():.4f} → {g.auc.mean():.4f}")
            chk = ["", "与主实验（未去重、相同 seed）对比，std 测试集："] + cmp_
            fs = OUT / "feature_sets.csv"
            if VARIANT and fs.exists():
                f_ = pd.read_csv(fs)
                chk += ["", "本变体特征集与主实验同方法特征集的 Top-10 Jaccard（1 = 完全相同）："]
                for _, r in f_.iterrows():
                    chk.append(f"  {r['class']:<15}{r.dataset:<6}{r.method:<16} {r.jaccard_vs_main_same_method:.2f}")
        else:
            chk = ["", "（未找到主实验 raw.csv：请先对主实验运行 aggregate，再对去重实验运行 aggregate）"]

    # 文字摘要
    title = {"trainshap": "（变体 trainshap：SHAP-10 改为源端训练集正例计算，MI/Gain 不变）",
             "reselect": "（变体 reselect：去重数据 + 在去重训练集上重新选择 SHAP/MI/Gain/NoPort）",
             "noscenario": "（变体 noscenario：去掉端口/TTL/TCP 窗口 6 个场景特征后，SHAP/MI/Gain 与随机子集均从其余 34 个特征中选）"}.get(
        VARIANT, "（去重实验：40 特征精确去重 + 删除标签冲突行）" if DEDUP else "")
    L = ["RQ3 结果摘要" + title, "=" * 72]
    if chk:
        L += chk
    # 各条件在所有配置上的平均（任何 seed 数都可输出，试跑时用来检查是否合理）
    cm = (raw.groupby(["semantic_class", "condition", "testset"])[["f1", "pr_auc", "auc"]].mean()
          .round(4).unstack("testset"))
    cm.columns = [f"{m}_{t}" for m, t in cm.columns]
    L += ["", f"各条件平均（跨所有源–目标配置与 {n_seeds} 个 seed）：", cm.to_string()]
    tsk = []
    for lg in LOGS.glob("*.log") if LOGS.exists() else []:
        txt = lg.read_text(encoding="utf-8")
        if "用时" in txt:
            tsk.append(float(txt.rsplit("用时 ", 1)[1].split("s")[0]))
    if tsk:
        L += ["", f"单任务耗时：平均 {np.mean(tsk)/60:.1f} 分钟，最长 {np.max(tsk)/60:.1f} 分钟（{len(tsk)} 个任务）"]
    if lv.empty:
        L += ["", f"（统计检验已跳过：当前 {n_seeds} 个 seed，至少需要 3 个；完整 30 seed 跑完后再运行 aggregate）"]
    key = lv[(lv.metric.isin(["f1", "pr_auc"]))].sort_values(["testset", "metric", "comparison"]) \
        if not lv.empty else lv
    L += ["", "配置级检验（n=18 个源–目标–类别配置；簇级=按无序数据集对合并为 9 簇的精确符号翻转）："] \
        if not lv.empty else []
    for _, r in key.iterrows():
        L.append(f"  [{r.testset}/{r.metric:<6}] {r.comparison:<38} Δ>0: {r.n_positive:>2}/{r.n_configs}  "
                 f"均值Δ={r.mean_delta:+.4f}  p_cfg={r.p_config_wilcoxon:.4f}  p_cluster={r.p_cluster_signflip:.4f}  "
                 f"单配置显著(Wilcoxon+Holm)={r.n_sig_wilcoxon_holm}  (NB+Holm)={r.n_sig_nb_holm}")
    dz = raw[(raw.condition == "SHAP-10-Direct") & (raw.testset == "std")].groupby(
        ["semantic_class", "source", "target"])["auc"].mean()
    L += ["", f"零样本 SHAP-10-Direct 平均 AUC-ROC（std）= {dz.mean():.3f}；>0.5 的配置 {int((dz > 0.5).sum())}/{len(dz)}"]
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    plog(f"aggregate 完成 → {OUT.relative_to(ROOT)}/summary.txt")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="RQ3 feature-set and model portability experiments")
    sp = ap.add_subparsers(dest="cmd", required=True)
    cpu = os.cpu_count() or 8

    p = sp.add_parser("prep")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)

    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=max(1, (cpu - 2) // 4),
                   help="并行任务数（默认 (核数-2)/4）")
    p.add_argument("--threads", type=int, default=4, help="每个任务的 XGBoost 线程数")
    p.add_argument("--n-seeds", type=int, default=None, help="默认 30；--dedup 时默认 10")
    p.add_argument("--n-random", type=int, default=None, help="默认 20；--dedup 时默认 10")
    p.add_argument("--n-random-direct", type=int, default=None, help="默认 10；--dedup 时默认 5")
    p.add_argument("--classes", nargs="+", default=list(SEMANTIC_MAPPING))
    p.add_argument("--timeout-min", type=float, default=120)
    p.add_argument("--stall-min", type=float, default=25)

    p = sp.add_parser("worker")
    p.add_argument("--cls", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-random-direct", type=int, default=10)

    sp.add_parser("status")
    sp.add_parser("aggregate")
    for name, sub in sp.choices.items():
        sub.add_argument("--dedup", action="store_true",
                         help="去重实验：40 特征精确去重后再划分，结果写入 results/rq3_dedup/")
        sub.add_argument("--variant", choices=list(VARIANTS), default=None,
                         help="trainshap：SHAP 用训练集正例；reselect：去重数据上重选全部特征集；noscenario：去掉端口/TTL/TCP窗口后重选")
    a = ap.parse_args()
    set_mode(a.dedup, a.variant)
    if a.cmd == "run":
        dflt = (10, 10, 5) if (DEDUP or VARIANT) else (30, 20, 10)
        a.n_seeds = a.n_seeds if a.n_seeds is not None else dflt[0]
        a.n_random = a.n_random if a.n_random is not None else dflt[1]
        a.n_random_direct = a.n_random_direct if a.n_random_direct is not None else dflt[2]
    if a.cmd == "worker":
        try:
            cmd_worker(a)
        except Exception:
            traceback.print_exc()
            sys.exit(2)
    else:
        {"prep": cmd_prep, "run": cmd_run, "status": cmd_status, "aggregate": cmd_aggregate}[a.cmd](a)
        if a.cmd == "prep":          # 防止 macOS 下进程池子进程未回收导致解释器退出时卡住
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
