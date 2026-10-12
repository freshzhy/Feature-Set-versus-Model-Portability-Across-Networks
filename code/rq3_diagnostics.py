#!/usr/bin/env python3
"""
rq3_diagnostics.py — 划分方式与泄漏诊断（rq3_transfer.py 之前先跑）
=========================================================================
回答三个问题：
  A. 按 IP 分组划分是否可行？（读原始 CSV，因为 processed/ 已删除 IP 列）
     对每个数据集 × 每个类别统计：源/目的 IP 及 IP 对的唯一数、最大组占比、
     有效组数（inverse Simpson）；并检查攻击 IP 是否与 Benign 完全不重叠
     （若完全不重叠，说明 IP 本身就能区分类别，分组划分会改变任务性质）。
  B. 训练/测试集之间有多少完全重复的行？（读 processed/，按 RQ3 的
     抽样逻辑 + seed=42 复现 80/20 划分，统计测试行的特征向量在训练集
     中出现的比例；分 40 特征和该数据集 SHAP Top-10 两种口径）
  C. 原始类别计数 → raw_class_counts.json（rq3_transfer.py 的 prevalence-matched 测试集要用）

用法（项目根目录，先 source .venv/bin/activate）：
  python code/rq3_diagnostics.py                 # 全部 4 个数据集，4 进程并行
  python code/rq3_diagnostics.py --workers 2     # 内存紧张时减少并行
  python code/rq3_diagnostics.py --skip-raw      # 只做 B（不读原始大 CSV）

输出：results/rq3_diagnostics/
  raw_class_counts.json        原始类别计数（rq3_transfer.py 依赖）
  ip_group_stats.csv           A 的明细
  dup_leakage.csv              B 的明细
  diag_summary.txt             中文结论摘要（直接看这个）
  diag.log                     运行日志（tail -f 查看进度）
预计耗时：读原始 CSV 是大头（BoT 3,776 万行），M1 Ultra 上约 5–15 分钟。
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
DATASETS = {
    "UNSW": "NF-UNSW-NB15-v2",
    "CIC":  "NF-CSE-CIC-IDS2018-v2",
    "ToN":  "NF-ToN-IoT-v2",
    "BoT":  "NF-BoT-IoT-v2",
}
SEMANTIC_MAPPING = {
    "DoS": {"UNSW": ["DoS"],
            "CIC": ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
            "ToN": ["dos"], "BoT": ["DoS"]},
    "Reconnaissance": {"UNSW": ["Reconnaissance"], "ToN": ["scanning"],
                       "BoT": ["Reconnaissance"]},
}
MAX_TRAIN_POS, NEG_RATIO_CAP, TEST_SIZE, TOP_K = 30_000, 10, 0.20, 10
CHUNK = 2_000_000

OUT = ROOT / "results" / "rq3_diagnostics"
LOG = OUT / "diag.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ─────────────────────────── A + C：原始 CSV ───────────────────────────
def raw_ip_stats(ds_short):
    """分块读原始 CSV，只取 IP 和 Attack 列。返回 (类别计数, 每类统计行列表)。"""
    ds = DATASETS[ds_short]
    path = ROOT / "data" / "raw" / f"{ds}.csv"
    if not path.exists():
        return ds_short, None, [], f"缺少原始文件 {path}"
    src_c, dst_c, pair_c = {}, {}, {}
    counts = Counter()
    t0, n = time.time(), 0
    for chunk in pd.read_csv(path, usecols=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"],
                             dtype=str, chunksize=CHUNK):
        n += len(chunk)
        for cls, g in chunk.groupby("Attack", sort=False):
            counts[cls] += len(g)
            src_c.setdefault(cls, Counter()).update(g["IPV4_SRC_ADDR"].value_counts().to_dict())
            dst_c.setdefault(cls, Counter()).update(g["IPV4_DST_ADDR"].value_counts().to_dict())
            pr = (g["IPV4_SRC_ADDR"] + ">" + g["IPV4_DST_ADDR"]).value_counts().to_dict()
            pair_c.setdefault(cls, Counter()).update(pr)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}]   {ds_short}: 已读 {n:,} 行 "
                    f"({time.time()-t0:.0f}s)\n")

    benign_src = set(src_c.get("Benign", {}).keys())
    benign_dst = set(dst_c.get("Benign", {}).keys())
    rows = []
    for cls in sorted(counts):
        r = {"dataset": ds_short, "class": cls, "n_flows": counts[cls]}
        for name, c in (("src", src_c[cls]), ("dst", dst_c[cls]), ("pair", pair_c[cls])):
            v = np.array(list(c.values()), dtype=np.float64)
            p = v / v.sum()
            r[f"n_unique_{name}"] = len(v)
            r[f"top1_share_{name}"] = round(float(p.max()), 4)
            r[f"eff_groups_{name}"] = round(float(1.0 / np.sum(p ** 2)), 2)
        if cls != "Benign":
            # 攻击流中，源 IP 也出现在 Benign 中的流量比例（0 = IP 完全可分）
            s = src_c[cls]
            tot = sum(s.values())
            r["src_overlap_benign_flow_share"] = round(
                sum(v for k, v in s.items() if k in benign_src) / tot, 4)
            d = dst_c[cls]
            r["dst_overlap_benign_flow_share"] = round(
                sum(v for k, v in d.items() if k in benign_dst) / sum(d.values()), 4)
        rows.append(r)
    return ds_short, dict(counts), rows, None


# ─────────────────────────── B：重复行与划分泄漏 ───────────────────────────
def build_split_indices(attack, labels, seed, n_rows):
    """复现 rq3_transfer.py 的抽样与划分，返回 (train_idx, test_idx, y_train, y_test)。"""
    from sklearn.model_selection import train_test_split
    pos_mask = np.isin(attack, labels)
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
    tr, te, ytr, yte = train_test_split(idx, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    return tr, te, ytr, yte


def row_hash(df_feats):
    return pd.util.hash_pandas_object(df_feats, index=False).to_numpy()


def dup_leakage(ds_short):
    ds = DATASETS[ds_short]
    p = ROOT / "data" / "processed" / f"{ds}_processed.csv"
    if not p.exists():
        return ds_short, [], f"缺少 {p}"
    df = pd.read_csv(p, low_memory=False)
    feats = [c for c in df.columns if c not in ("Label", "Attack")]
    attack = df["Attack"].astype(str).to_numpy()
    h40 = row_hash(df[feats])

    # Top-10（该数据集 rq1 SHAP，按语义类别加权）
    top10 = {}
    pkl = ROOT / "results" / "rq1_signatures" / f"{ds}_rq1_shap_values.pkl"
    if pkl.exists():
        sigs = pickle.load(open(pkl, "rb"))
        for sem, mp in SEMANTIC_MAPPING.items():
            if ds_short not in mp:
                continue
            m = {l: sigs[l] for l in mp[ds_short] if l in sigs}
            if not m:
                continue
            fc = list(m.values())[0]["feature_cols"]
            w = sum(s.get("n_pos_shap", 1) for s in m.values())
            vec = sum(s["abs_mean_shap"] * s.get("n_pos_shap", 1) for s in m.values()) / w
            top10[sem] = [fc[i] for i in np.argsort(vec)[::-1][:TOP_K]]

    rows = []
    # 全数据集层面
    dup_all = pd.Series(h40).duplicated(keep=False).to_numpy()
    tmp = pd.DataFrame({"h": h40, "a": attack})
    n_lab = tmp.groupby("h")["a"].nunique()
    conflict = tmp["h"].map(n_lab).to_numpy() > 1
    rows.append({"dataset": ds_short, "scope": "whole_dataset", "features": "40",
                 "n_rows": len(df),
                 "dup_row_share": round(dup_all.mean(), 4),
                 "conflict_label_row_share": round(conflict.mean(), 4)})
    for cls in sorted(set(attack)):
        m = attack == cls
        rows.append({"dataset": ds_short, "scope": f"class:{cls}", "features": "40",
                     "n_rows": int(m.sum()),
                     "dup_row_share": round(pd.Series(h40[m]).duplicated(keep=False).mean(), 4),
                     "conflict_label_row_share": round(conflict[m].mean(), 4)})

    # 划分泄漏（seed=42，与 RQ3 相同的划分）
    for sem, mp in SEMANTIC_MAPPING.items():
        if ds_short not in mp:
            continue
        tr, te, ytr, yte = build_split_indices(attack, mp[ds_short], 42, len(df))
        variants = [("40", h40)]
        if sem in top10:
            variants.append(("SHAP-Top10", row_hash(df[top10[sem]])))
        for name, h in variants:
            in_train = np.isin(h[te], h[tr])
            r = {"dataset": ds_short, "scope": f"split:{sem}", "features": name,
                 "n_rows": len(te),
                 "test_in_train_share_all": round(in_train.mean(), 4),
                 "test_in_train_share_pos": round(in_train[yte == 1].mean(), 4),
                 "test_in_train_share_neg": round(in_train[yte == 0].mean(), 4)}
            # 测试正例的特征向量在训练集中以“负例”身份出现（标签冲突泄漏）
            neg_h = h[tr][ytr == 0]
            r["test_pos_seen_as_train_neg"] = round(np.isin(h[te][yte == 1], neg_h).mean(), 4)
            rows.append(r)
    return ds_short, rows, None


# ─────────────────────────── 汇总 ───────────────────────────
def verdict_ip(df):
    lines = []
    targets = {("UNSW", "DoS"), ("UNSW", "Reconnaissance"), ("ToN", "dos"), ("ToN", "scanning"),
               ("BoT", "DoS"), ("BoT", "Reconnaissance"), ("UNSW", "Benign"), ("ToN", "Benign"),
               ("BoT", "Benign"), ("CIC", "Benign")}
    targets |= {("CIC", c) for c in SEMANTIC_MAPPING["DoS"]["CIC"]}
    for _, r in df.iterrows():
        if (r["dataset"], r["class"]) not in targets:
            continue
        best = max(("src", "dst", "pair"), key=lambda k: r[f"eff_groups_{k}"])
        eff, top = r[f"eff_groups_{best}"], r[f"top1_share_{best}"]
        ok = "可行" if (eff >= 5 and top <= 0.5) else ("勉强" if (eff >= 3 and top <= 0.7) else "不可行")
        extra = ""
        if r["class"] != "Benign" and pd.notna(r.get("src_overlap_benign_flow_share")):
            if r["src_overlap_benign_flow_share"] < 0.01:
                extra = "；⚠ 攻击源 IP 几乎不出现在 Benign 中（IP 本身可区分该类）"
        lines.append(f"  {r['dataset']:<5}{r['class']:<26} 最佳分组键={best:<4} 有效组数={eff:>8.1f} "
                     f"最大组占比={top:.2f} → {ok}{extra}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(f"开始诊断，数据集={a.only} 并行={a.workers}")

    ip_rows, dup_rows, counts = [], [], {}
    jobs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        if not a.skip_raw:
            jobs += [ex.submit(raw_ip_stats, d) for d in a.only]
        jobs += [ex.submit(dup_leakage, d) for d in a.only]
        for fut in as_completed(jobs):
            try:
                res = fut.result()
            except Exception:
                log("✗ 子任务异常：\n" + traceback.format_exc())
                continue
            if len(res) == 4:
                ds, c, rows, err = res
                if err:
                    log(f"✗ {ds} 原始 CSV：{err}")
                    continue
                counts[ds] = c
                ip_rows += rows
                log(f"✓ {ds} 原始 CSV 统计完成（{sum(c.values()):,} 行）")
            else:
                ds, rows, err = res
                if err:
                    log(f"✗ {ds} 重复行：{err}")
                    continue
                dup_rows += rows
                log(f"✓ {ds} 重复行 / 划分泄漏统计完成")

    summary = ["诊断结论摘要", "=" * 70]
    if counts:
        old = {}
        cj = OUT / "raw_class_counts.json"
        if cj.exists():
            old = json.loads(cj.read_text())
        old.update(counts)
        cj.write_text(json.dumps(old, indent=2, ensure_ascii=False))
        ipdf = pd.DataFrame(ip_rows).sort_values(["dataset", "class"])
        ipdf.to_csv(OUT / "ip_group_stats.csv", index=False)
        summary += ["", "A. 按 IP 分组划分的可行性（判据：有效组数≥5 且最大组占比≤0.5）"]
        summary += verdict_ip(ipdf)
        summary += ["  说明：处理后的数据已删除 IP 列；若要做分组划分，需要重新预处理并保留 IP 列"
                    "（作为分组键，不进入特征）。"]
    if dup_rows:
        dd = pd.DataFrame(dup_rows)
        dd.to_csv(OUT / "dup_leakage.csv", index=False)
        summary += ["", "B. 训练/测试集精确重复（seed=42，与 RQ3 相同的划分）"]
        sp = dd[dd["scope"].str.startswith("split:")]
        for _, r in sp.iterrows():
            summary.append(
                f"  {r['dataset']:<5}{r['scope'][6:]:<16}{r['features']:<11} 测试行在训练集出现: "
                f"全部 {r['test_in_train_share_all']:.1%}  正例 {r['test_in_train_share_pos']:.1%}  "
                f"负例 {r['test_in_train_share_neg']:.1%}  正例特征=训练负例 {r['test_pos_seen_as_train_neg']:.1%}")
        whole = dd[dd["scope"] == "whole_dataset"]
        summary += ["", "  整个数据集（40 特征）："]
        for _, r in whole.iterrows():
            summary.append(f"  {r['dataset']:<5} 重复行占比 {r['dup_row_share']:.1%}  "
                           f"同特征不同标签行占比 {r['conflict_label_row_share']:.1%}")
        summary += ["", "  解读：测试行在训练集出现比例越高，行级随机划分对 F1/稳定性的抬高越严重；"
                    "‘正例特征=训练负例’>0 表示存在标签冲突（同一特征向量两种标签）。"]
    (OUT / "diag_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    log("完成。结论见 results/rq3_diagnostics/diag_summary.txt")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
