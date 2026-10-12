#!/usr/bin/env python3
"""
code/rq1_dedup_robustness.py — RQ1/RQ2 签名对精确重复行的稳健性
=====================================================================================
做什么：
  按 RQ1 原流程（rq1_signatures.py 的 build_ovr_dataset / train_xgb / extract_shap_signature，
  完全相同的抽样、清洗、超参数与 TreeSHAP），在两种数据上重新提取 29 个攻击类别的签名：
    orig  —— 原始 processed 数据（复现性核对：应与 results/rq1_signatures 中保存的签名完全一致）
    dedup —— 与 rq3_transfer.py --dedup 相同的去重规则（40 特征完全相同的行：标签冲突全删，同标签只留第一条）
  然后比较 “原签名 vs 去重签名”：
    ① 每个类别：保留行比例、F1/AUC 变化、Top-10 Jaccard、40 特征 Spearman、第 1 名特征是否相同
    ② 数据集内区分度（Table III 口径）：类间成对 Spearman 的均值/中位数/最小值
    ③ 跨数据集（Table IV 口径）：DoS / DDoS / Reconnaissance 的成对 Jaccard 与 Spearman

用法（项目根目录）：
  python3 code/rq1_dedup_robustness.py run              # 并行提取（默认 6 进程 × 3 线程）
  python3 code/rq1_dedup_robustness.py status           # 查看进度
  python3 code/rq1_dedup_robustness.py compare          # 汇总 → results/rq1_dedup_robustness/summary.txt
  一般直接用 ./run.sh robustness（依次跑本脚本和 RQ3 的各稳健性实验）。
断点续跑：每个 (模式, 数据集, 类别) 结果单独保存，重新执行 run 会跳过已完成的。
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

_T = os.environ.get("A8_THREADS")
if _T:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = _T

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # repository root (scripts live in code/)
OUT = ROOT / "results" / "rq1_dedup_robustness"
CACHE = OUT / "cache"
PROGRESS = OUT / "progress.log"
DATASETS = {"UNSW": "NF-UNSW-NB15-v2", "CIC": "NF-CSE-CIC-IDS2018-v2",
            "ToN": "NF-ToN-IoT-v2", "BoT": "NF-BoT-IoT-v2"}
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]
MODES = ["orig", "dedup"]
TOPK = 10
RQ2_MAPPING = {
    "DoS": {"UNSW": ["DoS"],
            "CIC": ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
            "ToN": ["dos"], "BoT": ["DoS"]},
    "DDoS": {"CIC": ["DDOS attack-HOIC", "DDoS attacks-LOIC-HTTP"], "ToN": ["ddos"], "BoT": ["DDoS"]},
    "Reconnaissance": {"UNSW": ["Reconnaissance"], "ToN": ["scanning"], "BoT": ["Reconnaissance"]},
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def plog(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stored_sigs(ds):
    return pickle.load(open(ROOT / "results" / "rq1_signatures" / f"{DATASETS[ds]}_rq1_shap_values.pkl", "rb"))


# ══════════════════════════════════════════════════════════════════════════
# ① 缓存：每个 (模式, 数据集) 一份特征矩阵 + 标签（保持 processed CSV 的行顺序）
# ══════════════════════════════════════════════════════════════════════════
def build_cache(mode, ds):
    xp, yp, mp = (CACHE / f"{mode}_{ds}_X.npy", CACHE / f"{mode}_{ds}_y.npy",
                  CACHE / f"{mode}_{ds}_meta.json")
    if xp.exists() and yp.exists() and mp.exists():
        return f"{mode}/{ds}: 缓存已存在"
    df = pd.read_csv(ROOT / "data" / "processed" / f"{DATASETS[ds]}_processed.csv", low_memory=False)
    fc = [c for c in df.columns if c not in ("Label", "Attack")]     # 与 RQ1 load_data 相同
    att = df["Attack"].astype(str)
    before = att.value_counts().to_dict()
    info = None
    if mode == "dedup":                                              # 与 rq3_transfer.py --dedup 完全相同
        h = pd.util.hash_pandas_object(df[fc], index=False)
        n_lab = pd.DataFrame({"h": h.values, "a": att.values}).groupby("h")["a"].nunique()
        conflict = h.map(n_lab).to_numpy() > 1
        keep = ~conflict & ~pd.Series(h.values).duplicated(keep="first").to_numpy()
        df = df[keep].reset_index(drop=True)
        att = df["Attack"].astype(str)
        info = {"n_before": int(len(keep)), "n_after": int(keep.sum()),
                "n_conflict_rows_dropped": int(conflict.sum())}
    after = att.value_counts().to_dict()
    tmp = CACHE / f"{mode}_{ds}_X.tmp.npy"
    np.save(tmp, df[fc].to_numpy(dtype=np.float64))
    np.save(yp, att.to_numpy(dtype=object), allow_pickle=True)
    mp.write_text(json.dumps({"feature_cols": fc, "dedup": info,
                              "per_class": {k: [int(before.get(k, 0)), int(after.get(k, 0))]
                                            for k in before}}, ensure_ascii=False, indent=1))
    os.replace(tmp, xp)
    return f"{mode}/{ds}: 缓存完成 {len(df):,} 行" + (
        f"（去重前 {info['n_before']:,}，标签冲突删除 {info['n_conflict_rows_dropped']:,}）" if info else "")


# ══════════════════════════════════════════════════════════════════════════
# ② 单个类别：完全调用 RQ1 原函数
# ══════════════════════════════════════════════════════════════════════════
def class_job(mode, ds, cls, threads):
    out = OUT / mode / ds / f"{safe(cls)}.pkl"
    if out.exists():
        return mode, ds, cls, "已存在"
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rq1_signatures as rq1
    rq1.EXP_CONFIG["xgb_params"]["n_jobs"] = threads
    meta = json.loads((CACHE / f"{mode}_{ds}_meta.json").read_text())
    fc = meta["feature_cols"]
    X = np.load(CACHE / f"{mode}_{ds}_X.npy")
    y = np.load(CACHE / f"{mode}_{ds}_y.npy", allow_pickle=True)
    df = pd.DataFrame(X, columns=fc)
    df["Attack"] = y
    del X
    n_cls = int((df["Attack"] == cls).sum())
    if n_cls < 20:
        res = {"skipped": True, "n_rows": n_cls}
    else:
        Xtr, Xte, ytr, yte, n_pos, n_neg = rq1.build_ovr_dataset(df, fc, cls)
        del df
        model = rq1.train_xgb(Xtr, ytr)
        met = rq1.evaluate_model(model, Xte, yte, cls)
        sig = rq1.extract_shap_signature(model, Xte, yte, fc, 0)
        res = {"skipped": False, "n_rows": n_cls, "metrics": met, "sig": sig,
               "n_test_pos": int((yte == 1).sum()), "n_train_pos": int((ytr == 1).sum())}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pickle.dump(res, open(tmp, "wb"))
    os.replace(tmp, out)
    return mode, ds, cls, f"{time.time() - t0:.0f}s"


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def all_jobs():
    return [(m, d, c) for m in MODES for d in DS_ORDER for c in sorted(stored_sigs(d))]


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    CACHE.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    plog("rq1_dedup_robustness ① 构建缓存（原始 + 去重，4 个数据集）")
    with ProcessPoolExecutor(max_workers=min(4, a.workers), mp_context=ctx) as ex:
        for f in as_completed([ex.submit(build_cache, m, d) for m in MODES for d in DS_ORDER]):
            plog("   " + f.result())
    jobs = [j for j in all_jobs() if not (OUT / j[0] / j[1] / f"{safe(j[2])}.pkl").exists()]
    plog(f"rq1_dedup_robustness ② 提取签名：共 {len(all_jobs())} 个，待完成 {len(jobs)}，并行 {a.workers} × 线程 {a.threads}")
    os.environ["A8_THREADS"] = str(a.threads)
    failed = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        futs = {ex.submit(class_job, m, d, c, a.threads): (m, d, c) for m, d, c in jobs}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                m, d, c, msg = f.result()
                plog(f"   [{done}/{len(jobs)}] {m}/{d}/{c}  {msg}")
            except Exception:
                failed.append(futs[f])
                plog(f"   ✗ {futs[f]} 失败：\n{traceback.format_exc()}")
    if failed:
        plog(f"✗ 有 {len(failed)} 个失败，重新执行 run 会只重跑失败的部分")
        sys.exit(1)
    plog("rq1_dedup_robustness 签名提取完成；下一步：python3 code/rq1_dedup_robustness.py compare")


def cmd_status(a):
    jobs = all_jobs()
    ok = [j for j in jobs if (OUT / j[0] / j[1] / f"{safe(j[2])}.pkl").exists()]
    print(f"签名提取进度：{len(ok)}/{len(jobs)}")
    for m in MODES:
        print(f"  {m:<6}: {sum(1 for j in ok if j[0] == m)}/{sum(1 for j in jobs if j[0] == m)}")
    if PROGRESS.exists():
        print("\n最近日志：")
        print("\n".join(PROGRESS.read_text(encoding="utf-8").splitlines()[-8:]))


# ══════════════════════════════════════════════════════════════════════════
# ③ compare
# ══════════════════════════════════════════════════════════════════════════
def load_mode(mode, ds):
    res = {}
    for c in sorted(stored_sigs(ds)):
        p = OUT / mode / ds / f"{safe(c)}.pkl"
        if p.exists():
            res[c] = pickle.load(open(p, "rb"))
    return res


def topk(v, fc):
    return [fc[i] for i in np.argsort(np.asarray(v))[::-1][:TOPK]]


def jac(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def composite(sigs, labels):
    m = [sigs[l] for l in labels if l in sigs]
    if not m:
        return None
    w = sum(s.get("n_pos_shap", 1) for s in m)
    return sum(np.asarray(s["abs_mean_shap"]) * s.get("n_pos_shap", 1) for s in m) / w


def cmd_compare(a):
    from scipy.stats import spearmanr
    rows, intra, cross = [], [], []
    S, D, O = {}, {}, {}
    for ds in DS_ORDER:
        st = stored_sigs(ds)
        fc = list(st.values())[0]["feature_cols"]
        meta = json.loads((CACHE / f"dedup_{ds}_meta.json").read_text())
        rm = pd.read_csv(ROOT / "results" / "rq1_signatures" / f"{DATASETS[ds]}_rq1_model_metrics.csv").set_index("class")
        orig, dedup = load_mode("orig", ds), load_mode("dedup", ds)
        S[ds] = {c: st[c] for c in st}
        D[ds] = {c: r["sig"] for c, r in dedup.items() if not r["skipped"]}
        O[ds] = {c: r["sig"] for c, r in orig.items() if not r["skipped"]}
        for c in sorted(st):
            b, af = meta["per_class"].get(c, [np.nan, np.nan])
            r = {"dataset": ds, "class": c, "rows_before": b, "rows_after": af,
                 "kept_share": round(af / b, 4) if b else np.nan,
                 "f1_orig": round(float(rm.loc[c, "f1"]), 4) if c in rm.index else np.nan,
                 "auc_orig": round(float(rm.loc[c, "auc_roc"]), 4) if c in rm.index else np.nan}
            s0 = np.asarray(st[c]["abs_mean_shap"])
            if c in O[ds]:     # 复现性核对
                r["repro_jaccard"] = jac(topk(s0, fc), topk(O[ds][c]["abs_mean_shap"], fc))
                r["repro_spearman"] = round(float(spearmanr(s0, O[ds][c]["abs_mean_shap"])[0]), 4)
            if c in dedup and not dedup[c]["skipped"]:
                s1 = np.asarray(D[ds][c]["abs_mean_shap"])
                t0, t1 = topk(s0, fc), topk(s1, fc)
                r.update({"n_test_pos_dedup": dedup[c]["n_test_pos"],
                          "f1_dedup": round(dedup[c]["metrics"]["f1"], 4),
                          "auc_dedup": round(dedup[c]["metrics"]["auc_roc"], 4),
                          "top10_jaccard": round(jac(t0, t1), 4),
                          "top10_shared": len(set(t0) & set(t1)),
                          "spearman_40": round(float(spearmanr(s0, s1)[0]), 4),
                          "rank1_orig": t0[0], "rank1_dedup": t1[0], "rank1_same": t0[0] == t1[0],
                          "top10_dedup": ";".join(t1)})
                if c in O[ds]:   # 论文采用的口径：去重签名 vs 同环境复跑签名（排除重训练波动）
                    so = np.asarray(O[ds][c]["abs_mean_shap"])
                    to = topk(so, fc)
                    r.update({"f1_rerun": round(orig[c]["metrics"]["f1"], 4),
                              "vsrerun_jaccard": round(jac(to, t1), 4),
                              "vsrerun_spearman": round(float(spearmanr(so, s1)[0]), 4),
                              "rank1_rerun": to[0], "rank1_same_vs_rerun": to[0] == t1[0],
                              "repro_rank1_same": to[0] == t0[0]})
            rows.append(r)
        # 数据集内区分度（Table III 口径）
        for lab, sig in (("original", S[ds]), ("rerun", O[ds]), ("dedup", D[ds])):
            cl = sorted(c for c in S[ds] if c in D[ds] and c in O[ds])      # 两种数据都有的类别，保证可比
            rhos = [spearmanr(sig[x]["abs_mean_shap"], sig[y]["abs_mean_shap"])[0]
                    for i, x in enumerate(cl) for y in cl[i + 1:]]
            if rhos:
                intra.append({"dataset": ds, "signatures": lab, "classes": len(cl), "pairs": len(rhos),
                              "mean_rho": round(float(np.mean(rhos)), 4),
                              "median_rho": round(float(np.median(rhos)), 4),
                              "min_rho": round(float(np.min(rhos)), 4),
                              "max_rho": round(float(np.max(rhos)), 4)})
    # 跨数据集（Table IV 口径）
    fc = list(stored_sigs("UNSW").values())[0]["feature_cols"]
    for sem, mp in RQ2_MAPPING.items():
        dss = [d for d in DS_ORDER if d in mp]
        for i, d1 in enumerate(dss):
            for d2 in dss[i + 1:]:
                r = {"semantic_class": sem, "pair": f"{d1}↔{d2}"}
                for lab, sig in (("original", S), ("rerun", O), ("dedup", D)):
                    v1, v2 = composite(sig[d1], mp[d1]), composite(sig[d2], mp[d2])
                    if v1 is None or v2 is None:
                        continue
                    r[f"jaccard_{lab}"] = round(jac(topk(v1, fc), topk(v2, fc)), 4)
                    r[f"spearman_{lab}"] = round(float(spearmanr(v1, v2)[0]), 4)
                cross.append(r)
    per = pd.DataFrame(rows)
    intra = pd.DataFrame(intra)
    cross = pd.DataFrame(cross)
    per.to_csv(OUT / "signature_dedup.csv", index=False)
    intra.to_csv(OUT / "intra_distinctiveness.csv", index=False)
    cross.to_csv(OUT / "cross_dataset.csv", index=False)

    L = ["rq1_dedup_robustness：RQ1/RQ2 签名对精确去重的稳健性", "=" * 72]
    if "repro_jaccard" in per:
        rp = per.dropna(subset=["repro_jaccard"])
        L += ["", f"复现性核对（原始数据按 RQ1 流程复跑 vs results/rq1_signatures 已存签名，{len(rp)} 类）：",
              f"  Top-10 Jaccard 最小 {rp.repro_jaccard.min():.3f}，Spearman 最小 {rp.repro_spearman.min():.4f}"
              "（都应为 1；若不是 1，说明复跑环境与当初不同，下面的比较应改用 orig 复跑结果解读）"]
    pv = per.dropna(subset=["top10_jaccard"])
    L += ["", "每个类别：原签名 vs 去重签名（按保留比例升序）",
          f"  {'数据集':<5}{'类别':<26}{'保留':>7}{'测试正例':>8}{'F1 原→去重':>17}{'Jac':>6}{'共享':>5}{'ρ40':>7}  第1名"]
    for _, r in pv.sort_values("kept_share").iterrows():
        L.append(f"  {r.dataset:<5}{r['class'][:25]:<26}{r.kept_share:>7.1%}{int(r.n_test_pos_dedup):>8}"
                 f"{r.f1_orig:>8.3f}→{r.f1_dedup:<8.3f}{r.top10_jaccard:>6.2f}{int(r.top10_shared):>5}"
                 f"{r.spearman_40:>7.3f}  {'同' if r.rank1_same else r.rank1_orig + '→' + r.rank1_dedup}")
    sk = per[per.top10_jaccard.isna()] if "top10_jaccard" in per else per
    if len(sk):
        L.append(f"  （去重后样本过少未分析：{', '.join(sk.dataset + '/' + sk['class'])}）")
    hd = pv[pv.kept_share < 0.5]
    L += ["", f"汇总：全部 {len(pv)} 类 Top-10 Jaccard 中位数 {pv.top10_jaccard.median():.2f}"
              f"（范围 {pv.top10_jaccard.min():.2f}–{pv.top10_jaccard.max():.2f}），"
              f"Spearman 中位数 {pv.spearman_40.median():.3f}（最小 {pv.spearman_40.min():.3f}），"
              f"第 1 名特征不变 {int(pv.rank1_same.sum())}/{len(pv)}",
          f"      保留 < 50% 的 {len(hd)} 类：Jaccard 中位数 {hd.top10_jaccard.median():.2f}，"
          f"Spearman 中位数 {hd.spearman_40.median():.3f}" if len(hd) else ""]
    if "vsrerun_jaccard" in per:
        q = per.dropna(subset=["vsrerun_jaccard"]).copy()
        q["组"] = pd.cut(q.kept_share, [0, 0.5, 0.95, 1.01], right=False, labels=["<50%", "50–95%", "≥95%"])
        g = q.groupby("组", observed=True).agg(n=("class", "size"), 去重vs复跑_Jac=("vsrerun_jaccard", "median"),
                                            去重vs复跑_rho=("vsrerun_spearman", "median"),
                                            重训练参照_Jac=("repro_jaccard", "median"),
                                            重训练参照_rho=("repro_spearman", "median"),
                                            第1名不变=("rank1_same_vs_rerun", "sum")).round(3)
        L += ["", "【论文口径】去重签名 vs 同环境复跑签名（按保留比例分组，中位数；重训练参照 = 已存签名 vs 复跑）：",
              g.to_string(),
              f"  全部：重训练参照 Jaccard 中位数 {q.repro_jaccard.median():.2f}（{q.repro_jaccard.min():.2f}–{q.repro_jaccard.max():.2f}），"
              f"ρ 中位数 {q.repro_spearman.median():.3f}（最小 {q.repro_spearman.min():.3f}），第1名不变 {int(q.repro_rank1_same.sum())}/{len(q)}；"
              f"F1 去重−复跑 中位数 {(q.f1_dedup - q.f1_rerun).median():+.4f}，最大绝对值 {(q.f1_dedup - q.f1_rerun).abs().max():.4f}"]
    L += ["", "数据集内区分度（Table IV 口径，类间成对 Spearman）：", intra.to_string(index=False)]
    L += ["", "跨数据集签名相似度（Table V 口径）：", cross.to_string(index=False)]
    if len(cross):
        g = cross.groupby("semantic_class")[[c for c in cross if c.startswith(("jaccard", "spearman"))]].mean().round(3)
        L += ["", "按语义类别平均：", g.to_string()]
    (OUT / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    plog("compare 完成 → results/rq1_dedup_robustness/summary.txt")


def main():
    ap = argparse.ArgumentParser(description="rq1_dedup_robustness：签名去重稳健性")
    sp = ap.add_subparsers(dest="cmd", required=True)
    cpu = os.cpu_count() or 8
    p = sp.add_parser("run")
    p.add_argument("--workers", type=int, default=max(1, min(6, (cpu - 2) // 3)))
    p.add_argument("--threads", type=int, default=3)
    sp.add_parser("status")
    sp.add_parser("compare")
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
