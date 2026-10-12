#!/usr/bin/env python3
"""
plot_figures_rq3.py — RQ3 figures of the paper (Figs. 6–9 and Fig. S2).
Usage (repository root):  python code/plot_figures_rq3.py
Inputs : results/rq3_main/{config_summary,per_config_tests}.csv,
         results/rq3_jsd/jsd_matrix.csv, results/rq2_similarity/rq2_jaccard_*.csv
Output : results/figures/
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(os.environ.get("A7_ROOT", Path(__file__).resolve().parent.parent))
R = ROOT / "results"
OUT = R / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
W = 7.16
ENV = {"UNSW": "IT", "CIC": "IT", "ToN": "IoT", "BoT": "IoT"}
ORDER = [("DoS", p) for p in ["CIC→UNSW", "ToN→UNSW", "BoT→UNSW", "UNSW→CIC", "ToN→CIC", "BoT→CIC",
                              "UNSW→ToN", "CIC→ToN", "BoT→ToN", "UNSW→BoT", "CIC→BoT", "ToN→BoT"]] + \
        [("Reconnaissance", p) for p in ["ToN→UNSW", "BoT→UNSW", "UNSW→ToN", "BoT→ToN", "UNSW→BoT", "ToN→BoT"]]
LABELS = [("DoS " if c == "DoS" else "Recon ") + p for c, p in ORDER]
# 色盲友好配色（Okabe–Ito）
COL = {"SHAP-10": "#0072B2", "MI-10": "#E69F00", "Gain-10": "#009E73",
       "SHAP-10-NoPort": "#CC79A7", "Random-10": "#7F7F7F"}

cs = pd.read_csv(R / "rq3_main" / "config_summary.csv")
per = pd.read_csv(R / "rq3_main" / "per_config_tests.csv")


def val(cls, pair, cond, metric="f1", ts="std"):
    s, t = pair.split("→")
    src = "Random" if cond in ("Random-10", "Random-10-single") else s
    r = cs[(cs.semantic_class == cls) & (cs.target == t) & (cs.source == src)
           & (cs.condition == cond) & (cs.testset == ts)]
    return r[f"{metric}_mean"].values[0], r[f"{metric}_std"].values[0]


def ptest(comp, cls, pair, metric="f1", ts="std"):
    return per[(per.comparison == comp) & (per.semantic_class == cls) & (per.pair == pair)
               & (per.testset == ts) & (per.metric == metric)].iloc[0]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"✓ {OUT.relative_to(ROOT)}/{name}.pdf/.png")


# ── Fig. 7：各条件 F1 ─────────────────────────────────────────────
conds = ["SHAP-10", "MI-10", "Gain-10", "SHAP-10-NoPort", "Random-10"]
fig, ax = plt.subplots(figsize=(W, 3.0))
x = np.arange(len(ORDER))
off = np.linspace(-0.3, 0.3, len(conds))
for k, c in enumerate(conds):
    m = np.array([val(cl, p, c)[0] for cl, p in ORDER])
    sd = np.array([val(cl, p, c)[1] for cl, p in ORDER])
    ax.errorbar(x + off[k], m, yerr=sd, fmt="o", ms=3, lw=0.8, capsize=1.5, color=COL[c], label=c)
ax.axvline(11.5, color="#999999", lw=0.6, ls=":")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=55, ha="right")
ax.set_ylabel("F1 (1:10 test set)")
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.grid(axis="y", lw=0.3, alpha=0.5)
save(fig, "FigS2_transfer_f1")

# ── Fig. 8：森林图（NB 校正 95% CI）─────────────────────────────────
comps = [("SHAP-10 vs Random-10", "(a) SHAP-10 − Random-10"),
         ("SHAP-10 vs MI-10", "(b) SHAP-10 − MI-10"),
         ("SHAP-10 vs Gain-10", "(c) SHAP-10 − Gain-10")]
fig, axes = plt.subplots(1, 3, figsize=(W, 3.6), sharey=True)
y = np.arange(len(ORDER))[::-1]
for ax, (comp, title) in zip(axes, comps):
    for yi, (cl, p) in zip(y, ORDER):
        r = ptest(comp, cl, p)
        sig = r.p_nb_holm < 0.05
        c = "#0072B2" if r.delta > 0 else "#D55E00"
        ax.plot([r.nb_ci_lo, r.nb_ci_hi], [yi, yi], color=c, lw=1)
        ax.plot(r.delta, yi, "o", ms=4, mfc=c if sig else "white", mec=c, mew=1)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.axhline(5.5, color="#999999", lw=0.6, ls=":")
    ax.set_title(title, loc="left")
    ax.set_xlabel("ΔF1 (95% CI, Nadeau–Bengio)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels(LABELS)
save(fig, "Fig6_paired_differences")

# ── Fig. 9：环境类别 ─────────────────────────────────────────────
rows = []
for cl, p in ORDER:
    s, t = p.split("→")
    rows.append(dict(env="Same" if ENV[s] == ENV[t] else "Cross", f1=val(cl, p, "SHAP-10")[0],
                     d=ptest("SHAP-10 vs Random-10", cl, p).delta))
e = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(W * 0.75, 2.6))
fig.subplots_adjust(wspace=0.45)
for ax, col, lab in [(axes[0], "f1", "(a) SHAP-10 F1"), (axes[1], "d", "(b) ΔF1 (SHAP-10 − Random-10)")]:
    grp = [e[e.env == "Same"][col], e[e.env == "Cross"][col]]
    ax.boxplot(grp, widths=0.5, showfliers=False, medianprops=dict(color="black"))
    for i, g in enumerate(grp, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(g)), g, s=12, color="#0072B2", alpha=0.8, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Same (n={len(grp[0])})", f"Cross (n={len(grp[1])})"])
    p = mannwhitneyu(grp[0], grp[1], alternative="two-sided").pvalue
    ax.set_title(f"{lab}\nMann–Whitney p = {p:.3f}", loc="left")
    if col == "d":
        ax.axhline(0, color="#555555", lw=0.6)
save(fig, "Fig7_environment")

# ── Fig. 10：预测变量 ────────────────────────────────────────────
jsd = pd.read_csv(R / "rq3_jsd" / "jsd_matrix.csv")
recs = []
for cl, p in ORDER:
    s, t = p.split("→")
    j = jsd[(jsd.semantic_class == cl) & (jsd.source == s) & (jsd.target == t)].iloc[0]
    jm = pd.read_csv(R / "rq2_similarity" / f"rq2_jaccard_{cl}.csv", index_col=0)
    r = ptest("SHAP-10 vs Random-10", cl, p)
    recs.append(dict(cls=cl, jsd_all=j.jsd_all_feats, jsd_top=j.jsd_topk_feats, jac=jm.loc[s, t],
                     d=r.delta, sig=r.p_nb_holm < 0.05))
pdf = pd.DataFrame(recs)
fig, axes = plt.subplots(1, 3, figsize=(W, 2.4), sharey=True)
for ax, col, lab in [(axes[0], "jsd_all", "(a) JSD, all 40 features"),
                     (axes[1], "jsd_top", "(b) JSD, source Top-10"),
                     (axes[2], "jac", "(c) Top-10 Jaccard similarity")]:
    for cl, mk in [("DoS", "o"), ("Reconnaissance", "s")]:
        sub = pdf[pdf.cls == cl]
        ax.scatter(sub[sub.sig][col], sub[sub.sig].d, marker=mk, s=18, color="#0072B2", label=f"{cl} (sig.)")
        ax.scatter(sub[~sub.sig][col], sub[~sub.sig].d, marker=mk, s=18, facecolors="white",
                   edgecolors="#0072B2", label=f"{cl} (n.s.)")
    rho, pv = spearmanr(pdf[col], pdf.d)
    ax.set_title(f"{lab}\nSpearman ρ = {rho:.2f}, p = {pv:.3f}", loc="left")
    ax.axhline(0, color="#555555", lw=0.6)
axes[0].set_ylabel("ΔF1 (SHAP-10 − Random-10)")
axes[2].legend(loc="upper left", frameon=False, fontsize=6)
save(fig, "Fig8_predictors")

# ── Fig. 11：零样本 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(W, 3.6), sharey=True)
y = np.arange(len(ORDER))[::-1]
for ax, m, lab in [(axes[0], "auc", "(a) AUC-ROC"), (axes[1], "pauc01", "(b) pAUC@1%FPR")]:
    for yi, (cl, p) in zip(y, ORDER):
        a = val(cl, p, "SHAP-10-Direct", m)[0]
        b = val(cl, p, "Random-10-Direct", m)[0]
        ax.plot([b, a], [yi, yi], color="#BBBBBB", lw=1, zorder=1)
        ax.scatter(b, yi, s=16, color="#7F7F7F", zorder=2, label="Random-10-Direct" if yi == y[0] else None)
        ax.scatter(a, yi, s=16, color="#0072B2", zorder=3, label="SHAP-10-Direct" if yi == y[0] else None)
    ax.axvline(0.5, color="#D55E00", lw=0.7, ls="--")
    ax.axhline(5.5, color="#999999", lw=0.6, ls=":")
    ax.set_title(lab, loc="left")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
axes[0].set_yticks(y)
axes[0].set_yticklabels(LABELS)
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.subplots_adjust(bottom=0.14)
save(fig, "Fig9_zero_shot")
print("全部完成 →", OUT)
