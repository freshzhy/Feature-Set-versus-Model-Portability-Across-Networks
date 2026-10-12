#!/usr/bin/env python3
"""
论文可视化图表 v2 — 可读性改进版
主要改动：
  Fig1: 改为分组点图，减少视觉噪声，Y轴标签不再重叠
  Fig2: X轴特征名缩写 + 统一色标范围 + 增大字号
  Fig3: 气泡大小重新归一化，分开参考气泡和图例
  Fig4: Spearman 色标改为数据实际范围（0~0.8），提升对比度
  Fig5: 只保留出现>=2次的特征，大幅缩短左图
"""

import argparse, json, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STYLE = {
    "font_family":  "DejaVu Sans",
    "font_size":    8,
    "title_size":   9,
    "label_size":   8,
    "tick_size":    7,
    "legend_size":  7,
    "col_width":    3.5,
    "full_width":   7.16,
    "fig_dpi":      300,
    "colors": {
        "blue":   "#2166ac",
        "red":    "#d6604d",
        "green":  "#4dac26",
        "orange": "#f4a582",
        "purple": "#7b2d8b",
        "gray":   "#888888",
    },
    "dataset_colors":  {"UNSW":"#2166ac","CIC":"#d6604d","ToN":"#4dac26","BoT":"#7b2d8b"},
    "dataset_markers": {"UNSW":"o","CIC":"s","ToN":"^","BoT":"D"},
}

DATASET_SHORT = {
    "NF-UNSW-NB15-v2":"UNSW",
    "NF-CSE-CIC-IDS2018-v2":"CIC",
    "NF-ToN-IoT-v2":"ToN",
    "NF-BoT-IoT-v2":"BoT",
}
DS_ORDER = ["UNSW","CIC","ToN","BoT"]
OUT_DIR  = Path("./results/figures")

FEAT_ABBREV = {
    "L4_SRC_PORT":"SRC_PORT","L4_DST_PORT":"DST_PORT",
    "PROTOCOL":"PROTO","L7_PROTO":"L7_PROTO",
    "IN_BYTES":"IN_B","IN_PKTS":"IN_P",
    "OUT_BYTES":"OUT_B","OUT_PKTS":"OUT_P",
    "TCP_FLAGS":"TCP_FL","CLIENT_TCP_FLAGS":"CLI_FL","SERVER_TCP_FLAGS":"SRV_FL",
    "FLOW_DURATION_MILLISECONDS":"FLOW_DUR",
    "DURATION_IN":"DUR_IN","DURATION_OUT":"DUR_OUT",
    "MIN_TTL":"MIN_TTL","MAX_TTL":"MAX_TTL",
    "LONGEST_FLOW_PKT":"LONG_PKT","SHORTEST_FLOW_PKT":"SHRT_PKT",
    "MIN_IP_PKT_LEN":"MIN_IP_LEN",
    "SRC_TO_DST_SECOND_BYTES":"S2D_SEC_B","DST_TO_SRC_SECOND_BYTES":"D2S_SEC_B",
    "RETRANSMITTED_IN_BYTES":"REXMT_IN_B","RETRANSMITTED_IN_PKTS":"REXMT_IN_P",
    "RETRANSMITTED_OUT_BYTES":"REXMT_OT_B","RETRANSMITTED_OUT_PKTS":"REXMT_OT_P",
    "SRC_TO_DST_AVG_THROUGHPUT":"S2D_THRU","DST_TO_SRC_AVG_THROUGHPUT":"D2S_THRU",
    "NUM_PKTS_UP_TO_128_BYTES":"P≤128","NUM_PKTS_128_TO_256_BYTES":"P128-256",
    "NUM_PKTS_256_TO_512_BYTES":"P256-512","NUM_PKTS_512_TO_1024_BYTES":"P512-1K",
    "NUM_PKTS_1024_TO_1514_BYTES":"P1K-1.5K",
    "TCP_WIN_MAX_IN":"WIN_IN","TCP_WIN_MAX_OUT":"WIN_OUT",
    "ICMP_TYPE":"ICMP_T","ICMP_IPV4_TYPE":"ICMP4_T",
    "DNS_QUERY_ID":"DNS_ID","DNS_QUERY_TYPE":"DNS_TYP",
    "DNS_TTL_ANSWER":"DNS_TTL","FTP_COMMAND_RET_CODE":"FTP_RET",
}

def abbrev(name):
    return FEAT_ABBREV.get(name, name[:10])

def setup_style():
    plt.rcParams.update({
        "font.family":STYLE["font_family"],"font.size":STYLE["font_size"],
        "axes.titlesize":STYLE["title_size"],"axes.labelsize":STYLE["label_size"],
        "xtick.labelsize":STYLE["tick_size"],"ytick.labelsize":STYLE["tick_size"],
        "legend.fontsize":STYLE["legend_size"],"axes.linewidth":0.5,
        "axes.spines.top":False,"axes.spines.right":False,
        "grid.linewidth":0.4,"grid.alpha":0.4,
        "figure.dpi":STYLE["fig_dpi"],"savefig.dpi":STYLE["fig_dpi"],
        "savefig.bbox":"tight","savefig.pad_inches":0.05,
    })

def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR/f"{name}.pdf", format="pdf")
    fig.savefig(OUT_DIR/f"{name}.png", format="png")
    print(f"  ✓ {name}.pdf/.png")
    plt.close(fig)

def load_metrics(rq1_dir):
    r={}
    for ds_full,ds_short in DATASET_SHORT.items():
        p=Path(rq1_dir)/f"{ds_full}_rq1_model_metrics.csv"
        if p.exists(): r[ds_short]=pd.read_csv(p)
    return r

def load_shap_matrices(rq1_dir):
    r={}
    for ds_full,ds_short in DATASET_SHORT.items():
        p=Path(rq1_dir)/f"{ds_full}_rq1_shap_matrix.csv"
        if p.exists(): r[ds_short]=pd.read_csv(p,index_col=0)
    return r

def load_topk(rq1_dir):
    r={}
    for ds_full,ds_short in DATASET_SHORT.items():
        for k in [10,5,15]:
            p=Path(rq1_dir)/f"{ds_full}_rq1_top{k}_signatures.csv"
            if p.exists(): r[ds_short]=pd.read_csv(p); break
    return r

def load_rq2(rq2_dir):
    d={}
    sp=Path(rq2_dir)/"rq2_summary.json"
    rp=Path(rq2_dir)/"rq2_transferability_report.csv"
    if sp.exists():
        with open(sp,encoding="utf-8") as f: d["summary"]=json.load(f)
    if rp.exists(): d["report"]=pd.read_csv(rp)
    for cls in ["DoS","DDoS","Reconnaissance"]:
        for m in ["jaccard","spearman"]:
            p=Path(rq2_dir)/f"rq2_{m}_{cls}.csv"
            if p.exists(): d[f"{m}_{cls}"]=pd.read_csv(p,index_col=0)
    topk_p=Path(rq2_dir)/"rq2_topk_DoS.csv"
    if topk_p.exists(): d["topk_DoS"]=pd.read_csv(topk_p)
    return d


# ── Figure 1：分组点图（每数据集一行，类别为点） ─────────────────────────────
def plot_fig1(metrics_by_ds):
    print("\n[Figure 1] 性能分组点图")
    fig, axes = plt.subplots(1,2,figsize=(STYLE["full_width"],2.8))

    for ax_idx,(metric,xlabel) in enumerate([("f1","F1 score"),("auc_roc","AUC-ROC")]):
        ax=axes[ax_idx]
        yticks,ylabels=[],[]

        for y_pos,ds in enumerate(DS_ORDER):
            if ds not in metrics_by_ds: continue
            df=metrics_by_ds[ds]
            color=STYLE["dataset_colors"][ds]
            marker=STYLE["dataset_markers"][ds]
            vals=df[metric].values

            # 水平均值 ± std 误差棒
            mean_v=vals.mean(); std_v=vals.std()
            ax.barh(y_pos, mean_v, height=0.45,
                    color=color, alpha=0.18, zorder=1)
            ax.errorbar(mean_v, y_pos, xerr=std_v,
                        fmt="none", color=color, linewidth=1.2,
                        capsize=3, capthick=1.0, zorder=3)
            ax.plot(mean_v, y_pos, marker="|",
                    color=color, markersize=8, markeredgewidth=1.5, zorder=4)

            # 叠加各类别散点（抖动）
            jitter=np.random.uniform(-0.18,0.18,len(vals))
            ax.scatter(vals, np.full(len(vals),y_pos)+jitter,
                       s=18, color=color, alpha=0.75, marker=marker,
                       zorder=5, linewidths=0)

            # 标注低性能类别（F1 < 0.75）
            for v,cls_name in zip(vals,df["class"]):
                if v < 0.75:
                    ax.annotate(f"{'Infiltration' if cls_name=='Infilteration' else cls_name} ({v:.2f})",
                                xy=(v,y_pos), xytext=(v+0.03,y_pos+0.32),
                                fontsize=5.5, color=color,
                                arrowprops=dict(arrowstyle="-",lw=0.5,color=color))

            yticks.append(y_pos); ylabels.append(ds)

        ax.set_yticks(yticks); ax.set_yticklabels(ylabels,fontsize=8)
        for tick,ds in zip(ax.get_yticklabels(),DS_ORDER):
            tick.set_color(STYLE["dataset_colors"].get(ds,"black"))

        ax.set_xlabel(xlabel); ax.set_xlim(0.25,1.06)
        ax.axvline(0.9,color="#aaaaaa",linestyle="--",linewidth=0.6,alpha=0.7)
        ax.grid(axis="x",zorder=0); ax.set_ylim(-0.6,len(DS_ORDER)-0.4)
        panel=chr(ord("a")+ax_idx)
        ax.set_title(f"({panel}) {xlabel}",loc="left",fontweight="bold")

    pass
    fig.tight_layout()
    save_fig(fig,"Fig1_classification_performance")


# ── Figure 2：SHAP 热力图（统一色标，特征名缩写）─────────────────────────────
def plot_fig2(shap_matrices):
    print("\n[Figure 2] SHAP 热力图")
    available=[ds for ds in DS_ORDER if ds in shap_matrices]
    fig,axes=plt.subplots(2,2,figsize=(STYLE["full_width"],5.5))
    axes=axes.flatten()

    # 全局最大值（统一色标）
    gmax=max(m.values.max() for m in shap_matrices.values())

    for i,ds in enumerate(available):
        ax=axes[i]
        mat=shap_matrices[ds]

        # 只保留 SHAP 最大值>0.05*gmax 的特征
        feat_mask=(mat.max(axis=0)>0.05*gmax)
        mat_f=mat.loc[:,feat_mask]
        # 特征按全局最大值降序排列
        feat_order=mat_f.max(axis=0).sort_values(ascending=False).index
        mat_plot=mat_f[feat_order]

        im=ax.imshow(mat_plot.values,aspect="auto",cmap="YlOrRd",
                     vmin=0,vmax=gmax)

        # X 轴：缩写特征名
        ax.set_xticks(range(len(feat_order)))
        ax.set_xticklabels([abbrev(f) for f in feat_order],
                           rotation=65,ha="right",fontsize=5.0,
                           rotation_mode="anchor")
        ax.set_yticks(range(len(mat_plot.index)))
        ax.set_yticklabels([("Infiltration" if str(l)=="Infilteration" else l) for l in mat_plot.index],fontsize=6.5)

        # 标记每行最大值列
        max_col=mat_plot.values.argmax(axis=1)
        for r_idx,c_idx in enumerate(max_col):
            ax.text(c_idx,r_idx,"★",ha="center",va="center",
                    fontsize=6,color="white",alpha=0.95)

        cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("|SHAP| mean",fontsize=6)

        panel=chr(ord("a")+i)
        ax.set_title(f"({panel}) {ds}",loc="left",
                     fontweight="bold",fontsize=STYLE["title_size"])

    for j in range(len(available),4): axes[j].set_visible(False)

    pass
    fig.tight_layout(h_pad=1.5,w_pad=1.0)
    save_fig(fig,"Fig2_signature_heatmaps")


# ── Figure 3：气泡图（改进归一化，固定参考大小）──────────────────────────────
def plot_fig3(topk_by_ds):
    print("\n[Figure 3] SHAP 气泡图 (UNSW)")
    ds="UNSW"
    if ds not in topk_by_ds: print("  ⚠ 跳过"); return
    df=topk_by_ds[ds].copy()

    # 特征按均值重要度排列（高→低，从上到下）
    feat_imp=df.groupby("feature")["abs_shap"].mean().sort_values(ascending=False)
    feat_order=feat_imp.index.tolist()
    # 攻击类别按均值重要度降序（左→右）
    class_order=df.groupby("class")["abs_shap"].mean().sort_values(ascending=False).index.tolist()

    feat_idx={f:i for i,f in enumerate(feat_order)}
    cls_idx={c:i for i,c in enumerate(class_order)}

    x=[cls_idx[r["class"]] for _,r in df.iterrows()]
    y=[feat_idx[r["feature"]] for _,r in df.iterrows()]

    # 归一化气泡大小：相对于当前数据集最大 SHAP
    max_shap=df["abs_shap"].max()
    sizes=[(v/max_shap)**0.6 * 600 for v in df["abs_shap"]]
    colors=[STYLE["colors"]["blue"] if r["direction"]=="attack"
            else STYLE["colors"]["red"] for _,r in df.iterrows()]

    fig,ax=plt.subplots(figsize=(STYLE["full_width"],3.6))
    ax.scatter(x,y,s=sizes,c=colors,alpha=0.72,
               edgecolors="white",linewidths=0.4,zorder=3)

    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels(class_order,rotation=25,ha="right",fontsize=7)
    ax.set_yticks(range(len(feat_order)))
    ax.set_yticklabels([abbrev(f) for f in feat_order],fontsize=6.5)
    ax.set_xlim(-0.6,len(class_order)-0.4)
    ax.set_ylim(-0.6,len(feat_order)-0.4)
    ax.grid(True,linewidth=0.25,alpha=0.35,zorder=0)
    ax.set_xlabel("Attack class"); ax.set_ylabel("NetFlow feature (abbreviated)")

    # 图例：只保留方向说明（气泡大小已在标题副文字中解释）
    dir_handles=[
        mpatches.Patch(color=STYLE["colors"]["blue"],alpha=0.72,label="↑ Towards attack"),
        mpatches.Patch(color=STYLE["colors"]["red"], alpha=0.72,label="↓ Towards benign"),
    ]
    fig.legend(handles=dir_handles,loc="lower center",
               ncol=2,bbox_to_anchor=(0.5,-0.02),
               fontsize=STYLE["legend_size"],frameon=False,
               columnspacing=2.0,handletextpad=0.5)

    ax.set_title("Figure 3. SHAP feature signatures for NF-UNSW-NB15-v2\n"
                 "(bubble size ∝ |SHAP mean|, normalized to dataset max; color = contribution direction)",
                 fontsize=STYLE["title_size"],loc="left")
    fig.tight_layout()
    save_fig(fig,"FigS1_unsw_bubble")


# ── Figure 4：迁移性矩阵（Spearman 色标用实际数据范围）─────────────────────
def plot_fig4(rq2_data):
    print("\n[Figure 4] 迁移性矩阵热力图")
    classes=["DoS","DDoS","Reconnaissance"]
    available=[c for c in classes if f"jaccard_{c}" in rq2_data]
    if not available: print("  ⚠ 跳过"); return

    fig,axes=plt.subplots(len(available),2,
                          figsize=(STYLE["col_width"]*2+0.6,len(available)*1.9+0.3))
    if len(available)==1: axes=axes[np.newaxis,:]

    for row,cls in enumerate(available):
        jac_mat=rq2_data[f"jaccard_{cls}"]
        spe_mat=rq2_data[f"spearman_{cls}"]
        ds_labels=list(jac_mat.columns)

        for col,(mat,cmap,cbar_label,vmin,vmax) in enumerate([
            (jac_mat,"Blues","Jaccard",0,1),
            # Spearman 用实际数据最大值作上限，提升对比度
            (spe_mat,"Blues","Spearman ρ",0,
             round(spe_mat.values[spe_mat.values<1].max()+0.05,1)),
        ]):
            ax=axes[row,col]
            vals=mat.values.astype(float)
            im=ax.imshow(vals,cmap=cmap,vmin=vmin,vmax=vmax,aspect="equal")

            n=len(ds_labels)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(ds_labels,fontsize=7)
            ax.set_yticklabels(ds_labels,fontsize=7)

            for i in range(n):
                for j in range(n):
                    v=vals[i,j]
                    if i==j:
                        # 对角线：白色方块覆盖，显示"—"
                        ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                                     color="white",zorder=2))
                        ax.text(j,i,"—",ha="center",va="center",
                                fontsize=8,color="#aaaaaa",zorder=3)
                    else:
                        fw="bold"
                        threshold=vmax*0.55
                        tc="white" if v>threshold else "#222222"
                        ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                                fontsize=8,fontweight=fw,color=tc,zorder=3)

            cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            cb.ax.tick_params(labelsize=6)
            cb.set_label(cbar_label,fontsize=6.5)

            panel=chr(ord("a")+row*2+col)
            ax.set_title(f"({panel}) {cls} — {cbar_label}",
                         loc="left",fontsize=STYLE["label_size"],fontweight="bold")

    pass  # caption is given in the paper
    fig.tight_layout()
    save_fig(fig,"Fig4_cross_dataset_similarity")


# ── Figure 5：共同特征（只保留 n≥2 的特征，大幅缩短）────────────────────────
def plot_fig5(rq2_data,topk_by_ds):
    print("\n[Figure 5] DoS 共同特征集合图")

    # 优先用 rq2 的 topk_DoS，否则从 rq1 重建
    if "topk_DoS" in rq2_data:
        topk_df=rq2_data["topk_DoS"]
    else:
        rows=[]
        dos_labels={"UNSW":"DoS","ToN":"dos","BoT":"DoS"}
        for ds,lbl in dos_labels.items():
            if ds in topk_by_ds and lbl:
                sub=topk_by_ds[ds][topk_by_ds[ds]["class"]==lbl]
                for _,r in sub.iterrows():
                    rows.append({"dataset":ds,"rank":r["rank"],"feature":r["feature"]})
        if not rows: print("  ⚠ 跳过"); return
        topk_df=pd.DataFrame(rows)

    datasets=sorted(topk_df["dataset"].unique())
    feat_sets={ds:set(topk_df[topk_df["dataset"]==ds]["feature"]) for ds in datasets}
    all_feats=set().union(*feat_sets.values())

    # 只保留出现 >= 2 次的特征（剔除各数据集独有特征，减少图高）
    feat_count={f:sum(1 for s in feat_sets.values() if f in s) for f in all_feats}
    feat_shared={f:c for f,c in feat_count.items() if c>=2}
    feat_sorted=sorted(feat_shared,key=lambda f:(-feat_shared[f],f))

    fig,axes=plt.subplots(1,2,
                          figsize=(STYLE["full_width"],
                                   max(2.5,len(feat_sorted)*0.35+1.0)),
                          gridspec_kw={"width_ratios":[3,1.4]})
    ax_l,ax_r=axes

    colors_ds=[STYLE["dataset_colors"].get(d,"#888") for d in datasets]

    for y_idx,feat in enumerate(feat_sorted):
        cnt=feat_shared[feat]
        ax_l.barh(y_idx,len(datasets),height=0.65,color="#f4f4f4",zorder=1)
        for x_idx,ds in enumerate(datasets):
            if feat in feat_sets.get(ds,set()):
                ax_l.scatter(x_idx,y_idx,s=110,
                             color=colors_ds[x_idx],
                             marker=STYLE["dataset_markers"].get(ds,"o"),
                             zorder=3,linewidths=0)
            else:
                ax_l.scatter(x_idx,y_idx,s=35,color="#d8d8d8",
                             marker="o",zorder=2,linewidths=0)
        ax_l.text(len(datasets)+0.15,y_idx,f"n={cnt}",
                  va="center",fontsize=6.5,color="#666666")

    ax_l.set_yticks(range(len(feat_sorted)))
    ax_l.set_yticklabels([abbrev(f) for f in feat_sorted],fontsize=7)
    ax_l.set_xticks(range(len(datasets)))
    ax_l.set_xticklabels(datasets,fontsize=8)
    ax_l.set_xlim(-0.5,len(datasets)+0.7)
    ax_l.set_ylim(-0.6,len(feat_sorted)-0.4)
    ax_l.set_xlabel("Dataset"); ax_l.invert_yaxis()
    ax_l.set_title("(a) Shared DoS features (appearing in ≥2 datasets)",
                   loc="left",fontsize=STYLE["label_size"],fontweight="bold")

    leg_h=[plt.scatter([],[],s=90,color=STYLE["dataset_colors"].get(d,"#888"),
                       marker=STYLE["dataset_markers"].get(d,"o"),label=d)
           for d in datasets]
    fig.legend(handles=leg_h,loc="lower center",
               ncol=len(datasets),bbox_to_anchor=(0.35,-0.04),
               fontsize=STYLE["legend_size"],frameon=False,
               columnspacing=1.2,handletextpad=0.3)

    # 右图：两两交集矩阵
    n=len(datasets)
    inter_mat=np.zeros((n,n))
    for i,da in enumerate(datasets):
        for j,db in enumerate(datasets):
            inter_mat[i,j]=(len(feat_sets.get(da,set())&feat_sets.get(db,set()))
                            if i!=j else len(feat_sets.get(da,set())))

    im=ax_r.imshow(inter_mat,cmap="Blues",vmin=0,vmax=10,aspect="equal")
    ax_r.set_xticks(range(n)); ax_r.set_yticks(range(n))
    ax_r.set_xticklabels(datasets,fontsize=7); ax_r.set_yticklabels(datasets,fontsize=7)
    for i in range(n):
        for j in range(n):
            v=int(inter_mat[i,j])
            tc="white" if v>6 else "#222222"
            ax_r.text(j,i,str(v),ha="center",va="center",
                      fontsize=9,fontweight="bold",color=tc)
    cb=plt.colorbar(im,ax=ax_r,fraction=0.046,pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Shared features (out of 10)",fontsize=6)
    ax_r.set_title("(b) Pairwise intersection",
                   loc="left",fontsize=STYLE["label_size"],fontweight="bold")

    pass  # caption is given in the paper
    fig.tight_layout()
    save_fig(fig,"Fig5_dos_feature_overlap")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--rq1-dir",default="./results/rq1_signatures/")
    p.add_argument("--rq2-dir",default="./results/rq2_similarity/")
    p.add_argument("--fig",nargs="+",type=int,default=[1,2,3,4,5])
    return p.parse_args()

def main():
    args=parse_args()
    setup_style()
    np.random.seed(42)

    print(f"\n{'#'*55}")
    print(f"#  论文图表 v2 — 输出到 {OUT_DIR}")
    print(f"{'#'*55}")

    need_rq1=any(f in args.fig for f in [1,2,3])
    metrics   =load_metrics(args.rq1_dir)   if need_rq1 else {}
    shap_mats =load_shap_matrices(args.rq1_dir) if need_rq1 else {}
    topk      =load_topk(args.rq1_dir)      if need_rq1 or 5 in args.fig else {}
    rq2       =load_rq2(args.rq2_dir)       if any(f in args.fig for f in [4,5]) else {}

    if 1 in args.fig: plot_fig1(metrics)
    if 2 in args.fig: plot_fig2(shap_mats)
    if 3 in args.fig: plot_fig3(topk)
    if 4 in args.fig: plot_fig4(rq2)
    if 5 in args.fig: plot_fig5(rq2,topk)

    print(f"\n✓ 完成，输出: {OUT_DIR.resolve()}\n")

if __name__=="__main__":
    main()
