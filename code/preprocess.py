#!/usr/bin/env python3
"""
NFv2 数据集预处理管道（多数据集版）
第二篇NIDS论文 — 数据预处理阶段 Step 2

功能：
  1. 验证 MAX_IP_PKT_LEN 与 LONGEST_FLOW_PKT 是否完全重复
  2. 执行各数据集专属预处理（排除低样本类别、分层采样）
  3. 保存处理后的 CSV 和特征列名文件，供后续 SHAP 实验使用

用法：
  python code/preprocess.py --data ./data/raw/NF-UNSW-NB15-v2.csv
  python code/preprocess.py --data ./data/raw/NF-BoT-IoT-v2.csv --skip-dup-check
  python code/preprocess.py --data ./data/raw/NF-UNSW-NB15-v2.csv --no-save
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 全局配置（不依赖数据集的共享决策）
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_CONFIG = {
    # 始终排除：IP 地址标识符（不可泛化，跨网络迁移无意义）
    "drop_identifier_cols": ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],

    # 重复列：四个数据集均已验证 MAX_IP_PKT_LEN ≡ LONGEST_FLOW_PKT
    "duplicate_pair": ("MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT"),
    "duplicate_keep": "LONGEST_FLOW_PKT",   # 语义更直观

    # 标签列名（四个数据集统一）
    "binary_label_col": "Label",
    "multi_label_col":  "Attack",

    # 采样目标行数（大数据集缩减到此规模）
    "sample_target": 2_000_000,

    # Benign 占目标行数的上限比例
    # 设 0.5 的理由：SHAP 攻击签名分析需要充足的攻击样本，
    # 若 Benign 无上限，CIC-IDS2018（Benign 1663万）会挤占全部配额
    "max_benign_ratio": 0.5,

    # 每个攻击类的保底样本数（类别总数不足则全取）
    # 保证小类别（如 mitm 7,723）不会因比例分配被压缩到无法分析
    "min_per_class": 10_000,

    # 采样随机种子（保证可复现）
    "sample_seed": 42,
}


# ══════════════════════════════════════════════════════════════════════════════
# 各数据集专属配置
# key = CSV 文件名（无后缀），与 data_path.stem 对应
# ══════════════════════════════════════════════════════════════════════════════
DATASET_CONFIG = {
    "NF-UNSW-NB15-v2": {
        # 排除原因：164 条，无法产生可靠 SHAP 签名
        "exclude_classes": ["Worms"],
        # low-confidence：保留但论文中标注
        "low_confidence_classes": ["Shellcode"],
        # 全量（239 万，在可接受范围内）
        "do_sample": False,
    },
    "NF-CSE-CIC-IDS2018-v2": {
        # 排除原因：均 < 5000 条
        "exclude_classes": [
            "Brute Force -Web",     # 2,143
            "Brute Force -XSS",     #   927
            "SQL Injection",        #   432
            "DDOS attack-LOIC-UDP", # 2,112
        ],
        "low_confidence_classes": [],
        # 1889 万 → 分层采样至 200 万
        "do_sample": True,
    },
    "NF-ToN-IoT-v2": {
        # 排除原因：3,425 条 < 5000
        "exclude_classes": ["ransomware"],
        "low_confidence_classes": [],
        # 1694 万 → 分层采样至 200 万
        "do_sample": True,
    },
    "NF-BoT-IoT-v2": {
        # 排除原因：2,431 条 < 5000
        "exclude_classes": ["Theft"],
        "low_confidence_classes": [],
        # 3776 万 → 分层采样至 200 万
        # Benign 仅 135,037 条，全部保留（在 200 万配额内按比例分配攻击类）
        "do_sample": True,
    },
}

# 未在上表中列出的数据集使用此默认配置
DEFAULT_DATASET_CONFIG = {
    "exclude_classes": [],
    "low_confidence_classes": [],
    "do_sample": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="NFv2 预处理管道（多数据集版）")
    parser.add_argument("--data", required=True, help="输入 CSV 路径")
    parser.add_argument("--out", default="./data/processed/",
                        help="输出目录（默认 ./data/processed/）")
    parser.add_argument("--no-save", action="store_true",
                        help="只做验证和预览，不保存文件")
    parser.add_argument("--skip-dup-check", action="store_true",
                        help="跳过重复列逐行验证（大数据集已验证时使用，节省时间）")
    return parser.parse_args()


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1：验证重复列
# ══════════════════════════════════════════════════════════════════════════════
def verify_duplicate_columns(df: pd.DataFrame) -> bool:
    """
    验证 MAX_IP_PKT_LEN 与 LONGEST_FLOW_PKT 是否逐行完全相同。
    返回 True 表示确认重复，应删除其中一列。
    大数据集可用 --skip-dup-check 跳过，直接返回 True（已预验证）。
    """
    section("Step 1：验证重复列 MAX_IP_PKT_LEN vs LONGEST_FLOW_PKT")

    col_a, col_b = GLOBAL_CONFIG["duplicate_pair"]

    if col_a not in df.columns or col_b not in df.columns:
        print(f"  ⚠ 其中一列不存在，跳过验证，两列均保留")
        return False

    equal_mask = df[col_a] == df[col_b]
    n_equal = equal_mask.sum()
    n_total = len(df)
    n_diff  = n_total - n_equal

    print(f"  完全相等的行: {n_equal:,d} / {n_total:,d}  ({n_equal/n_total*100:.4f}%)")
    print(f"  存在差异的行: {n_diff:,d}")

    if n_diff == 0:
        keep = GLOBAL_CONFIG["duplicate_keep"]
        drop = col_a if keep == col_b else col_b
        print(f"\n  ✓ 两列完全相同 → 删除 '{drop}'，保留 '{keep}'")
        return True
    else:
        diff_sample = df[~equal_mask][[col_a, col_b]].head(5)
        print(f"\n  ⚠ 两列存在差异，两列均保留。差异样本（前5行）：")
        print(diff_sample.to_string())
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Step 2：核心预处理管道
# ══════════════════════════════════════════════════════════════════════════════
def allocate_quota(counts: dict, budget: int, floor: int) -> dict:
    """
    水填充式配额分配：在 budget 总量内为各类别分配样本数。

    规则：
    1. 每类先分配保底量 min(该类实际数量, floor)
    2. 剩余预算按各类原始规模比例分配
    3. 任何类分配量不超过其实际样本数；被"填满"的类退出下一轮，
       其剩余份额重新分给还有余量的类（迭代直到预算用尽或全部填满）

    返回：{类别名: 分配样本数}
    """
    alloc = {c: min(n, floor) for c, n in counts.items()}
    used  = sum(alloc.values())

    if used >= budget:
        # 保底量已超预算，直接返回保底分配（宁可略超也要保证小类可分析）
        return alloc

    remaining = budget - used
    active = {c for c in counts if alloc[c] < counts[c]}

    while remaining > 0 and active:
        weight_total = sum(counts[c] for c in active)
        distributed  = 0

        # 从小类开始分配，避免大类抢光预算后小类拿不到
        for c in sorted(active, key=lambda x: counts[x]):
            share   = int(remaining * counts[c] / weight_total)
            headroom = counts[c] - alloc[c]
            add     = min(share, headroom)
            alloc[c] += add
            distributed += add

        if distributed == 0:
            break

        remaining -= distributed
        active = {c for c in active if alloc[c] < counts[c]}

    return alloc


def stratified_sample(df: pd.DataFrame, attack_col: str, target: int,
                      seed: int, benign_ratio: float, floor: int,
                      benign_label: str = "Benign") -> pd.DataFrame:
    """
    分层采样：将总样本缩减到 target 行。

    策略（针对 SHAP 攻击签名分析优化）：
    1. Benign 上限 = target * benign_ratio，超出则采样，不足则全保留
    2. 剩余预算分配给攻击类，每类保底 floor 条（不足则全取）
    3. 保底后的余量按各攻击类原始规模比例分配（水填充）

    这样确保：
    - CIC-IDS2018（Benign 1663万）：Benign 截断至 100 万，攻击类拿到 100 万
    - BoT-IoT（Benign 13.5万）：Benign 全保留，攻击类拿到 186.5 万
    - 小类别（如 mitm 7,723）不会被比例分配压缩到无法分析
    """
    n_total = len(df)
    if n_total <= target:
        print(f"  [采样] 当前 {n_total:,d} 行 ≤ 目标 {target:,d}，无需采样")
        return df

    class_counts = df[attack_col].value_counts().to_dict()

    # ── 1. Benign 配额 ────────────────────────────────────────
    n_benign     = class_counts.get(benign_label, 0)
    benign_cap   = int(target * benign_ratio)
    benign_take  = min(n_benign, benign_cap)

    # ── 2. 攻击类配额 ─────────────────────────────────────────
    attack_counts = {c: n for c, n in class_counts.items() if c != benign_label}
    attack_budget = target - benign_take

    print(f"  [采样] {n_total:,d} 行 → 目标 {target:,d} 行")
    print(f"         Benign 上限比例 {benign_ratio:.0%} → 上限 {benign_cap:,d}")
    print(f"         攻击类保底 {floor:,d} 条/类")
    print()

    alloc = allocate_quota(attack_counts, attack_budget, floor)

    # ── 3. 执行采样 ───────────────────────────────────────────
    parts = []

    # Benign
    benign_df = df[df[attack_col] == benign_label]
    if benign_take < n_benign:
        benign_df = benign_df.sample(n=benign_take, random_state=seed)
        tag = ""
    else:
        tag = "  (全部保留)"
    parts.append(benign_df)
    print(f"         {benign_label:>30s}: {n_benign:>10,d} → {benign_take:>8,d}{tag}")

    # 攻击类
    for cls in sorted(alloc, key=lambda x: -attack_counts[x]):
        n_cls  = attack_counts[cls]
        n_take = alloc[cls]
        cls_df = df[df[attack_col] == cls]
        if n_take < n_cls:
            cls_df = cls_df.sample(n=n_take, random_state=seed)
            tag = ""
        else:
            tag = "  (全部保留)"
        parts.append(cls_df)
        print(f"         {str(cls):>30s}: {n_cls:>10,d} → {n_take:>8,d}{tag}")

    result = pd.concat(parts, ignore_index=True)
    print(f"\n  [采样] 完成: {len(result):,d} 行")
    return result


def run_pipeline(df: pd.DataFrame, drop_duplicate: bool,
                 dataset_name: str) -> dict:
    """
    执行预处理管道，返回包含特征矩阵和标签的字典。
    """
    section("Step 2：执行预处理管道")

    gcfg = GLOBAL_CONFIG
    dcfg = DATASET_CONFIG.get(dataset_name, DEFAULT_DATASET_CONFIG)

    print(f"  数据集配置: {dataset_name}")
    print(f"  排除类别:   {dcfg['exclude_classes'] or '（无）'}")
    print(f"  低置信类别: {dcfg['low_confidence_classes'] or '（无）'}")
    sample_info = f"是，目标 {gcfg['sample_target']:,d} 行" if dcfg['do_sample'] else "否（全量）"
    print(f"  分层采样:   {sample_info}")

    # ── 2a. 删除标识符列 ─────────────────────────────────────────
    existing_drop = [c for c in gcfg["drop_identifier_cols"] if c in df.columns]
    df = df.drop(columns=existing_drop)
    print(f"\n[2a] 删除标识符列: {existing_drop}  →  剩余 {df.shape[1]} 列")

    # ── 2b. 删除重复列 ───────────────────────────────────────────
    col_a, col_b = gcfg["duplicate_pair"]
    keep    = gcfg["duplicate_keep"]
    drop_dup = col_a if keep == col_b else col_b

    if drop_duplicate and drop_dup in df.columns:
        df = df.drop(columns=[drop_dup])
        print(f"[2b] 删除重复列: '{drop_dup}'  →  剩余 {df.shape[1]} 列")
    else:
        print(f"[2b] 重复列验证未通过或列不存在，两列均保留")

    # ── 2c. 排除低样本攻击类别 ───────────────────────────────────
    attack_col = gcfg["multi_label_col"]
    exclude    = dcfg["exclude_classes"]

    if exclude:
        before = len(df)
        df = df[~df[attack_col].isin(exclude)].reset_index(drop=True)
        after  = len(df)
        print(f"[2c] 排除低样本类别 {exclude}:")
        print(f"     {before:,d} → {after:,d} 行（移除 {before - after:,d} 行）")
    else:
        print(f"[2c] 无需排除类别")

    # ── 2d. 分层采样（仅大数据集）───────────────────────────────
    if dcfg["do_sample"]:
        df = stratified_sample(
            df, attack_col,
            target=gcfg["sample_target"],
            seed=gcfg["sample_seed"],
            benign_ratio=gcfg["max_benign_ratio"],
            floor=gcfg["min_per_class"],
        )
    else:
        print(f"[2d] 全量保留: {len(df):,d} 行")

    # ── 2e. 分离特征与标签 ────────────────────────────────────────
    label_cols   = [gcfg["binary_label_col"], gcfg["multi_label_col"]]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X       = df[feature_cols].copy()
    y_bin   = df[gcfg["binary_label_col"]].copy()
    y_multi = df[gcfg["multi_label_col"]].copy()

    print(f"\n[2e] 特征矩阵 X:      {X.shape}")
    print(f"     二分类标签 y_bin:  {y_bin.shape}  值域={sorted(y_bin.unique())}")
    print(f"     多分类标签 y_multi:{y_multi.shape}  类别数={y_multi.nunique()}")

    # ── 2f. 最终类别分布 ─────────────────────────────────────────
    section("Step 2f：最终攻击类别分布")
    vc    = y_multi.value_counts().sort_values(ascending=False)
    total = len(y_multi)
    low_conf = dcfg["low_confidence_classes"]

    for cls, cnt in vc.items():
        flag = "  ← ⚠ low-confidence" if cls in low_conf else ""
        print(f"  {str(cls):>30s}: {cnt:>10,d}  ({cnt/total*100:6.2f}%){flag}")
    print(f"  {'Total':>30s}: {total:>10,d}")

    # ── 2g. 数据类型检查 ─────────────────────────────────────────
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"\n⚠ 非数值特征列（需额外处理）: {non_num}")
    else:
        print(f"\n✓ 所有特征均为数值类型，可直接送入 XGBoost/SHAP")

    return {
        "X":            X,
        "y_bin":        y_bin,
        "y_multi":      y_multi,
        "feature_cols": feature_cols,
        "n_features":   len(feature_cols),
        "n_samples":    len(X),
        "classes":      sorted(y_multi.unique()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3：保存结果
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(report: dict, out_dir: Path, dataset_name: str):
    section("Step 3：保存预处理结果")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = report["X"].copy()
    out_df["Label"]  = report["y_bin"].values
    out_df["Attack"] = report["y_multi"].values

    out_csv = out_dir / f"{dataset_name}_processed.csv"
    out_df.to_csv(out_csv, index=False)
    size_mb = out_csv.stat().st_size / 1024 / 1024
    print(f"  ✓ CSV:    {out_csv}  ({size_mb:.1f} MB)")
    print(f"            {out_df.shape[0]:,d} 行 × {out_df.shape[1]} 列")
    print(f"            特征数: {report['n_features']}  |  类别: {report['classes']}")

    feat_file = out_dir / f"{dataset_name}_feature_cols.txt"
    feat_file.write_text("\n".join(report["feature_cols"]), encoding="utf-8")
    print(f"  ✓ 特征列: {feat_file}")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args      = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"错误: 文件不存在 → {data_path}")
        sys.exit(1)

    dataset_name = data_path.stem   # e.g. "NF-CSE-CIC-IDS2018-v2"

    print(f"\n{'#'*70}")
    print(f"#  数据集: {dataset_name}")
    print(f"#  文件:   {data_path}  ({data_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"{'#'*70}")

    print("正在读取（大文件请耐心等待）...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"读取完成: {df.shape[0]:,d} 行 × {df.shape[1]} 列")

    # Step 1：验证重复列
    if args.skip_dup_check:
        section("Step 1：跳过重复列验证（--skip-dup-check，已预验证）")
        print("  四个数据集均已确认 MAX_IP_PKT_LEN ≡ LONGEST_FLOW_PKT，直接删除")
        is_duplicate = True
    else:
        is_duplicate = verify_duplicate_columns(df)

    # Step 2：预处理管道
    report = run_pipeline(df, drop_duplicate=is_duplicate,
                          dataset_name=dataset_name)

    # Step 3：保存
    if not args.no_save:
        save_outputs(report, Path(args.out), dataset_name)
    else:
        section("Step 3：跳过保存（--no-save 模式）")

    section("完成")
    print(f"  最终规格: {report['n_samples']:,d} 样本 × {report['n_features']} 特征")
    print(f"  攻击类别: {report['classes']}\n")


if __name__ == "__main__":
    main()