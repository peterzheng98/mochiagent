'''
画正负样本概率分布直方图的代码
输入test parquet的路径，在test parquet文件夹下生成直方图
'''
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_auc_score
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='<PREDICTION_PARQUET>')
args = parser.parse_args()


# 加载中文字体，翻译和缩写
my_font = fm.FontProperties(fname='<FONT_PATH>/NotoSansCJKsc-Regular.otf')
trans = json.load(open('<PROJECT_ROOT>/plot/extract_embed_and_plot_umap/data/dis_trans.json','r'))
abbr_map = json.load(open('<PROJECT_ROOT>/plot/extract_embed_and_plot_umap/data/abbr.json','r'))
def abbr(x:str):
    x = x.replace('c_cls_labels_diag_','').replace('f_cls_labels_diag_','')
    return abbr_map[x.split('_')[1]]+' ('+x.split('_')[0][0].upper()+')'
def full_and_abbr(x:str):
    x = x.replace('c_cls_labels_diag_','').replace('f_cls_labels_diag_','')
    return abbr_map[x.split('_')[1]]+' ('+x.split('_')[0][0].upper()+')'+': '+trans[x.split('_')[1]]+' ('+x.split('_')[0]+')'
def full_form(x):
    x = x.replace('c_cls_labels_diag_','').replace('f_cls_labels_diag_','').replace('c_cls_labels_pregnancy_status','mother_pregnancystatus').replace('f_cls_labels_pregnancy_status','mother_pregnancystatus')
    return trans[x.split('_')[1]]+' ('+x.split('_')[0]+')'

# # === 1. 加载所有 Parquet 文件 ===

# df = pd.read_parquet("output/finetune_mother/pred/pred_version14/test_pred.0.parquet")
# output_dir = "output/finetune_mother/log/lightning_logs/version_14/figs_en"
# df_path = '<PROJECT_ROOT>/output/finetune_mother/pred/pred_version14_history_ehr_9_26/test_pred.0.parquet'
# df_path = "<PROJECT_ROOT>/output/finetune_mother/pred/pred_version_17_history_ehr_only/test_pred.0.parquet"
df_path = args.data_path
df = pd.read_parquet(df_path)
output_dir = os.path.join(os.path.dirname(df_path), "figs_hist_en")
os.makedirs(output_dir, exist_ok=True)
print(f"总样本数: {len(df)}")

# === 2. 找出所有二分类任务（存在 {task}_prob_1 和 {task} 列）===
prob_cols = [col for col in df.columns if col.endswith('_prob_1')]
tasks = [col.replace('_prob_1', '') for col in prob_cols]
valid_tasks = []
for task in tasks:
    if task in df.columns:
        valid_tasks.append(task)
    else:
        print(f"警告: 找到预测列 {task}_prob_1，但未找到标签列 '{task}'，跳过。")

if not valid_tasks:
    raise ValueError("未找到有效的二分类任务（需同时存在 {task}_prob_1 和 {task} 列）")
# valid_tasks = valid_tasks[:1]

print(f"有效二分类任务: {valid_tasks}")
# === 3. 对每个任务绘制正负例预测分布 ===
for i,task in enumerate(valid_tasks):
    prob_col = f"{task}_prob_1"
    label_col = task
    mask_col = f"{task}_mask"

    # 过滤非空行
    mask = df[prob_col].notna() & df[label_col].notna()
    df_sub = df[mask]

    if df_sub.empty:
        print(f"任务 {task} 无有效数据，跳过。")
        continue

    try:
        preds_flat = np.concatenate(df_sub[prob_col].tolist())
        labels_flat = np.concatenate(df_sub[label_col].tolist())
        mask_flat = np.concatenate(df_sub[mask_col].tolist())
    except Exception as e:
        print(f"任务 {task} 展平失败: {e}")
        continue

    # 对齐并过滤非法值
    assert len(preds_flat) == len(labels_flat) == len(mask_flat), "预测与标签长度不一致"
    preds_flat = preds_flat[mask_flat==1.]
    labels_flat = labels_flat[mask_flat==1.]
    finite_mask = np.isfinite(preds_flat)
    preds_flat = preds_flat[finite_mask]
    labels_flat = labels_flat[finite_mask].astype(int)

    # print('labels_flat',labels_flat)

    # 分离正负例
    neg_preds = preds_flat[labels_flat == 0]
    pos_preds = preds_flat[labels_flat == 1]
    # print('neg_preds', np.unique(neg_preds))
    # print('pos_preds', np.unique(pos_preds))
    if len(neg_preds) == 0 and len(pos_preds) == 0:
        print(f"任务 {task} 无有效正/负例，跳过。")
        continue
    auc = roc_auc_score(labels_flat, preds_flat)
    # === 绘图：两个子图，分别显示 label=0 和 label=1 ===
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x_min, x_max = min(min(neg_preds), min(pos_preds) if len(pos_preds) else 0), max(max(neg_preds), max(pos_preds) if len(pos_preds) else 1)
    # Label = 0
    if len(neg_preds) > 0:
        ax0.hist(neg_preds, bins=50, range=(x_min, x_max), color='skyblue', edgecolor='black', linewidth=0.5)
        ax0.set_title(f'Label = 0 (n={len(neg_preds):,})', fontsize=12)
        ax0.grid(True, linestyle='--', alpha=0.6)
        ax0.set_ylabel('Frequency')
    else:
        ax0.text(0.5, 0.5, 'No negative samples', ha='center', va='center', transform=ax0.transAxes)
        ax0.set_title('Label = 0 (n=0)', fontsize=12)
        ax0.set_ylabel('Frequency')
    ax0.set_xlim(x_min, x_max)

    # Label = 1
    if len(pos_preds) > 0:
        ax1.hist(pos_preds, bins=50, range=(x_min, x_max), color='salmon', edgecolor='black', linewidth=0.5)
        ax1.set_title(f'Label = 1 (n={len(pos_preds):,})', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Predicted Probability (P=1)')
    else:
        ax1.text(0.5, 0.5, 'No positive samples', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Label = 1 (n=0)', fontsize=12)
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Predicted Probability (P=1)')

    # 共享 x 轴设置
    ax1.set_xlim(x_min, x_max)

    # 主标题
    mean_diff = (pos_preds.mean() - neg_preds.mean()) if (len(pos_preds) > 0 and len(neg_preds) > 0) else np.nan
    main_title = f"{full_form(task)} auc: {auc:.3f}"
    if not np.isnan(mean_diff):
        main_title += f" | ΔMean = {mean_diff:.3f}"
    fig.suptitle(main_title, fontsize=14, fontproperties=my_font)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为 suptitle 留出空间

    # 保存
    save_path = os.path.join(output_dir, f'pred_dist_task_{i}_split.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 已保存: {save_path}")
    plt.close()
    
# === 4. 打印详细统计（可选）===
# print("\n=== 详细统计 ===")
# for task in valid_tasks:
#     prob_col = f"{task}_prob_1"
#     label_col = task
#     mask = df[prob_col].notna() & df[label_col].notna()
#     if not mask.any():
#         continue
#     preds_flat = np.concatenate(df.loc[mask, prob_col].tolist())
#     labels_flat = np.concatenate(df.loc[mask, label_col].tolist())
#     finite_mask = np.isfinite(preds_flat)
#     preds_flat = preds_flat[finite_mask]
#     labels_flat = labels_flat[finite_mask].astype(int)

#     neg = preds_flat[labels_flat == 0]
    # pos = preds_flat[labels_flat == 1]

    # print(f"\n任务: {task}")
    # print(f"  负例数量: {len(neg):,} | 预测均值: {neg.mean():.4f} ± {neg.std():.4f}")
    # print(f"  正例数量: {len(pos):,} | 预测均值: {pos.mean():.4f} ± {pos.std():.4f}")
    # print(f"  正负例预测均值差: {pos.mean() - neg.mean():.4f}")