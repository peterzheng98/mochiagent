import pandas as pd
import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file','-f', type=str, default='test.csv')
args = parser.parse_args()
df = pd.read_csv(args.file)

df = df[[col for col in df.columns if ('auc' in col or 'r2' in col)]]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def report(df, row):
    print("\n## 模型性能对比")
    print("| 时期 |", end="")

    # 先收集所有指标名称
    all_metrics = []
    for col in df.columns:
        if col != 'auc_mean' and 'reg' not in col and 'val_mauc' not in col and 'val_mr2' not in col:
            name = col.replace('val_c_cls_labels_diag_', '').replace('val_f_cls_labels_diag_', '').replace('_auc', '')
            if name not in all_metrics:
                all_metrics.append(name)

    # 打印表头
    for metric in all_metrics:
        print(f" {metric} |", end="")
    print()

    # 打印表头分隔线
    print("|------|", end="")
    for _ in all_metrics:
        print("--------|", end="")
    print()

    # 打印数据行
    future_data = ["Future"]
    current_data = ["Current"]

    # 收集数据
    for metric in all_metrics:
        # 查找对应的 future 列
        future_col = None
        if metric == 'overall':
            future_col = 'val_f_cls_labels_diag_auc'
        else:
            future_col = f'val_f_cls_labels_diag_{metric}_auc'
        
        # 查找对应的 current 列  
        current_col = f'val_c_cls_labels_diag_{metric}_auc'
        
        future_value = "N/A"
        current_value = "N/A"
        
        # 获取值（假设只有一行数据）

        if future_col in df.columns:
            future_value = f"{row[future_col]:.3f}"
        if current_col in df.columns:
            current_value = f"{row[current_col]:.3f}"
        
        future_data.append(future_value)
        current_data.append(current_value)
    # 打印 Current 行  
    print("|", end="")
    for item in current_data:
        print(f" {item} |", end="")
    print()
    # 打印 Future 行
    print("|", end="")
    for item in future_data:
        print(f" {item} |", end="")
    print()



# # 打印每列的最大值
# for col in df.columns:
#     print(col, df[col].max())

# 找到所有auc列平均最大值对应的行
# 1. 首先筛选出所有包含'auc'的列
auc_cols = [col for col in df.columns if 'auc' in col.lower()]

if auc_cols:
    print(f"\n找到的AUC列: {auc_cols}")
    
    # 2. 计算每行在auc列上的平均值
    df['auc_mean'] = df[auc_cols].mean(axis=1)
    
    # 3. 找到平均auc最大的行
    max_auc_mean = df['auc_mean'].max()
    max_auc_row = df[df['auc_mean'] == max_auc_mean]
    
    print(f"\nAUC列平均最大值: {max_auc_mean:.4f}")
    print(f"对应的行索引: {max_auc_row.index.tolist()}")
    
    # 4. 打印该行的所有信息
    print(f"\n平均AUC最大值对应的行数据:")
    print("=" * 80)
    for idx, row in max_auc_row.iterrows():
        print(f"行 {idx}:")
        for col in df.columns:
            if col != 'auc_mean' and 'reg' not in col:  # 不打印临时列
                name = col.replace('val_c_cls_labels_diag_','current_').replace('val_f_cls_labels_diag','future').replace('_auc','')
                print(f"  {name:<30}: {row[col]:.3f}")
        print("-" * 40)
        # report(df, row)
        break   
    # # 5. 如果有多个行具有相同的最大平均值，打印所有
    # if len(max_auc_row) > 1:
    #     print(f"注意: 有 {len(max_auc_row)} 行具有相同的平均AUC最大值")
        
else:
    print("\n没有找到包含'auc'的列")

# 可选：同样处理r2列
r2_cols = [col for col in df.columns if 'r2' in col.lower()]
if r2_cols:
    print(f"\n找到的R2列: {r2_cols}")
    
    df['r2_mean'] = df[r2_cols].mean(axis=1)
    max_r2_mean = df['r2_mean'].max()
    max_r2_row = df[df['r2_mean'] == max_r2_mean]
    
    print(f"R2列平均最大值: {max_r2_mean:.4f}")
    print(f"对应的行索引: {max_r2_row.index.tolist()}")
    
    print(f"\n平均R2最大值对应的行数据:")
    print("=" * 80)
    for idx, row in max_r2_row.iterrows():
        print(f"行 {idx}:")
        for col in df.columns:
            if col not in ['auc_mean', 'r2_mean'] and 'auc' not in col:  # 不打印临时列
                name = col.replace('val_c_reg_labels_reg_','')
                print(f"  {name:<10}: {row[col]:.3f}")
        print("-" * 40)
        break  # 只打印第一行