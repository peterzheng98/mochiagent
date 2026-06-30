import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import json
# === 字体 ===
my_font = fm.FontProperties(fname='<FONT_PATH>/NotoSansCJKsc-Regular.otf')
trans = json.load(open('<PROJECT_ROOT>/plot/extract_embed_and_plot_umap/data/dis_trans.json','r'))
abbr_map = json.load(open('<PROJECT_ROOT>/plot/extract_embed_and_plot_umap/data/abbr.json','r'))


def abbr(x: str):
    x = x.replace('c_cls_labels_diag_','')
    return abbr_map[x.split('_')[1]]+' ('+x.split('_')[0][0].upper()+')'


def full_and_abbr(x: str):
    x = x.replace('c_cls_labels_diag_','')
    return abbr_map[x.split('_')[1]]+' ('+x.split('_')[0][0].upper()+')'+': '+trans[x.split('_')[1]]+' ('+x.split('_')[0]+')'


def full_form(x: str):
    x = x.replace('val_','').replace('c_cls_labels_diag_','').replace('f_cls_labels_diag_','').replace('c_cls_labels_pregnancy_status','mother_pregnancystatus').replace('f_cls_labels_pregnancy_status','mother_pregnancystatus')
    # print(x)
    parts = x.split('_')
    if len(parts) < 2:
        return x
    dis_key = parts[1]
    prefix = parts[0]
    dis_name = trans.get(dis_key, dis_key)
    return dis_name + ' (' + prefix + ')'


def compare_metrics(files, out_csv="comparison.csv", plot_file="comparison.png"):
    """Compare multiple metrics CSV files.

    Each file should contain a header row and numeric values. The script will:
    1. Read each file and take the last row.
    2. Identify the set of column names common to all files.
    3. Produce a new dataframe where each row corresponds to a file and
       each column is one of the shared metric names. The values are taken
       from the last row of the original file.
    4. Save the resulting table to a CSV and produce a bar plot.

    Args:
        files (list[str]): Paths to the metrics CSV files to compare.
        out_csv (str): Path to the output CSV file that will contain the
            comparison table.
        plot_file (str): Path to the output PNG file where the bar chart
            will be stored.
    """

    dfs = []
    names = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            print(f"Warning: file {f} is empty, skipping.")
            continue
        last = df.iloc[[-1]].copy()
        # flatten to series
        last.index = [f]
        dfs.append(last)
        names.append(f)

    if not dfs:
        raise ValueError("No valid dataframes found")

    # concatenate along index
    combined = pd.concat(dfs, axis=0, sort=False)
    # keep only columns present in all rows
    shared_cols = combined.columns[combined.notna().all()]
    auc_cols = [col for col in shared_cols if col.lower().endswith('_auc') and 'val_f_cls' in col.lower()]
    comparison = combined[auc_cols]
    transformed_cols = {col: full_form(col) for col in comparison.columns}
    comparison_renamed = comparison.rename(columns=transformed_cols)
    comparison_renamed.T.to_csv(out_csv)
    print(f"Saved comparison csv to {out_csv}")

    # construct bar plot: each file as a group
    ax = comparison_renamed.plot.bar(rot=45, figsize=(10, 6))
    ax.set_ylabel("Value")
    ax.set_title("Metrics comparison")
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Saved bar chart to {plot_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_metrics.py file1.csv file2.csv [...]")
        sys.exit(1)

    files = sys.argv[1:]
    compare_metrics(files)
