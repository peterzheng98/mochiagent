# EHRFormer

This directory contains EHRFormer training, validation, baseline, and plotting utilities for longitudinal EHR tasks.

## Layout

- `pretrain.py`: pretraining entry point for the 1D EHR VAE model.
- `finetune.py`: downstream finetuning entry point for EHRFormer.
- `validate_pretrain.py`: validation-only entry point for pretrained checkpoints.
- `validate_finetune.py`: validation-only entry point for finetuned checkpoints.
- `train_lstm.py`: LSTM baseline for parquet inputs with `cls_` and `reg_` target columns.
- `compare_metrics.py`: compare metric CSV files and export a CSV/PNG summary.
- `ehrformer.py`: model definitions.
- `scripts/`: Lightning data modules, model wrappers, and multistage task configuration helpers.
- `utils/`: training utilities, scheduler helper, logging callbacks, FlashAttention patching, and plotting helpers.

## Local Path Placeholders

Do not commit machine-specific absolute paths such as user home directories or mounted data roots.
Use placeholders in examples and config templates, then replace them locally before running:

- `<PROJECT_ROOT>`: repository or experiment root.
- `<DATA_ROOT>`: local EHR parquet/cache directory.
- `<OUTPUT_ROOT>`: local output directory for logs, checkpoints, metrics, and predictions.
- `<CHECKPOINT_PATH>`: model checkpoint to load.
- `<FONT_PATH>`: local font directory used by matplotlib plots.
- `<PREDICTION_PARQUET>`: prediction parquet used by plotting scripts.

Example:

```bash
python utils/plot/plot_prob_hist.py \
  --data_path "<PROJECT_ROOT>/output/finetune_mother/pred/test_pred.0.parquet"
```

## Configuration

The training scripts expect JSON config files under `configs/` by default. This repository snapshot does not include those config files, so create them locally or pass a path with `--config`.

Common config fields include:

- `df_path` or `df_paths`: input parquet/cache path or paths.
- `feat_info_path`: JSON describing category and float feature metadata.
- `task_info_path`: JSON describing downstream task metadata.
- `output_dir`: output root used by the CSV logger and checkpoint callbacks.
- `pred_folder`: prediction subdirectory, resolved under the logger directory at runtime.
- `ckpt_path`: checkpoint for resume/validation, or `null`.
- `ehr_emb_ckpt_path`: optional pretrained EHR embedding checkpoint for finetuning.
- `train_folds`, `valid_folds`, `test_folds`: dataset split fold IDs.

Prefer placeholders in shared config templates:

```json
{
  "df_paths": ["<DATA_ROOT>/precessed2D"],
  "feat_info_path": "<DATA_ROOT>/info_feat.json",
  "task_info_path": "<DATA_ROOT>/info_task.json",
  "output_dir": "<OUTPUT_ROOT>/finetune_mother",
  "pred_folder": "pred",
  "ckpt_path": null
}
```

## Usage

Run pretraining:

```bash
python pretrain.py --config configs/pretrain_mother_labonly.json
```

Run finetuning:

```bash
python finetune.py --config configs/finetune_mother_labonly.json
```

Run validation from a checkpoint:

```bash
python validate_finetune.py --config configs/finetune_mother_labonly.json
```

Run the LSTM baseline:

```bash
python train_lstm.py \
  --input_path "<DATA_ROOT>/data_mother.parquet" \
  --output_root "<OUTPUT_ROOT>/lstm" \
  --epochs 10 \
  --batch_size 16
```

`train_lstm.py` also supports `--config <CONFIG_JSON>` and will override matching command-line arguments from that JSON.

## Outputs

Training and validation scripts write Lightning CSV logs, checkpoints, metrics, and prediction files under `output_dir`. The final prediction directory is built by joining the CSV logger directory with `pred_folder`, so keep `pred_folder` relative in shared configs.

`train_lstm.py` creates an auto-incremented `version<N>` directory under `output_root` and saves metrics, epoch-level metric parquet files, and the resolved run config.
