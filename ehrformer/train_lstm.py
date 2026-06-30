#!/usr/bin/env python3
"""
LSTM baseline for tabular EHR sequences.

- Predict columns starting with "cls_" (binary classification)
- Regress columns starting with "reg_" (continuous regression)
- Save outputs to output/lstm/versionX (auto-increment)
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import diskcache
from tqdm import tqdm


TQDM_KW = {
    "ncols": 100,
    "dynamic_ncols": False,
    "mininterval": 1.0,
    "maxinterval": 1.0,
    "smoothing": 0.0,
    "leave": True,
}
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, r2_score, precision_recall_curve


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sha_split(pid: str, n_fold: int = 10) -> int:
    return int(hashlib.sha256(str(pid).encode("utf-8")).hexdigest(), 16) % n_fold


def next_version(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    versions = []
    for item in output_root.iterdir():
        if item.is_dir() and item.name.startswith("version"):
            suffix = item.name.replace("version", "")
            if suffix.isdigit():
                versions.append(int(suffix))
    next_id = max(versions) + 1 if versions else 0
    version_dir = output_root / f"version{next_id}"
    version_dir.mkdir(parents=True, exist_ok=False)
    return version_dir


def normalize_features(train_feats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([x.reshape(-1, x.shape[-1]) for x in train_feats], axis=0)
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std[std == 0] = 1.0
    return mean, std


def infer_feature_cols(df: pd.DataFrame, cls_cols: List[str], reg_cols: List[str]) -> List[str]:
    drop_cols = set(cls_cols + reg_cols)
    drop_cols.update(["pid", "vid", "visit_id", "date", "visit_date"])

    feature_cols = []
    for col in df.columns:
        if col in drop_cols:
            continue
        if df[col].dtype.kind in "ifb":
            feature_cols.append(col)
        elif df[col].dtype == "object":
            # Encode object columns as category codes
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)
            feature_cols.append(col)
        elif str(df[col].dtype).startswith("category"):
            df[col] = df[col].cat.codes.replace(-1, np.nan)
            feature_cols.append(col)
    return feature_cols


def _is_gfe_precessed2d(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.parquet").exists()


def _load_gfe_info_task(path: Path) -> Tuple[List[str], List[str], Dict[str, Dict]]:
    info_path = path / "info_task.json"
    if not info_path.exists():
        return [], [], {}
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    cls_cols = info.get("category_cols", []) or []
    reg_info = info.get("float_cols", {}) or {}
    reg_cols = list(reg_info.keys())
    return cls_cols, reg_cols, reg_info


def build_sequences_from_gfe(precessed2d_dir: Path, normalize_reg: bool = True) -> Tuple[List[Dict[str, np.ndarray]], List[str], List[str]]:
    meta_path = precessed2d_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {precessed2d_dir}")

    base_dir = precessed2d_dir.parent
    cls_cols, reg_cols, reg_info = _load_gfe_info_task(base_dir)
    reg_means = np.array([reg_info.get(k, {}).get("mean", 0.0) for k in reg_cols], dtype=np.float32)
    reg_stds = np.array([reg_info.get(k, {}).get("std", 1.0) for k in reg_cols], dtype=np.float32)
    reg_stds[reg_stds == 0] = 1.0

    meta = pd.read_parquet(meta_path)
    cache = diskcache.Cache(directory=precessed2d_dir, eviction_policy="none")

    records = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading precessed2D", **TQDM_KW):
        pid = row["pid"]
        valid_mask = row.get("valid_mask")
        if valid_mask is None:
            raise ValueError("metadata.parquet must include valid_mask column")
        valid_mask = np.asarray(valid_mask).astype(bool)
        length = int(valid_mask.sum())

        cached = cache[pid]
        if "tokenized_category_feats" in cached:
            cat_feats = cached.get("tokenized_category_feats")
        else:
            cat_feats = cached.get("category_feats")

        if "tokenized_float_feats" in cached:
            float_feats = cached.get("tokenized_float_feats")
        else:
            float_feats = cached.get("float_feats")

        if cat_feats is None:
            cat_feats = np.zeros((0, len(valid_mask)), dtype=np.float32)
        if float_feats is None:
            float_feats = np.zeros((0, len(valid_mask)), dtype=np.float32)

        cat_feats = np.asarray(cat_feats, dtype=np.float32)
        float_feats = np.asarray(float_feats, dtype=np.float32)

        feats = np.concatenate([cat_feats, float_feats], axis=0).T
        feats = feats[valid_mask]

        cls_labels = cached.get("c_cls_labels")
        if cls_labels is None:
            cls_labels = np.zeros((0, len(valid_mask)), dtype=np.float32)
        cls_labels = np.asarray(cls_labels, dtype=np.float32).T
        cls_labels = cls_labels[valid_mask]

        reg_labels = cached.get("c_reg_labels")
        if reg_labels is None:
            reg_labels = np.zeros((0, len(valid_mask)), dtype=np.float32)
        reg_labels = np.asarray(reg_labels, dtype=np.float32).T
        reg_labels = reg_labels[valid_mask]

        if normalize_reg and reg_labels.size and reg_means.size:
            reg_labels = (reg_labels - reg_means) / reg_stds

        cls_mask = cls_labels != -1
        reg_mask = ~np.isnan(reg_labels)

        cls_labels = np.where(cls_mask, cls_labels, 0.0)
        reg_labels = np.where(reg_mask, reg_labels, 0.0)

        record = {
            "pid": pid,
            "features": feats,
            "cls_labels": cls_labels,
            "reg_labels": reg_labels,
            "cls_mask": cls_mask,
            "reg_mask": reg_mask,
            "length": length,
        }
        if "dataset_fold10" in row:
            record["fold"] = int(row["dataset_fold10"])
        records.append(record)

    cache.close()
    return records, cls_cols, reg_cols


class SequenceDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        cls_cols: List[str],
        reg_cols: List[str],
    ):
        self.records = records
        self.cls_cols = cls_cols
        self.reg_cols = reg_cols

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.records[idx]


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["length"] for x in batch)
    feat_dim = batch[0]["features"].shape[-1]
    cls_dim = batch[0]["cls_labels"].shape[-1] if batch[0]["cls_labels"].size else 0
    reg_dim = batch[0]["reg_labels"].shape[-1] if batch[0]["reg_labels"].size else 0

    feats = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    cls_labels = None
    cls_mask = None
    if cls_dim > 0:
        cls_labels = torch.zeros(len(batch), max_len, cls_dim, dtype=torch.float32)
        cls_mask = torch.zeros(len(batch), max_len, cls_dim, dtype=torch.bool)

    reg_labels = None
    reg_mask = None
    if reg_dim > 0:
        reg_labels = torch.zeros(len(batch), max_len, reg_dim, dtype=torch.float32)
        reg_mask = torch.zeros(len(batch), max_len, reg_dim, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = item["length"]
        feats[i, :length] = torch.from_numpy(item["features"]).float()
        mask[i, :length] = True

        if cls_dim > 0:
            cls_labels[i, :length] = torch.from_numpy(item["cls_labels"]).float()
            cls_mask[i, :length] = torch.from_numpy(item["cls_mask"]).bool()

        if reg_dim > 0:
            reg_labels[i, :length] = torch.from_numpy(item["reg_labels"]).float()
            reg_mask[i, :length] = torch.from_numpy(item["reg_mask"]).bool()

    return {
        "features": feats,
        "lengths": lengths,
        "mask": mask,
        "cls_labels": cls_labels,
        "cls_mask": cls_mask,
        "reg_labels": reg_labels,
        "reg_mask": reg_mask,
    }


class LSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, cls_dim: int, reg_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.cls_head = nn.Linear(hidden_dim, cls_dim) if cls_dim > 0 else None
        self.reg_head = nn.Linear(hidden_dim, reg_dim) if reg_dim > 0 else None

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=x.size(1))
        cls_logits = self.cls_head(out) if self.cls_head is not None else None
        reg_preds = self.reg_head(out) if self.reg_head is not None else None
        return cls_logits, reg_preds


def build_sequences(df: pd.DataFrame, cls_cols: List[str], reg_cols: List[str], feature_cols: List[str]) -> List[Dict[str, np.ndarray]]:
    records = []
    if "date" in df.columns:
        df = df.sort_values(["pid", "date"])
    elif "visit_date" in df.columns:
        df = df.sort_values(["pid", "visit_date"])

    for pid, sub_df in df.groupby("pid"):
        feats = sub_df[feature_cols].to_numpy(dtype=np.float32)
        cls_labels = sub_df[cls_cols].to_numpy(dtype=np.float32) if cls_cols else np.zeros((len(sub_df), 0), dtype=np.float32)
        reg_labels = sub_df[reg_cols].to_numpy(dtype=np.float32) if reg_cols else np.zeros((len(sub_df), 0), dtype=np.float32)

        cls_mask = ~np.isnan(cls_labels) if cls_cols else np.zeros_like(cls_labels, dtype=bool)
        reg_mask = ~np.isnan(reg_labels) if reg_cols else np.zeros_like(reg_labels, dtype=bool)

        feats = np.nan_to_num(feats, nan=0.0)
        cls_labels = np.nan_to_num(cls_labels, nan=0.0)
        reg_labels = np.nan_to_num(reg_labels, nan=0.0)

        fold = None
        if "dataset_fold10" in sub_df.columns and len(sub_df["dataset_fold10"].unique()) == 1:
            fold = int(sub_df["dataset_fold10"].iloc[0])

        record = {
            "pid": pid,
            "features": feats,
            "cls_labels": cls_labels,
            "reg_labels": reg_labels,
            "cls_mask": cls_mask,
            "reg_mask": reg_mask,
            "length": len(sub_df),
        }
        if fold is not None:
            record["fold"] = fold
        records.append(record)
    return records


def split_records(
    records: List[Dict[str, np.ndarray]],
    train_folds: List[int] | None = None,
    valid_folds: List[int] | None = None,
    test_folds: List[int] | None = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    train, val, test = [], [], []
    use_custom = train_folds is not None or valid_folds is not None or test_folds is not None
    for rec in records:
        if "fold" in rec and rec["fold"] is not None:
            fold = int(rec["fold"])
        else:
            fold = sha_split(rec["pid"], n_fold=10)
        if use_custom:
            if train_folds is not None and fold in train_folds:
                train.append(rec)
            if valid_folds is not None and fold in valid_folds:
                val.append(rec)
            if test_folds is not None and fold in test_folds:
                test.append(rec)
        else:
            if fold <= 7:
                train.append(rec)
            elif fold == 8:
                val.append(rec)
            else:
                test.append(rec)
    return train, val, test


def compute_metrics(
    cls_logits: np.ndarray,
    cls_labels: np.ndarray,
    cls_mask: np.ndarray,
    reg_preds: np.ndarray,
    reg_labels: np.ndarray,
    reg_mask: np.ndarray,
    cls_cols: List[str],
    reg_cols: List[str],
    optimize_f1: bool = False,
) -> Dict[str, Dict]:
    metrics = {"cls": {"per_target": {}, "macro": {}}, "reg": {"per_target": {}, "macro": {}}}

    if cls_cols:
        aucs, f1s = [], []
        probs = 1 / (1 + np.exp(-cls_logits))
        for i, col in enumerate(cls_cols):
            y_true = cls_labels[:, i][cls_mask[:, i]]
            y_prob = probs[:, i][cls_mask[:, i]]
            if len(np.unique(y_true)) < 2:
                auc = np.nan
            else:
                auc = float(roc_auc_score(y_true, y_prob))
            pos_rate = float(np.mean(y_true)) if len(y_true) else np.nan

            if len(y_true) == 0 or np.sum(y_true) == 0:
                f1 = np.nan
                best_thr = np.nan
            elif optimize_f1:
                precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
                if len(thresholds) == 0:
                    f1 = np.nan
                    best_thr = np.nan
                else:
                    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
                    best_idx = int(np.nanargmax(f1_scores))
                    best_thr = float(thresholds[best_idx])
                    f1 = float(f1_scores[best_idx])
            else:
                y_pred = (y_prob >= 0.5).astype(int)
                f1 = float(f1_score(y_true, y_pred, zero_division=0))
                best_thr = 0.5

            metrics["cls"]["per_target"][col] = {
                "auc": auc,
                "f1": f1,
                "pos_rate": pos_rate,
                "f1_threshold": best_thr,
            }
            if not np.isnan(auc):
                aucs.append(auc)
            if not np.isnan(f1):
                f1s.append(f1)
        metrics["cls"]["macro"]["auc"] = float(np.mean(aucs)) if aucs else np.nan
        metrics["cls"]["macro"]["f1"] = float(np.mean(f1s)) if f1s else np.nan

    if reg_cols:
        r2s = []
        for i, col in enumerate(reg_cols):
            y_true = reg_labels[:, i][reg_mask[:, i]]
            y_pred = reg_preds[:, i][reg_mask[:, i]]
            if len(y_true) < 2:
                r2 = np.nan
            else:
                r2 = float(r2_score(y_true, y_pred))
            metrics["reg"]["per_target"][col] = {"r2": r2}
            if not np.isnan(r2):
                r2s.append(r2)
        metrics["reg"]["macro"]["r2"] = float(np.mean(r2s)) if r2s else np.nan

    return metrics


def collect_outputs(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    cls_cols: List[str],
    reg_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_cls_logits, all_cls_labels, all_cls_mask = [], [], []
    all_reg_preds, all_reg_labels, all_reg_mask = [], [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            cls_labels = batch["cls_labels"].to(device) if batch["cls_labels"] is not None else None
            cls_mask = batch["cls_mask"].to(device) if batch["cls_mask"] is not None else None
            reg_labels = batch["reg_labels"].to(device) if batch["reg_labels"] is not None else None
            reg_mask = batch["reg_mask"].to(device) if batch["reg_mask"] is not None else None

            cls_logits, reg_preds = model(feats, lengths)

            if cls_logits is not None:
                cls_logits_np = cls_logits.cpu().numpy().reshape(-1, len(cls_cols))
                cls_labels_np = cls_labels.cpu().numpy().reshape(-1, len(cls_cols))
                cls_mask_np = cls_mask.cpu().numpy().reshape(-1, len(cls_cols))
                all_cls_logits.append(cls_logits_np)
                all_cls_labels.append(cls_labels_np)
                all_cls_mask.append(cls_mask_np)
            if reg_preds is not None:
                reg_preds_np = reg_preds.cpu().numpy().reshape(-1, len(reg_cols))
                reg_labels_np = reg_labels.cpu().numpy().reshape(-1, len(reg_cols))
                reg_mask_np = reg_mask.cpu().numpy().reshape(-1, len(reg_cols))
                all_reg_preds.append(reg_preds_np)
                all_reg_labels.append(reg_labels_np)
                all_reg_mask.append(reg_mask_np)

    cls_logits = np.concatenate(all_cls_logits, axis=0) if cls_cols else np.zeros((0, 0))
    cls_labels = np.concatenate(all_cls_labels, axis=0) if cls_cols else np.zeros((0, 0))
    cls_mask = np.concatenate(all_cls_mask, axis=0) if cls_cols else np.zeros((0, 0), dtype=bool)

    reg_preds = np.concatenate(all_reg_preds, axis=0) if reg_cols else np.zeros((0, 0))
    reg_labels = np.concatenate(all_reg_labels, axis=0) if reg_cols else np.zeros((0, 0))
    reg_mask = np.concatenate(all_reg_mask, axis=0) if reg_cols else np.zeros((0, 0), dtype=bool)

    return cls_logits, cls_labels, cls_mask, reg_preds, reg_labels, reg_mask


def save_metrics(metrics: Dict, output_dir: Path) -> None:
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    rows = []
    for target, vals in metrics.get("cls", {}).get("per_target", {}).items():
        rows.append({"task": "cls", "target": target, "auc": vals.get("auc"), "f1": vals.get("f1")})
    for target, vals in metrics.get("reg", {}).get("per_target", {}).items():
        rows.append({"task": "reg", "target": target, "r2": vals.get("r2")})
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_folds(value) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return [int(v) for v in value.split(",") if str(v).strip() != ""]
    return None


def _normalize_gpu_ids(value) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return [int(v) for v in value.split(",") if str(v).strip() != ""]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM baseline for cls_/reg_ columns")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--input_path", type=str, default="data/preprocess_script/data_mother.parquet", help="Input parquet file")
    parser.add_argument("--output_root", type=str, default="output/lstm", help="Output root directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_folds", type=str, default=None, help="Comma-separated fold ids, e.g. 5,6,7")
    parser.add_argument("--valid_folds", type=str, default=None, help="Comma-separated fold ids, e.g. 0")
    parser.add_argument("--test_folds", type=str, default=None, help="Comma-separated fold ids, e.g. 0")
    parser.add_argument("--use_data_parallel", type=bool, default=True)
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU ids, e.g. 0,1,2")
    parser.add_argument("--normalize_reg", type=bool, default=True)
    parser.add_argument("--optimize_f1", type=bool, default=False)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)

    args.train_folds = _normalize_folds(args.train_folds)
    args.valid_folds = _normalize_folds(args.valid_folds)
    args.test_folds = _normalize_folds(args.test_folds)
    args.gpu_ids = _normalize_gpu_ids(args.gpu_ids)

    set_seed(args.seed)

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = next_version(Path(args.output_root))

    if _is_gfe_precessed2d(input_path):
        records, cls_cols, reg_cols = build_sequences_from_gfe(input_path, normalize_reg=args.normalize_reg)
        if not cls_cols and not reg_cols:
            raise ValueError("No cls or reg tasks found in info_task.json")
        if not records:
            raise ValueError("No records found in precessed2D metadata")
        feature_dim = records[0]["features"].shape[-1]
        feature_cols = [f"feat_{i}" for i in range(feature_dim)]
    else:
        df = pd.read_parquet(input_path)
        if "pid" not in df.columns:
            raise ValueError("Input parquet must include pid column")

        cls_cols = [c for c in df.columns if c.startswith("cls_")]
        reg_cols = [c for c in df.columns if c.startswith("reg_")]

        if not cls_cols and not reg_cols:
            raise ValueError("No cls_ or reg_ columns found")

        feature_cols = infer_feature_cols(df, cls_cols, reg_cols)
        if not feature_cols:
            raise ValueError("No usable feature columns found")

        records = build_sequences(df, cls_cols, reg_cols, feature_cols)
    train_records, val_records, test_records = split_records(
        records,
        train_folds=args.train_folds,
        valid_folds=args.valid_folds,
        test_folds=args.test_folds,
    )

    if not train_records:
        raise ValueError("Training split is empty. Check train_folds or dataset_fold10.")
    if not val_records:
        raise ValueError("Validation split is empty. Check valid_folds or dataset_fold10.")
    if not test_records:
        raise ValueError("Test split is empty. Check test_folds or dataset_fold10.")

    train_feats = [r["features"] for r in train_records]
    mean, std = normalize_features(train_feats)

    for rec in records:
        rec["features"] = (rec["features"] - mean) / std

    train_ds = SequenceDataset(train_records, cls_cols, reg_cols)
    val_ds = SequenceDataset(val_records, cls_cols, reg_cols)
    test_ds = SequenceDataset(test_records, cls_cols, reg_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMHead(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_size,
        num_layers=args.num_layers,
        cls_dim=len(cls_cols),
        reg_dim=len(reg_cols),
        dropout=args.dropout,
    )

    if device.type == "cuda" and args.use_data_parallel:
        available = torch.cuda.device_count()
        if available > 1:
            device_ids = args.gpu_ids if args.gpu_ids else list(range(available))
            device_ids = [i for i in device_ids if i < available]
            if len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device)

    if args.weight_decay > 0:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    def run_epoch(loader: DataLoader, train: bool, epoch: int) -> float:
        model.train() if train else model.eval()
        total_loss, total_count = 0.0, 0
        phase = "train" if train else "val"
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} {phase}", **TQDM_KW)
        for batch in pbar:
            feats = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            cls_labels = batch["cls_labels"].to(device) if batch["cls_labels"] is not None else None
            cls_mask = batch["cls_mask"].to(device) if batch["cls_mask"] is not None else None
            reg_labels = batch["reg_labels"].to(device) if batch["reg_labels"] is not None else None
            reg_mask = batch["reg_mask"].to(device) if batch["reg_mask"] is not None else None

            if train:
                optim.zero_grad()

            cls_logits, reg_preds = model(feats, lengths)

            loss = 0.0
            if cls_logits is not None:
                cls_loss = bce(cls_logits, cls_labels)
                cls_loss = cls_loss * cls_mask.float()
                cls_loss = cls_loss.sum() / cls_mask.float().sum().clamp(min=1.0)
                loss = loss + cls_loss
            if reg_preds is not None:
                reg_loss = mse(reg_preds, reg_labels)
                reg_loss = reg_loss * reg_mask.float()
                reg_loss = reg_loss.sum() / reg_mask.float().sum().clamp(min=1.0)
                loss = loss + reg_loss

            if train:
                loss.backward()
                optim.step()

            total_loss += loss.item()
            total_count += 1
            pbar.set_postfix(loss=f"{total_loss / max(total_count, 1):.4f}")
        return total_loss / max(total_count, 1)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    macro_rows = []
    per_target_rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, train=True, epoch=epoch)
        val_loss = None
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            val_loss = run_epoch(val_loader, train=False, epoch=epoch)
            cls_logits, cls_labels, cls_mask, reg_preds, reg_labels, reg_mask = collect_outputs(
                val_loader, model, device, cls_cols, reg_cols
            )
            val_metrics = compute_metrics(
                cls_logits,
                cls_labels,
                cls_mask,
                reg_preds,
                reg_labels,
                reg_mask,
                cls_cols,
                reg_cols,
                optimize_f1=args.optimize_f1,
            )
            macro = val_metrics.get("cls", {}).get("macro", {})
            if macro:
                macro_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "cls",
                    "metric": "auc",
                    "value": macro.get("auc"),
                })
                macro_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "cls",
                    "metric": "f1",
                    "value": macro.get("f1"),
                })
            macro_reg = val_metrics.get("reg", {}).get("macro", {})
            if macro_reg:
                macro_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "reg",
                    "metric": "r2",
                    "value": macro_reg.get("r2"),
                })

            for target, vals in val_metrics.get("cls", {}).get("per_target", {}).items():
                per_target_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "cls",
                    "target": target,
                    "metric": "auc",
                    "value": vals.get("auc"),
                })
                per_target_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "cls",
                    "target": target,
                    "metric": "f1",
                    "value": vals.get("f1"),
                })
            for target, vals in val_metrics.get("reg", {}).get("per_target", {}).items():
                per_target_rows.append({
                    "epoch": epoch,
                    "split": "val",
                    "task": "reg",
                    "target": target,
                    "metric": "r2",
                    "value": vals.get("r2"),
                })
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        else:
            print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val=skipped")

        if args.early_stop_patience and no_improve >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch:03d} (no improve {no_improve} evals)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if macro_rows:
        pd.DataFrame(macro_rows).to_parquet(output_dir / "metrics_epoch_macro.parquet", index=False)
    if per_target_rows:
        pd.DataFrame(per_target_rows).to_parquet(output_dir / "metrics_epoch_per_target.parquet", index=False)

    cls_logits, cls_labels, cls_mask, reg_preds, reg_labels, reg_mask = collect_outputs(
        tqdm(test_loader, desc="Testing", **TQDM_KW), model, device, cls_cols, reg_cols
    )

    metrics = compute_metrics(
        cls_logits,
        cls_labels,
        cls_mask,
        reg_preds,
        reg_labels,
        reg_mask,
        cls_cols,
        reg_cols,
        optimize_f1=args.optimize_f1,
    )
    save_metrics(metrics, output_dir)

    config = {
        "input_path": str(input_path),
        "feature_cols": feature_cols,
        "cls_cols": cls_cols,
        "reg_cols": reg_cols,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_folds": args.train_folds,
        "valid_folds": args.valid_folds,
        "test_folds": args.test_folds,
        "use_data_parallel": args.use_data_parallel,
        "gpu_ids": args.gpu_ids,
        "normalize_reg": args.normalize_reg,
        "optimize_f1": args.optimize_f1,
        "val_interval": args.val_interval,
        "early_stop_patience": args.early_stop_patience,
        "version_dir": str(output_dir),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
