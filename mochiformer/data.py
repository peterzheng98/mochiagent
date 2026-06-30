"""Data processing for MoChiFormer (paper "Stage 1: Data processing").

Implements the continuous-feature discretization D(x), the tokenized longitudinal
representation, padding/collation, and a synthetic longitudinal-EHR cohort
generator so the full train -> infer -> serve pipeline is runnable without any
private clinical data. Replace ``make_synthetic_cohort`` (and point
``LongitudinalEHRDataset`` at your own ``PatientRecord`` list) to train on real
longitudinal EHR data; nothing else in the pipeline changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import MoChiFormerConfig


# ---------------------------------------------------------------------------
# A single patient's longitudinal record (raw, pre-discretization)
# ---------------------------------------------------------------------------
@dataclass
class PatientRecord:
    cat_codes: np.ndarray     # [T, n_cat_feats] int, -1 = missing
    float_raw: np.ndarray     # [T, n_float_feats] float, np.nan = missing
    time_index: np.ndarray    # [T] int days since first visit
    cls_labels: np.ndarray    # [K] {0,1} patient-level disease labels
    reg_labels: np.ndarray    # [R] float patient-level age targets
    cohort_id: int = 0
    event_codes: Optional[np.ndarray] = None  # [T, n_event_feats] int, -1 = missing


# ---------------------------------------------------------------------------
# Feature schema: per-feature min/max for D(x) + target standardization
# ---------------------------------------------------------------------------
@dataclass
class FeatureSchema:
    n_cat_feats: int
    n_float_feats: int
    n_cat_values: int
    n_float_bins: int
    n_event_feats: int
    n_event_values: int
    float_min: np.ndarray            # [n_float_feats]
    float_max: np.ndarray            # [n_float_feats]
    cls_target_names: List[str] = field(default_factory=list)
    reg_target_names: List[str] = field(default_factory=list)
    reg_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reg_std: np.ndarray = field(default_factory=lambda: np.ones(0))

    # ---- discretization D(x) -------------------------------------------------
    def discretize(self, float_raw: np.ndarray) -> np.ndarray:
        """floor((x - xmin)/(xmax - xmin) * n_bins), clipped; nan -> -1."""
        rng = np.maximum(self.float_max - self.float_min, 1e-8)
        z = (float_raw - self.float_min) / rng
        codes = np.floor(z * self.n_float_bins)
        codes = np.clip(codes, 0, self.n_float_bins - 1)
        codes = np.where(np.isnan(float_raw), -1, codes)
        return codes.astype(np.int64)

    def standardize_reg(self, reg: np.ndarray) -> np.ndarray:
        if self.reg_std.size == 0:
            return reg
        return (reg - self.reg_mean) / np.maximum(self.reg_std, 1e-8)

    def destandardize_reg(self, reg: np.ndarray) -> np.ndarray:
        if self.reg_std.size == 0:
            return reg
        return reg * np.maximum(self.reg_std, 1e-8) + self.reg_mean

    # ---- fit / (de)serialize -------------------------------------------------
    @classmethod
    def fit(cls, records: List[PatientRecord], cfg: MoChiFormerConfig) -> "FeatureSchema":
        floats = np.concatenate([r.float_raw for r in records], axis=0)  # [sum T, n_float]
        fmin = np.nanmin(floats, axis=0)
        fmax = np.nanmax(floats, axis=0)
        regs = np.stack([r.reg_labels for r in records], axis=0) if cfg.n_reg_tasks else np.zeros((len(records), 0))
        reg_mean = regs.mean(axis=0) if regs.shape[1] else np.zeros(0)
        reg_std = regs.std(axis=0) if regs.shape[1] else np.ones(0)
        return cls(
            n_cat_feats=cfg.n_cat_feats,
            n_float_feats=cfg.n_float_feats,
            n_cat_values=cfg.n_cat_values,
            n_float_bins=cfg.n_float_bins,
            n_event_feats=cfg.n_event_feats,
            n_event_values=cfg.n_event_values,
            float_min=fmin,
            float_max=fmax,
            cls_target_names=list(cfg.cls_target_names),
            reg_target_names=list(cfg.reg_target_names),
            reg_mean=reg_mean,
            reg_std=reg_std,
        )

    def to_dict(self) -> Dict:
        return {
            "n_cat_feats": self.n_cat_feats,
            "n_float_feats": self.n_float_feats,
            "n_cat_values": self.n_cat_values,
            "n_float_bins": self.n_float_bins,
            "n_event_feats": self.n_event_feats,
            "n_event_values": self.n_event_values,
            "float_min": self.float_min.tolist(),
            "float_max": self.float_max.tolist(),
            "cls_target_names": self.cls_target_names,
            "reg_target_names": self.reg_target_names,
            "reg_mean": self.reg_mean.tolist(),
            "reg_std": self.reg_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureSchema":
        return cls(
            n_cat_feats=d["n_cat_feats"],
            n_float_feats=d["n_float_feats"],
            n_cat_values=d["n_cat_values"],
            n_float_bins=d["n_float_bins"],
            n_event_feats=d.get("n_event_feats", 0),
            n_event_values=d.get("n_event_values", 8),
            float_min=np.array(d["float_min"], dtype=np.float64),
            float_max=np.array(d["float_max"], dtype=np.float64),
            cls_target_names=d.get("cls_target_names", []),
            reg_target_names=d.get("reg_target_names", []),
            reg_mean=np.array(d.get("reg_mean", []), dtype=np.float64),
            reg_std=np.array(d.get("reg_std", []), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LongitudinalEHRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: List[PatientRecord],
        schema: FeatureSchema,
        cfg: MoChiFormerConfig,
        mode: str = "finetune",          # "finetune" | "pretrain"
        seed: int = 0,
    ):
        assert mode in ("finetune", "pretrain")
        self.records = records
        self.schema = schema
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _truncate(self, arr: np.ndarray) -> np.ndarray:
        return arr[: self.cfg.max_visits]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.records[idx]
        cat = self._truncate(r.cat_codes).astype(np.int64)
        float_codes = self.schema.discretize(self._truncate(r.float_raw))
        time_index = self._truncate(r.time_index).astype(np.int64)
        t = cat.shape[0]

        if r.event_codes is not None and self.cfg.n_event_feats > 0:
            event = self._truncate(r.event_codes).astype(np.int64)
        else:
            event = np.full((t, self.cfg.n_event_feats), -1, dtype=np.int64)

        item = {
            "cat_feats": torch.from_numpy(cat),
            "float_feats": torch.from_numpy(float_codes),
            "event_feats": torch.from_numpy(event),
            "time_index": torch.from_numpy(time_index),
            "valid_mask": torch.ones(t, dtype=torch.float32),
        }

        if self.mode == "finetune":
            item["cls_labels"] = torch.from_numpy(r.cls_labels.astype(np.int64))
            reg = self.schema.standardize_reg(r.reg_labels.astype(np.float64))
            item["reg_labels"] = torch.from_numpy(reg.astype(np.float32))
            item["cohort_id"] = torch.tensor(int(r.cohort_id), dtype=torch.long)
        else:
            # masked-feature reconstruction targets
            cat_target = cat.copy()
            float_target = float_codes.copy()
            cat_input = cat.copy()
            float_input = float_codes.copy()
            mask_cat = np.zeros_like(cat, dtype=bool)
            mask_float = np.zeros_like(float_codes, dtype=bool)
            # only mask positions that are actually observed
            for arr_in, arr_codes, mask, n in (
                (cat_input, cat, mask_cat, self.cfg.n_cat_feats),
                (float_input, float_codes, mask_float, self.cfg.n_float_feats),
            ):
                observed = np.argwhere(arr_codes >= 0)
                if len(observed) == 0:
                    continue
                k = max(1, int(round(self.cfg.mask_ratio * len(observed))))
                chosen = observed[self.rng.choice(len(observed), size=k, replace=False)]
                for (i, j) in chosen:
                    mask[i, j] = True
                    arr_in[i, j] = -1  # replace with missing/mask token
            item["cat_input"] = torch.from_numpy(cat_input)
            item["float_input"] = torch.from_numpy(float_input)
            item["cat_target"] = torch.from_numpy(cat_target)
            # numerical reconstruction target normalized to [0,1] over the bin range
            float_target_norm = float_target.astype(np.float32) / max(1, self.cfg.n_float_bins - 1)
            item["float_target"] = torch.from_numpy(float_target_norm)
            item["mask_cat"] = torch.from_numpy(mask_cat)
            item["mask_float"] = torch.from_numpy(mask_float)
        return item


# ---------------------------------------------------------------------------
# Collation (pad to max T in batch)
# ---------------------------------------------------------------------------
def _pad_stack(tensors: List[torch.Tensor], max_t: int, pad_value) -> torch.Tensor:
    out = []
    for x in tensors:
        t = x.shape[0]
        if t < max_t:
            pad_shape = (max_t - t,) + tuple(x.shape[1:])
            pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        out.append(x)
    return torch.stack(out, dim=0)


def make_collate(mode: str):
    seq_keys_int = {
        "cat_feats": -1, "float_feats": -1, "event_feats": -1, "time_index": 0,
        "cat_input": -1, "float_input": -1, "cat_target": -1,
    }
    seq_keys_float = {"valid_mask": 0.0, "float_target": 0.0}
    seq_keys_bool = {"mask_cat": False, "mask_float": False}
    flat_keys = ("cls_labels", "reg_labels", "cohort_id")

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_t = max(b["cat_feats"].shape[0] for b in batch)
        out: Dict[str, torch.Tensor] = {}
        for k, pad in seq_keys_int.items():
            if k in batch[0]:
                out[k] = _pad_stack([b[k] for b in batch], max_t, pad)
        for k, pad in seq_keys_float.items():
            if k in batch[0]:
                out[k] = _pad_stack([b[k] for b in batch], max_t, pad)
        for k, pad in seq_keys_bool.items():
            if k in batch[0]:
                out[k] = _pad_stack([b[k] for b in batch], max_t, pad)
        for k in flat_keys:
            if k in batch[0]:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out

    return collate


# ---------------------------------------------------------------------------
# Synthetic longitudinal-EHR cohort (planted signal, with batch effect)
# ---------------------------------------------------------------------------
def make_synthetic_cohort(
    n_patients: int, cfg: MoChiFormerConfig, seed: int = 0, signal_seed: int = 0xC0FFEE
) -> List[PatientRecord]:
    rng = np.random.default_rng(seed)
    K, R = cfg.n_cls_tasks, cfg.n_reg_tasks
    nf, nc, ne = cfg.n_float_feats, cfg.n_cat_feats, cfg.n_event_feats

    # Ground-truth disease/age model: the SAME planted signal must underlie every
    # cohort (train/eval/test) so a model learned on one generalizes to another.
    # Hence it is drawn from a fixed `signal_seed`, independent of the per-cohort
    # patient-sampling `seed`.
    srng = np.random.default_rng(signal_seed)
    w_cls = srng.normal(size=(K, nf)) * 1.5
    b_cls = srng.normal(size=(K,)) * 0.3
    w_reg = srng.normal(size=(R, nf)) * 1.0

    records: List[PatientRecord] = []
    for _ in range(n_patients):
        t = int(rng.integers(2, cfg.max_visits + 1))
        cohort = int(rng.integers(0, max(1, cfg.n_cohorts)))
        # base per-feature scale + a cohort-specific additive shift (batch effect)
        base = rng.normal(loc=5.0, scale=2.0, size=(nf,))
        cohort_shift = (cohort - 0.5) * rng.normal(loc=1.0, scale=0.3, size=(nf,))
        float_raw = rng.normal(loc=base + cohort_shift, scale=1.0, size=(t, nf))
        # inject ~10% missingness
        miss = rng.random(size=(t, nf)) < 0.10
        float_raw[miss] = np.nan
        cat_codes = rng.integers(0, cfg.n_cat_values, size=(t, nc)).astype(np.int64)
        # ~5% missing categorical
        cat_codes[rng.random(size=(t, nc)) < 0.05] = -1
        # structured-EHR diagnostic events (discrete codes; ~10% missing)
        if ne > 0:
            event_codes = rng.integers(0, max(1, cfg.n_event_values), size=(t, ne)).astype(np.int64)
            event_codes[rng.random(size=(t, ne)) < 0.10] = -1
        else:
            event_codes = np.full((t, 0), -1, dtype=np.int64)
        time_index = np.cumsum(rng.integers(1, 30, size=(t,))).astype(np.int64) - 0  # days
        time_index = time_index - time_index[0]

        with np.errstate(invalid="ignore"):
            feat_mean = np.nanmean(np.where(np.isnan(float_raw).all(axis=0), 0.0, float_raw), axis=0)
        feat_mean = np.where(np.isnan(feat_mean), 0.0, feat_mean)
        feat_mean_z = (feat_mean - 5.0) / 2.0

        cls_logits = w_cls @ feat_mean_z + b_cls
        cls_labels = (rng.random(size=K) < 1.0 / (1.0 + np.exp(-cls_logits))).astype(np.int64)
        reg_labels = (w_reg @ feat_mean_z) * 3.0 + 30.0 + rng.normal(scale=1.0, size=R)

        records.append(
            PatientRecord(
                cat_codes=cat_codes,
                float_raw=float_raw,
                time_index=time_index,
                cls_labels=cls_labels,
                reg_labels=reg_labels.astype(np.float64),
                cohort_id=cohort,
                event_codes=event_codes,
            )
        )
    return records


def save_schema(schema: FeatureSchema, path: str) -> None:
    with open(path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2)


def load_schema(path: str) -> FeatureSchema:
    with open(path) as f:
        return FeatureSchema.from_dict(json.load(f))
