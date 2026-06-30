"""Inference wrapper for MoChiFormer.

``MoChiFormerPredictor`` loads a trained checkpoint and exposes:
  * ``predict_records`` -- batched prediction over PatientRecord objects;
  * ``predict_raw``     -- an adapter from MoChiAgent's loose
                          ``{ehr: [str], lab_tests: [[float]]}`` request shape to a
                          single-patient prediction, returning the score / risk /
                          embedding payload the transformer_server tools expose.

Note (scope): raw free-text EHR strings are NOT tokenized into structured event
features here — the core model consumes discretized laboratory measurements and
structured-EHR event codes. Supplying real categorical-event codes (a
``cat_feats`` matrix) exercises the event channel; with only ``lab_tests`` that
channel is treated as missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .data import FeatureSchema, PatientRecord
from .model import MoChiFormer


def _risk_category(score: float) -> str:
    if score < 0.3:
        return "LOW"
    if score < 0.7:
        return "MODERATE"
    return "HIGH"


class MoChiFormerPredictor:
    def __init__(self, model: MoChiFormer, cfg, schema: FeatureSchema, device: str = "cpu"):
        self.model = model
        self.cfg = cfg
        self.schema = schema
        self.device = device
        self.model.to(device).eval()

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[str] = None) -> "MoChiFormerPredictor":
        from .train import load_checkpoint

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, schema = load_checkpoint(path, device=device)
        return cls(model, cfg, schema, device=device)

    # ---- core batched prediction --------------------------------------------
    @torch.no_grad()
    def predict_records(self, records: List[PatientRecord]) -> List[Dict[str, Any]]:
        from .data import LongitudinalEHRDataset, make_collate
        from torch.utils.data import DataLoader

        ds = LongitudinalEHRDataset(records, self.schema, self.cfg, mode="finetune")
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                            collate_fn=make_collate("finetune"))
        results: List[Dict[str, Any]] = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model.predict_patient(
                batch["cat_feats"], batch["float_feats"], batch["event_feats"],
                batch["valid_mask"], batch["time_index"]
            )
            probs = out["cls_probs"].cpu().numpy() if out["cls_probs"] is not None else None
            regs = out["reg_vals"].cpu().numpy() if out["reg_vals"] is not None else None
            embs = out["patient_emb"].cpu().numpy()
            for i in range(embs.shape[0]):
                results.append(self._format_one(
                    probs[i] if probs is not None else None,
                    regs[i] if regs is not None else None,
                    embs[i],
                ))
        return results

    def _format_one(self, probs, regs, emb) -> Dict[str, Any]:
        per_disease = {}
        if probs is not None:
            per_disease = {name: float(probs[k]) for k, name in enumerate(self.schema.cls_target_names)}
        biological_age = {}
        if regs is not None and self.schema.reg_target_names:
            de = self.schema.destandardize_reg(np.asarray(regs))
            biological_age = {name: float(de[r]) for r, name in enumerate(self.schema.reg_target_names)}

        if per_disease:
            top_disease = max(per_disease, key=per_disease.get)
            score = float(per_disease[top_disease])
            confidence = float(np.mean([2 * abs(p - 0.5) for p in per_disease.values()]))
        else:
            top_disease, score, confidence = None, 0.0, 0.0

        return {
            "prediction_score": score,
            "risk_category": _risk_category(score),
            "confidence": confidence,
            "top_disease": top_disease,
            "disease_probabilities": per_disease,
            "biological_age": biological_age,
            "patient_embedding": emb.tolist(),
        }

    # ---- adapter from MoChiAgent's raw request shape -------------------------
    def predict_raw(
        self,
        ehr: List[str],
        lab_tests: List[List[Any]],
        timestamps: Optional[List[int]] = None,
        cat_feats: Optional[List[List[int]]] = None,
        return_embedding: bool = False,
    ) -> Dict[str, Any]:
        record = self._build_record(ehr, lab_tests, timestamps, cat_feats)
        result = self.predict_records([record])[0]
        if not return_embedding:
            result.pop("patient_embedding", None)
        result["n_visits"] = int(record.float_raw.shape[0])
        result["n_ehr_records"] = len(ehr or [])
        return result

    def _build_record(self, ehr, lab_tests, timestamps, cat_feats) -> PatientRecord:
        nf = self.schema.n_float_feats
        nc = self.schema.n_cat_feats

        lab = np.asarray(lab_tests, dtype=np.float64) if lab_tests else np.zeros((1, nf))
        if lab.ndim == 1:
            lab = lab.reshape(1, -1)
        t = lab.shape[0]
        # align column count to the schema (pad with nan / truncate)
        if lab.shape[1] < nf:
            pad = np.full((t, nf - lab.shape[1]), np.nan)
            lab = np.concatenate([lab, pad], axis=1)
        elif lab.shape[1] > nf:
            lab = lab[:, :nf]

        if cat_feats is not None:
            cat = np.asarray(cat_feats, dtype=np.int64)
            if cat.shape[0] != t:
                cat = np.full((t, nc), -1, dtype=np.int64)
        else:
            cat = np.full((t, nc), -1, dtype=np.int64)  # no structured-event encoding from raw text

        if timestamps is not None and len(timestamps) == t:
            time_index = np.asarray(timestamps, dtype=np.int64)
        else:
            time_index = np.arange(t, dtype=np.int64)  # assume unit-spaced visits

        # raw free-text EHR is not tokenized into events here -> event channel missing
        event = np.full((t, self.schema.n_event_feats), -1, dtype=np.int64)

        return PatientRecord(
            cat_codes=cat,
            float_raw=lab,
            time_index=time_index,
            cls_labels=np.zeros(self.cfg.n_cls_tasks, dtype=np.int64),
            reg_labels=np.zeros(self.cfg.n_reg_tasks, dtype=np.float64),
            cohort_id=0,
            event_codes=event,
        )
