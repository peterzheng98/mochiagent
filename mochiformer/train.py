"""Training for MoChiFormer (paper "Stage 2: Model training").

Two stages, matching the paper:
  * pretrain  -- self-supervised masked-feature reconstruction (+ optional KL)
  * finetune  -- multi-task focal classification + age regression (+ optional
                 cohort-adversarial de-biasing), initialized from the pretrained
                 visit encoder.

Run directly for a fully self-contained synthetic demo that produces a usable
checkpoint:

    python -m mochiformer.train demo --out checkpoints/mochiformer_demo.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import MoChiFormerConfig, demo_config
from .data import (
    FeatureSchema,
    LongitudinalEHRDataset,
    PatientRecord,
    make_collate,
    make_synthetic_cohort,
)
from .model import MoChiFormer, focal_loss, masked_mean


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _loader(records, schema, cfg, mode, shuffle, seed=0) -> DataLoader:
    ds = LongitudinalEHRDataset(records, schema, cfg, mode=mode, seed=seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=make_collate(mode))


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------
def pretrain(
    cfg: MoChiFormerConfig,
    records: List[PatientRecord],
    schema: FeatureSchema,
    device: str = "cpu",
    model: Optional[MoChiFormer] = None,
    log_every: int = 0,
) -> MoChiFormer:
    model = model or MoChiFormer(cfg)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _loader(records, schema, cfg, "pretrain", shuffle=True)

    for epoch in range(cfg.pretrain_epochs):
        tot = 0.0
        for batch in loader:
            batch = _to_device(batch, device)
            out = model.pretrain_forward(
                batch["cat_input"], batch["float_input"], batch["event_feats"], batch["valid_mask"]
            )
            mc, mf = batch["mask_cat"], batch["mask_float"]
            loss = out["cat_logits"].new_zeros(())
            if mc.any():
                loss = loss + F.cross_entropy(out["cat_logits"][mc], batch["cat_target"][mc])
            if mf.any():
                loss = loss + F.mse_loss(out["float_pred"][mf], batch["float_target"][mf])
            if "kl" in out:
                loss = loss + cfg.vae_kl_weight * out["kl"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[pretrain] epoch {epoch+1}/{cfg.pretrain_epochs} loss={tot/len(loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def finetune(
    cfg: MoChiFormerConfig,
    records: List[PatientRecord],
    schema: FeatureSchema,
    device: str = "cpu",
    init_model: Optional[MoChiFormer] = None,
    log_every: int = 0,
) -> MoChiFormer:
    model = MoChiFormer(cfg)
    if init_model is not None:
        # warm-start overlapping params (visit encoder, vae, etc.) from pretrain
        model.load_state_dict(init_model.state_dict(), strict=False)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _loader(records, schema, cfg, "finetune", shuffle=True)

    for epoch in range(cfg.finetune_epochs):
        # GRL strength ramps 0 -> grl_lambda_max over training
        if model.cohort_disc is not None:
            frac = (epoch + 1) / max(1, cfg.finetune_epochs)
            model.set_grl_lambda(cfg.grl_lambda_max * frac)
        tot = 0.0
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(batch["cat_feats"], batch["float_feats"], batch["event_feats"],
                        batch["valid_mask"], batch["time_index"], batch.get("cohort_id"))
            vm = batch["valid_mask"]

            # classification: per-visit focal loss, patient label broadcast over visits
            cls_loss = out["hidden"].new_zeros(())
            for k, logits in enumerate(out["cls_logits"]):
                tgt = batch["cls_labels"][:, k].unsqueeze(1).expand(-1, logits.shape[1])  # [B,T]
                fl = focal_loss(logits.reshape(-1, 2), tgt.reshape(-1), cfg.focal_alpha, cfg.focal_gamma)
                fl = fl.reshape(vm.shape)
                cls_loss = cls_loss + (fl * vm).sum() / vm.sum().clamp(min=1.0)
            cls_loss = cls_loss / max(1, len(out["cls_logits"]))

            # regression: patient-level masked-mean pooling, MSE
            reg_loss = out["hidden"].new_zeros(())
            for r, preds in enumerate(out["reg_preds"]):
                pooled = masked_mean(preds, vm)              # [B]
                reg_loss = reg_loss + F.mse_loss(pooled, batch["reg_labels"][:, r])
            reg_loss = reg_loss / max(1, len(out["reg_preds"]))

            loss = cfg.lambda_cls * cls_loss + cfg.lambda_reg * reg_loss
            if out["cohort_logits"] is not None:
                cohort_loss = F.cross_entropy(out["cohort_logits"], batch["cohort_id"])
                loss = loss + cfg.lambda_cohort * cfg.cohort_adv_weight * cohort_loss
            if out["rel_scores"] is not None:
                # target = empirical co-occurrence (both diseases present in the patient)
                K = cfg.n_cls_tasks
                iu = torch.triu_indices(K, K, offset=1, device=vm.device)
                A = (batch["cls_labels"][:, iu[0]] * batch["cls_labels"][:, iu[1]]).float()
                rel_loss = F.binary_cross_entropy_with_logits(out["rel_scores"], A)
                loss = loss + cfg.lambda_rel * rel_loss
            if cfg.use_vae:
                loss = loss + cfg.vae_kl_weight * out["kl"]

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[finetune] epoch {epoch+1}/{cfg.finetune_epochs} loss={tot/len(loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation (macro AUC) — sanity metric
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: MoChiFormer, cfg, records, schema, device="cpu") -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    model.eval()
    loader = _loader(records, schema, cfg, "finetune", shuffle=False)
    probs, labels = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        p = model.predict_patient(batch["cat_feats"], batch["float_feats"], batch["event_feats"],
                                  batch["valid_mask"], batch["time_index"])
        probs.append(p["cls_probs"].cpu().numpy())
        labels.append(batch["cls_labels"].cpu().numpy())
    probs = np.concatenate(probs, 0)
    labels = np.concatenate(labels, 0)
    aucs = []
    for k in range(cfg.n_cls_tasks):
        if len(np.unique(labels[:, k])) > 1:
            aucs.append(roc_auc_score(labels[:, k], probs[:, k]))
    return {"macro_auc": float(np.mean(aucs)) if aucs else float("nan"), "n_eval": len(labels)}


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model: MoChiFormer, cfg: MoChiFormerConfig, schema: FeatureSchema) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.to_dict(), "schema": schema.to_dict()},
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    from .data import FeatureSchema as _FS

    # Checkpoint holds only tensors + plain dict/list config & schema, so the
    # safe loader (weights_only=True) is sufficient and avoids arbitrary unpickling.
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = MoChiFormerConfig.from_dict(ckpt["config"])
    schema = _FS.from_dict(ckpt["schema"])
    model = MoChiFormer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, schema


# ---------------------------------------------------------------------------
# End-to-end demo on synthetic data
# ---------------------------------------------------------------------------
def train_demo(out: str, cfg: Optional[MoChiFormerConfig] = None, device: Optional[str] = None,
               n_train: int = 512, n_eval: int = 128, log_every: int = 1) -> Dict[str, float]:
    cfg = cfg or demo_config()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    train_records = make_synthetic_cohort(n_train, cfg, seed=cfg.seed)
    eval_records = make_synthetic_cohort(n_eval, cfg, seed=cfg.seed + 1)
    schema = FeatureSchema.fit(train_records, cfg)

    print(f"[demo] device={device} train={n_train} eval={n_eval}")
    pre = pretrain(cfg, train_records, schema, device=device, log_every=log_every)
    model = finetune(cfg, train_records, schema, device=device, init_model=pre, log_every=log_every)
    metrics = evaluate(model, cfg, eval_records, schema, device=device)
    print(f"[demo] eval macro_auc={metrics['macro_auc']:.4f} (n={metrics['n_eval']})")

    save_checkpoint(out, model, cfg, schema)
    print(f"[demo] saved checkpoint -> {out}")
    return metrics


def _main() -> None:
    p = argparse.ArgumentParser(description="MoChiFormer training")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("demo", help="train on synthetic data and save a checkpoint")
    pd.add_argument("--out", default="checkpoints/mochiformer_demo.ckpt")
    pd.add_argument("--config", default=None, help="optional JSON config override")
    pd.add_argument("--n-train", type=int, default=512)
    pd.add_argument("--n-eval", type=int, default=128)
    pd.add_argument("--device", default=None)

    args = p.parse_args()
    if args.cmd == "demo":
        cfg = demo_config()
        if args.config:
            with open(args.config) as f:
                cfg = MoChiFormerConfig.from_dict({**cfg.to_dict(), **json.load(f)})
        train_demo(args.out, cfg=cfg, device=args.device, n_train=args.n_train, n_eval=args.n_eval)


if __name__ == "__main__":
    _main()
