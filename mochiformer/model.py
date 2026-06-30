"""MoChiFormer model.

A hybrid encoder--decoder transformer for longitudinal EHR analysis, following
the paper's "Stage 2: Model training" / "Implementation details":

    visit-level BERT encoder  ->  (optional VAE)  ->  + continuous time embedding
        ->  patient-level GPT-2 temporal (causal) decoder  ->  task heads

It ingests each patient's chronologically ordered visits of discretized
laboratory measurements and structured-EHR events, applies the paper's
anti-leakage temporal controls to the event channel (current-visit diagnosis
padded; event sequence shifted one step behind the lab sequence), and includes
the disease-pair pairwise-relation head alongside gradient-reversal de-biasing of
cross-site batch effects (the paper's Imputation Tool).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from .config import MoChiFormerConfig

# Reserved token ids in the shared value-embedding table.
CLS_ID = 0
PAD_ID = 1          # also the "missing measurement" vector
VALUE_OFFSET = 2    # real values start at id 2


# ---------------------------------------------------------------------------
# Gradient reversal (for adversarial cohort / batch-effect de-biasing)
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


class CohortDiscriminator(nn.Module):
    """Predicts the originating cohort from the pooled patient representation.

    Placed behind a gradient-reversal layer so that minimizing its loss pushes
    the encoder toward cohort-invariant (batch-effect-free) representations.
    """

    def __init__(self, hidden_dim: int, n_cohorts: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cohorts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Variational latent pathway
# ---------------------------------------------------------------------------
class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean(x), self.logvar(x)


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # -0.5 * sum(1 + log sigma^2 - mu^2 - sigma^2), per element along last dim
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)


# ---------------------------------------------------------------------------
# Continuous time embedding
# ---------------------------------------------------------------------------
class TemporalPositionalEmbedding(nn.Module):
    """Maps the elapsed-days time index T_t to a hidden-size vector."""

    def __init__(self, max_time_days: int, embedding_dim: int):
        super().__init__()
        self.max_time_days = float(max_time_days)
        self.net = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, time_days: torch.Tensor) -> torch.Tensor:
        t = (time_days.float() / self.max_time_days).clamp(0.0, 4.0).unsqueeze(-1)
        return self.net(t)


# ---------------------------------------------------------------------------
# Visit-level encoder
# ---------------------------------------------------------------------------
class VisitEncoder(nn.Module):
    """Per-visit BERT encoder over discretized categorical, continuous, and event
    features.

    Builds the token sequence
    ``[CLS] (cat+2) (float+2+n_cat_values) (event+2+n_cat_values+n_float_bins)``
    following the paper's "Stage 3: Inference" tokenization, with the [PAD] vector
    standing in for missing measurements.
    """

    def __init__(self, cfg: MoChiFormerConfig):
        super().__init__()
        self.cfg = cfg
        bert_cfg = BertConfig(
            vocab_size=cfg.value_vocab_size,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.visit_encoder_layers,
            num_attention_heads=cfg.num_heads,
            intermediate_size=cfg.ffn_size,
            hidden_act="gelu",
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
            max_position_embeddings=max(1, cfg.n_feats + 1),
            type_vocab_size=cfg.type_vocab_size,
            pad_token_id=PAD_ID,
            position_embedding_type="none",  # features form a set; type embeds distinguish them
            attn_implementation="eager",     # required with non-absolute position embeddings
        )
        self.bert = BertModel(bert_cfg, add_pooling_layer=False)
        # token_type id per position: 0 for [CLS], then one per feature column.
        self.register_buffer(
            "token_type_ids",
            torch.arange(cfg.n_feats + 1, dtype=torch.long),
            persistent=False,
        )

    def _build_input_ids(
        self, cat: torch.Tensor, flt: torch.Tensor, evt: torch.Tensor
    ) -> torch.Tensor:
        """cat/flt/evt: int codes (-1 = missing), each [..., n_*_feats]."""
        cfg = self.cfg
        cat_ids = torch.where(cat < 0, torch.full_like(cat, PAD_ID), cat + VALUE_OFFSET)
        flt_ids = torch.where(
            flt < 0, torch.full_like(flt, PAD_ID), flt + VALUE_OFFSET + cfg.n_cat_values
        )
        evt_ids = torch.where(
            evt < 0,
            torch.full_like(evt, PAD_ID),
            evt + VALUE_OFFSET + cfg.n_cat_values + cfg.n_float_bins,
        )
        cls = torch.full(cat.shape[:-1] + (1,), CLS_ID, dtype=torch.long, device=cat.device)
        return torch.cat([cls, cat_ids, flt_ids, evt_ids], dim=-1)

    def forward(
        self, cat_feats: torch.Tensor, float_feats: torch.Tensor, event_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cls_emb [N, H], token_hidden [N, 1+n_feats, H]).

        Inputs are flattened over batch*visits to shape [N, n_*_feats].
        """
        input_ids = self._build_input_ids(cat_feats, float_feats, event_feats)
        n = input_ids.shape[0]
        ttype = self.token_type_ids.unsqueeze(0).expand(n, -1)
        out = self.bert(input_ids=input_ids, token_type_ids=ttype)
        token_hidden = out.last_hidden_state
        return token_hidden[:, 0], token_hidden


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------
def _mlp_head(in_dim: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    # paper: lightweight two-/three-layer MLP with ReLU
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim, out_dim),
    )


class FinetuneTaskHeads(nn.Module):
    def __init__(self, cfg: MoChiFormerConfig):
        super().__init__()
        h = cfg.hidden_size
        self.cls_heads = nn.ModuleList([_mlp_head(h, 2, cfg.dropout) for _ in cfg.cls_target_names])
        self.reg_heads = nn.ModuleList([_mlp_head(h, 1, cfg.dropout) for _ in cfg.reg_target_names])

    def forward(self, hidden: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_logits = [head(hidden) for head in self.cls_heads]          # each [B, T, 2]
        reg_preds = [head(hidden).squeeze(-1) for head in self.reg_heads]  # each [B, T]
        return cls_logits, reg_preds


class PretrainReconHeads(nn.Module):
    """Shared masked-feature reconstruction heads (categorical CE, numerical MSE)."""

    def __init__(self, cfg: MoChiFormerConfig):
        super().__init__()
        h = cfg.hidden_size
        self.cat_head = _mlp_head(h, cfg.n_cat_values, cfg.dropout)
        self.float_head = _mlp_head(h, 1, cfg.dropout)

    def forward(self, token_hidden: torch.Tensor, n_cat: int, n_float: int):
        # token_hidden: [N, 1+n_feats, H]; positions 1..n_cat are categorical,
        # the next n_float are continuous.
        cat_h = token_hidden[:, 1 : 1 + n_cat]
        flt_h = token_hidden[:, 1 + n_cat : 1 + n_cat + n_float]
        cat_logits = self.cat_head(cat_h)                  # [N, n_cat, n_cat_values]
        float_pred = torch.sigmoid(self.float_head(flt_h)).squeeze(-1)  # [N, n_float] in [0,1]
        return cat_logits, float_pred


class PairwiseRelationHead(nn.Module):
    """Disease-pair co-occurrence head (paper "Task-specific decoders").

    Per-disease learned query vectors attention-pool the patient's hidden states
    into disease embeddings h_i; each pair is scored from
    z_ij = [h_i, h_j, h_i ⊙ h_j] by a two-layer MLP, symmetrized so s_ij = s_ji.
    A gradient-reversal layer (coefficient lambda) sits before the head so its
    gradients do not dominate the shared representation.
    """

    def __init__(self, hidden: int, n_diseases: int, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_diseases, hidden) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        i, j = torch.triu_indices(n_diseases, n_diseases, offset=1)
        self.register_buffer("idx_i", i, persistent=False)
        self.register_buffer("idx_j", j, persistent=False)

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        """hidden [B,T,H], valid_mask [B,T] -> pair logits s_ij [B, n_pairs] (i<j)."""
        h = grad_reverse(hidden, grl_lambda)
        scores = torch.einsum("kh,bth->bkt", self.queries, h)             # [B,K,T]
        scores = scores.masked_fill(valid_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                              # [B,K,T]
        disease_emb = torch.einsum("bkt,bth->bkh", attn, h)               # [B,K,H]
        hi = disease_emb[:, self.idx_i]                                   # [B,P,H]
        hj = disease_emb[:, self.idx_j]                                   # [B,P,H]
        prod = hi * hj
        # symmetrize the scorer so the pair score is permutation invariant
        s = 0.5 * (self.scorer(torch.cat([hi, hj, prod], dim=-1))
                   + self.scorer(torch.cat([hj, hi, prod], dim=-1)))
        return s.squeeze(-1)                                              # [B, n_pairs]


# ---------------------------------------------------------------------------
# Losses / pooling utilities
# ---------------------------------------------------------------------------
def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    """Binary focal loss on 2-logit output. logits [*, 2], target [*] in {0,1}."""
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return alpha * (1 - pt) ** gamma * ce


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Mean of x over `dim`, weighting by mask (broadcast over trailing dims)."""
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    s = (x * m).sum(dim=dim)
    d = m.sum(dim=dim).clamp(min=1e-6)
    return s / d


# ---------------------------------------------------------------------------
# Top-level MoChiFormer
# ---------------------------------------------------------------------------
class MoChiFormer(nn.Module):
    def __init__(self, cfg: MoChiFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.visit_encoder = VisitEncoder(cfg)

        if cfg.use_vae:
            self.vae = VAEEncoder(cfg.hidden_size, cfg.latent_dim)
            self.vae_decoder = nn.Linear(cfg.latent_dim, cfg.hidden_size)

        self.time_embedding = TemporalPositionalEmbedding(cfg.max_time_days, cfg.hidden_size)

        gpt_cfg = GPT2Config(
            vocab_size=1,  # we always pass inputs_embeds; wte is unused
            n_positions=cfg.max_visits,
            n_embd=cfg.hidden_size,
            n_layer=cfg.decoder_layers,
            n_head=cfg.num_heads,
            n_inner=cfg.ffn_size,
            resid_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
        )
        self.temporal_decoder = GPT2Model(gpt_cfg)

        self.task_heads = FinetuneTaskHeads(cfg)
        self.recon_heads = PretrainReconHeads(cfg)

        if cfg.use_cohort_adv and cfg.n_cohorts >= 2:
            self.cohort_disc = CohortDiscriminator(cfg.hidden_size, cfg.n_cohorts, cfg.dropout)
        else:
            self.cohort_disc = None

        if cfg.use_pairwise_rel and cfg.n_cls_tasks >= 2:
            self.rel_head = PairwiseRelationHead(cfg.hidden_size, cfg.n_cls_tasks, cfg.dropout)
        else:
            self.rel_head = None

        self.grl_lambda = 0.0

    def set_grl_lambda(self, value: float) -> None:
        self.grl_lambda = float(value)

    def _shift_events(self, event_feats):
        """Anti-leakage temporal controls (paper, Stage 1 + architecture):
        shift the event channel one visit so visit t sees only events from j<t,
        and the current visit's diagnosis (event[t]) is replaced by [PAD].
        """
        if event_feats is None or event_feats.shape[-1] == 0:
            return event_feats
        shifted = torch.full_like(event_feats, -1)  # visit 0 -> all PAD
        shifted[:, 1:] = event_feats[:, :-1]
        return shifted

    # ---- shared visit encoding ------------------------------------------
    def _encode_visits(self, cat_feats, float_feats, event_feats):
        """cat/float/event: [B, T, n_*]. Returns (cls_emb [B,T,H], token_hidden [B,T,1+nf,H])."""
        b, t = cat_feats.shape[0], cat_feats.shape[1]
        cat_flat = cat_feats.reshape(b * t, -1)
        flt_flat = float_feats.reshape(b * t, -1)
        evt_flat = event_feats.reshape(b * t, -1)
        cls_emb, token_hidden = self.visit_encoder(cat_flat, flt_flat, evt_flat)
        h = self.cfg.hidden_size
        return cls_emb.reshape(b, t, h), token_hidden.reshape(b, t, 1 + self.cfg.n_feats, h)

    # ---- pretrain (visit-level masked reconstruction) -------------------
    def pretrain_forward(self, cat_feats, float_feats, event_feats, valid_mask) -> Dict[str, torch.Tensor]:
        # events enter as context (unshifted, unmasked); reconstruction targets are
        # the masked lab features only (categorical CE + numerical MSE).
        cls_emb, token_hidden = self._encode_visits(cat_feats, float_feats, event_feats)
        b, t = cat_feats.shape[0], cat_feats.shape[1]
        token_hidden_flat = token_hidden.reshape(b * t, 1 + self.cfg.n_feats, self.cfg.hidden_size)
        cat_logits, float_pred = self.recon_heads(
            token_hidden_flat, self.cfg.n_cat_feats, self.cfg.n_float_feats
        )
        out = {
            "cat_logits": cat_logits.reshape(b, t, self.cfg.n_cat_feats, self.cfg.n_cat_values),
            "float_pred": float_pred.reshape(b, t, self.cfg.n_float_feats),
        }
        if self.cfg.use_vae:
            mean, logvar = self.vae(cls_emb)
            kl = masked_mean(kl_divergence(mean, logvar), valid_mask).mean()
            out["kl"] = kl
        return out

    # ---- finetune / inference (patient-level) ---------------------------
    def forward(self, cat_feats, float_feats, event_feats, valid_mask, time_index,
                cohort_id=None, compute_aux: bool = True):
        # apply the anti-leakage event shift before encoding
        event_in = self._shift_events(event_feats)
        cls_emb, _ = self._encode_visits(cat_feats, float_feats, event_in)

        kl = cls_emb.new_zeros(())
        visit_emb = cls_emb
        if self.cfg.use_vae:
            mean, logvar = self.vae(cls_emb)
            # sample during training; use the posterior mean at inference for determinism
            z = reparameterize(mean, logvar) if self.training else mean
            visit_emb = self.vae_decoder(z)
            kl = masked_mean(kl_divergence(mean, logvar), valid_mask).mean()

        emb = visit_emb + self.time_embedding(time_index)
        dec = self.temporal_decoder(inputs_embeds=emb, attention_mask=valid_mask)
        hidden = dec.last_hidden_state  # [B, T, H], causal

        cls_logits, reg_preds = self.task_heads(hidden)

        # Training-only auxiliary heads (cohort-adversarial de-biasing and the
        # pairwise-relation head). Skipped at inference (compute_aux=False) to
        # avoid their attention-pooling / discriminator compute.
        cohort_logits = None
        rel_scores = None
        if compute_aux:
            if self.cohort_disc is not None:
                pooled = masked_mean(hidden, valid_mask)            # [B, H]
                cohort_logits = self.cohort_disc(grad_reverse(pooled, self.grl_lambda))
            if self.rel_head is not None:
                rel_scores = self.rel_head(hidden, valid_mask, self.grl_lambda)  # [B, n_pairs]

        return {
            "hidden": hidden,
            "visit_emb": visit_emb,
            "cls_logits": cls_logits,     # list of [B, T, 2]
            "reg_preds": reg_preds,       # list of [B, T]
            "cohort_logits": cohort_logits,
            "rel_scores": rel_scores,     # [B, n_pairs] disease-pair logits (i<j)
            "kl": kl,
        }

    # ---- convenience: patient-level predictions for inference ------------
    @torch.inference_mode()
    def predict_patient(self, cat_feats, float_feats, event_feats, valid_mask, time_index) -> Dict[str, torch.Tensor]:
        out = self.forward(cat_feats, float_feats, event_feats, valid_mask, time_index, compute_aux=False)
        # per-visit positive prob = sigmoid(logit_1 - logit_0); patient = masked mean over visits
        cls_probs = []
        for logits in out["cls_logits"]:
            p = torch.sigmoid(logits[..., 1] - logits[..., 0])   # [B, T]
            cls_probs.append(masked_mean(p, valid_mask))         # [B]
        reg_vals = [masked_mean(r, valid_mask) for r in out["reg_preds"]]  # each [B]
        patient_emb = masked_mean(out["hidden"], valid_mask)     # [B, H]
        return {
            "cls_probs": torch.stack(cls_probs, dim=-1) if cls_probs else None,  # [B, K]
            "reg_vals": torch.stack(reg_vals, dim=-1) if reg_vals else None,     # [B, R]
            "patient_emb": patient_emb,
        }
