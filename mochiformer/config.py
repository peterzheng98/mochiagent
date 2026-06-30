"""Configuration for MoChiFormer.

A single dataclass holds every architectural and training hyperparameter so the
exact configuration can be serialized into a checkpoint and restored verbatim at
inference time. ``demo_config()`` returns a small, CPU-friendly configuration
used by the smoke test; production runs override the relevant fields (e.g.
``hidden_size=768``, ``visit_encoder_layers=2``, ``decoder_layers=12`` to match
the paper's "Implementation details").
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class MoChiFormerConfig:
    # ---- feature schema dimensions (filled in from FeatureSchema) -----------
    n_cat_feats: int = 1          # number of categorical feature columns per visit
    n_float_feats: int = 8        # number of continuous (lab) feature columns per visit
    n_cat_values: int = 16        # size of the categorical value space (max cardinality)
    n_float_bins: int = 64        # d_cont: number of discretization bins for continuous features
    n_event_feats: int = 0        # number of structured-EHR diagnostic-event columns per visit
    n_event_values: int = 8       # size of the event value space

    # ---- visit-level encoder (BERT-style) -----------------------------------
    hidden_size: int = 128        # paper: 768
    visit_encoder_layers: int = 2 # paper: 2
    num_heads: int = 4            # paper: 12
    intermediate_size: int = 0    # 0 -> 4 * hidden_size
    dropout: float = 0.1

    # ---- patient-level temporal decoder (GPT-2-style) -----------------------
    decoder_layers: int = 4       # paper: 12
    max_visits: int = 64          # max sequence length (visits) the decoder sees
    max_time_days: int = 3650     # horizon for the continuous time embedding

    # ---- variational latent pathway (optional) ------------------------------
    use_vae: bool = False
    latent_dim: int = 64
    vae_kl_weight: float = 0.1    # beta in L_pretrain = L_recon + beta * L_KL

    # ---- cohort-adversarial de-biasing (batch-effect removal) ---------------
    use_cohort_adv: bool = False
    n_cohorts: int = 1
    cohort_adv_weight: float = 0.1
    grl_lambda_max: float = 1.0

    # ---- task heads ----------------------------------------------------------
    # classification targets (diseases): each is a binary head (2 logits)
    cls_target_names: List[str] = field(default_factory=lambda: ["disease_0"])
    # regression targets (biological ages): each is a scalar head
    reg_target_names: List[str] = field(default_factory=lambda: ["age"])
    # pairwise disease co-occurrence head (attention-pooled, GRL-gated)
    use_pairwise_rel: bool = False

    # ---- losses --------------------------------------------------------------
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    lambda_cls: float = 1.0
    lambda_reg: float = 1.0
    lambda_cohort: float = 1.0
    lambda_rel: float = 0.3

    # ---- pretraining ---------------------------------------------------------
    mask_ratio: float = 0.15      # fraction of valid features masked per visit

    # ---- optimization --------------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 16
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    seed: int = 42

    # ---- derived -------------------------------------------------------------
    @property
    def ffn_size(self) -> int:
        return self.intermediate_size if self.intermediate_size > 0 else 4 * self.hidden_size

    @property
    def n_feats(self) -> int:
        return self.n_cat_feats + self.n_float_feats + self.n_event_feats

    @property
    def value_vocab_size(self) -> int:
        # shared value-embedding table: cat + float-bin + event value spaces + 2
        # reserved ids (0 = [CLS], 1 = [PAD]).
        return self.n_cat_values + self.n_float_bins + self.n_event_values + 2

    @property
    def type_vocab_size(self) -> int:
        # one type id per feature column, plus the [CLS] type (id 0).
        return self.n_feats + 1

    @property
    def n_cls_tasks(self) -> int:
        return len(self.cls_target_names)

    @property
    def n_reg_tasks(self) -> int:
        return len(self.reg_target_names)

    # ---- (de)serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MoChiFormerConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def demo_config(**overrides: Any) -> MoChiFormerConfig:
    """Small, fast configuration for tests / CPU smoke runs."""
    cfg = MoChiFormerConfig(
        n_cat_feats=2,
        n_float_feats=8,
        n_cat_values=12,
        n_float_bins=32,
        n_event_feats=2,
        n_event_values=6,
        hidden_size=64,
        visit_encoder_layers=2,
        num_heads=4,
        decoder_layers=2,
        max_visits=24,
        use_vae=True,
        latent_dim=32,
        use_cohort_adv=True,
        n_cohorts=2,
        use_pairwise_rel=True,
        cls_target_names=["gdm", "preeclampsia", "miscarriage"],
        reg_target_names=["gestational_age"],
        batch_size=32,
        pretrain_epochs=5,
        finetune_epochs=20,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
