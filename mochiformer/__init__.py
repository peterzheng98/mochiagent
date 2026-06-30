"""
MoChiFormer — the core longitudinal-EHR foundation model for MoChiAgent.

This is a self-contained, inference-and-training-capable implementation of the
MoChiFormer architecture described in the paper (Methods, "The core prediction
model: MoChiFormer"). It ingests each patient's chronologically ordered visits
of discretized laboratory measurements and structured-EHR events.

Public API
----------
- MoChiFormerConfig, demo_config        (mochiformer.config)
- MoChiFormer                           (mochiformer.model)
- FeatureSchema, LongitudinalEHRDataset,
  make_synthetic_cohort, collate_visits (mochiformer.data)
- pretrain, finetune                    (mochiformer.train)
- MoChiFormerPredictor                  (mochiformer.inference)

See README.md in this folder for how it maps onto the paper and what is
intentionally out of scope for this "core model" step.
"""

from .config import MoChiFormerConfig, demo_config

__all__ = ["MoChiFormerConfig", "demo_config"]

__version__ = "0.1.0"
