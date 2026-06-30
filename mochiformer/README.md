# MoChiFormer (core model)

The real, trainable longitudinal-EHR foundation model that backs MoChiAgent's
transformer tool — replacing the previous random-number placeholder in
`server/transformer_server.py`. It ingests each patient's chronologically
ordered visits of discretized laboratory measurements and structured-EHR events.

It does three things: **train** (pretrain + finetune), **infer** (load a
checkpoint and score patients), and **serve** (drive the `transformer_server`
`predict` tool).

## Quickstart

```bash
cd mochiagent
PY=python   # use a Python env with torch + transformers installed

# 1. Train a self-contained synthetic demo -> writes checkpoints/mochiformer_demo.ckpt
$PY -m mochiformer.train demo --out checkpoints/mochiformer_demo.ckpt

# 2. End-to-end smoke test (train -> infer -> serve)
$PY tests/test_smoke.py

# 3. Serve: the transformer_server auto-loads checkpoints/mochiformer_demo.ckpt
#    (override with env MOCHIFORMER_CKPT=/path/to.ckpt). No code change needed.
```

## How it maps to the paper (Methods → code)

| Paper (Methods) | Code |
| --- | --- |
| Stage 1 discretization `D(x)`; CLS=0 / PAD=1 / value offsets; `-1` missing | `data.py` `FeatureSchema.discretize`, `model.VisitEncoder._build_input_ids` |
| Visit-level 2-layer BERT encoder, type embeddings, learned missing vector | `model.VisitEncoder` (`BertModel`, `position_embedding_type="none"`) |
| Patient-level GPT-2 temporal (causal) decoder + continuous time embedding | `model.MoChiFormer.temporal_decoder` (`GPT2Model`), `TemporalPositionalEmbedding` |
| Variational latent + KL (`z = μ + ε·exp(0.5 logσ²)`) | `model.VAEEncoder`, `reparameterize`, `kl_divergence` |
| Masked-feature reconstruction pretrain (CE cat + MSE num) | `model.PretrainReconHeads`, `train.pretrain` |
| Multi-disease focal loss + age regression (masked-mean pooling) | `model.focal_loss`, `FinetuneTaskHeads`, `train.finetune` |
| Adversarial cross-site batch-effect removal (GRL) | `model._GradReverse`, `CohortDiscriminator` |
| Anti-leakage temporal controls: current-visit event → [PAD], event sequence shifted one step vs lab (visit t sees only j<t) | `model.MoChiFormer._shift_events` (applied in `forward`) |
| Pairwise disease co-occurrence head: per-disease attention pooling, `z_ij=[h_i,h_j,h_i⊙h_j]`, symmetric scorer, GRL, BCE vs co-occurrence | `model.PairwiseRelationHead`, `train.finetune` (`lambda_rel`) |
| Patient prob = mean over visits of `σ(o₁−o₀)`; threshold/AUC eval | `model.predict_patient`, `train.evaluate` |

## Configuration

`config.py` holds every hyperparameter. `demo_config()` is small and
CPU-runnable. To match the paper's "Implementation details" set, override:
`hidden_size=768, visit_encoder_layers=2, decoder_layers=12, num_heads=12`,
plus `pretrain_epochs=200, finetune_epochs=100, lr` as reported.

## Training on real data

Replace `make_synthetic_cohort` with your own loader that yields a list of
`data.PatientRecord` (raw per-visit categorical codes, raw continuous labs with
`np.nan` for missing, day offsets, patient-level disease labels + age targets,
cohort id). Fit a `FeatureSchema` on the training split and reuse it everywhere.
Nothing else in the pipeline changes.

## Scope / known limitations (this "core model" step)

- **Raw free-text EHR strings are not tokenized** into structured event features
  by the `predict_raw` adapter; supply `cat_feats` / `event_codes` matrices for
  the categorical and diagnostic-event channels (the demo adapter treats the
  event channel as missing).
- The bundled `mochiformer_demo.ckpt` is trained on **synthetic** data, so real
  lab values fall outside its feature range and score ≈ 0.5 — that demonstrates
  the plumbing, not clinical performance. Train on real, schema-matched data for
  meaningful predictions.
- Out of scope for step 1 (and unchanged): the clustering / trajectory /
  knowledge-retrieval tools and the LLM orchestration.
