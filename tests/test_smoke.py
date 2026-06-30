"""End-to-end smoke test for the MoChiFormer core model.

Exercises all three required capabilities on synthetic data:
  1. TRAIN   -- pretrain + finetune produce a checkpoint and learn the planted
                signal (eval macro-AUC clearly above chance);
  2. INFER   -- MoChiFormerPredictor loads the checkpoint and scores a patient;
  3. SERVE   -- the transformer_server `predict` tool returns the real engine's
                output through its normal tool interface (no Redis required).

Run:  python tests/test_smoke.py        (from anywhere; needs torch + transformers)
"""

import os
import sys
import tempfile

# make the mochiagent package root importable regardless of CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

from mochiformer.config import demo_config
from mochiformer.train import train_demo
from mochiformer.inference import MoChiFormerPredictor


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_smoke() -> None:
    device = _device()
    cfg = demo_config(pretrain_epochs=3, finetune_epochs=12)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "mochiformer_smoke.ckpt")

        # 1) TRAIN ----------------------------------------------------------
        metrics = train_demo(ckpt, cfg=cfg, device=device, n_train=512, n_eval=256, log_every=0)
        assert os.path.exists(ckpt), "checkpoint was not written"
        auc = metrics["macro_auc"]
        assert auc == auc, "macro_auc is NaN"                       # not NaN
        assert auc > 0.55, f"model did not learn the planted signal (AUC={auc:.3f})"
        print(f"[1/3] TRAIN ok  -- eval macro_auc={auc:.3f}")

        # 2) INFER ----------------------------------------------------------
        predictor = MoChiFormerPredictor.from_checkpoint(ckpt, device=device)
        ehr = ["Patient 45M, history of T2DM.", "Presenting with fatigue and polyuria."]
        lab = [[100, 20, 30], [110, 22, 32], [105, 21, 31]]
        out = predictor.predict_raw(ehr, lab, return_embedding=True)
        assert set(["prediction_score", "risk_category", "confidence",
                    "disease_probabilities", "biological_age"]).issubset(out)
        assert 0.0 <= out["prediction_score"] <= 1.0
        assert len(out["disease_probabilities"]) == cfg.n_cls_tasks
        assert len(out["patient_embedding"]) == cfg.hidden_size
        assert out["n_visits"] == len(lab)
        print(f"[2/3] INFER ok  -- score={out['prediction_score']:.3f} "
              f"risk={out['risk_category']} top={out['top_disease']}")

        # 3) SERVE ----------------------------------------------------------
        from server.transformer_server import TransformerMCPServer

        class _NoRedisServer(TransformerMCPServer):
            # avoid any Redis side effects during the test
            def _register(self):  # noqa: D401
                pass

            def _create_redis_connection(self):
                return None

        srv = _NoRedisServer(model_path=ckpt, device=device)
        assert srv.predictor is not None, "server did not load the MoChiFormer engine"
        res = srv.call_tool("predict", {"ehr_data": ehr, "lab_tests": lab, "return_embeddings": True})
        assert res["status"] == "success"
        assert res["model_info"]["engine"] == "mochiformer"
        assert 0.0 <= res["prediction"]["prediction_score"] <= 1.0
        assert "patient_embedding" in res["embeddings"]
        print(f"[3/3] SERVE ok  -- engine={res['model_info']['engine']} "
              f"score={res['prediction']['prediction_score']:.3f}")

    print("\nSMOKE TEST PASSED")


def test_leakage_and_relation() -> None:
    """Directly verify the two paper-fidelity mechanisms behave correctly."""
    from mochiformer.config import demo_config
    from mochiformer.model import MoChiFormer

    cfg = demo_config()
    model = MoChiFormer(cfg).eval()

    # 1) Anti-leakage event shift: visit t may see only events from j < t, and the
    #    current visit's diagnosis (event[t]) is replaced by [PAD] (-1).
    ev = torch.arange(1, 1 + 4 * cfg.n_event_feats).reshape(1, 4, cfg.n_event_feats)
    shifted = model._shift_events(ev)
    assert (shifted[:, 0] == -1).all(), "visit 0 must be PADded (sees no event)"
    assert torch.equal(shifted[:, 1:], ev[:, :-1]), "visit t must see only event[t-1]"
    print("[leakage]  event-shift ok  -- visit 0 padded; visit t sees only j<t")

    # 2) Pairwise relation head: K*(K-1)/2 symmetric pair scores.
    B, T, K = 2, 4, cfg.n_cls_tasks
    cat = torch.zeros(B, T, cfg.n_cat_feats, dtype=torch.long)
    flt = torch.zeros(B, T, cfg.n_float_feats, dtype=torch.long)
    evt = torch.zeros(B, T, cfg.n_event_feats, dtype=torch.long)
    vm = torch.ones(B, T)
    ti = torch.arange(T).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        out = model(cat, flt, evt, vm, ti)
    assert out["rel_scores"] is not None
    assert out["rel_scores"].shape == (B, K * (K - 1) // 2), out["rel_scores"].shape
    print(f"[relation] head ok  -- {out['rel_scores'].shape[1]} pair scores for K={K} diseases")
    print("\nFIDELITY CHECKS PASSED")


if __name__ == "__main__":
    test_leakage_and_relation()
    test_smoke()
