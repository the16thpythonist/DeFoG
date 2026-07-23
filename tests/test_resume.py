"""
Resumable-checkpoint tests for multi-job chaining.

A chained foundation run must restore the FULL training state from ``last.ckpt``
-- optimizer moments, the EMA shadow, and the global step -- not silently restart
them (the two bugs that would make chained training subtly wrong). These tests:

- round-trip the EMACallback / TrainingMonitorCallback state dicts;
- train a tiny model, then assert the written checkpoint actually contains the
  optimizer state + EMA shadow + step counter;
- resume from it and assert no extra training happened (weights == checkpoint),
  proving the run continued rather than starting over.

CPU-only and fast.
"""
import os

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from defog.core import DeFoGModel, EMACallback, TrainingMonitorCallback
from defog.domains.molecule import build_encoders, smiles_to_pyg_data

ATOM = ["C", "N", "O", "F"]


def _loader():
    ae, _, be, _ = build_encoders(ATOM, ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    smis = ["CCO", "CCN", "CC=O", "c1ccccc1", "CCC", "CCF", "CC(C)O", "CCOCC"]
    data = [d for d in (smiles_to_pyg_data(s, ae, be) for s in smis) if d is not None]
    return DataLoader(data, batch_size=4, shuffle=False)


def _model():
    return DeFoGModel(
        num_node_classes=4, num_edge_classes=5, n_layers=2, hidden_dim=32,
        hidden_mlp_dim=64, n_heads=2, noise_type="uniform", max_nodes=12,
        extra_features_type="rrwp", rrwp_steps=3, lr=1e-3, lr_scheduler="cosine",
    )


def _trainer(**kw):
    return pl.Trainer(
        accelerator="cpu", devices=1, enable_progress_bar=False, logger=False,
        limit_val_batches=0, num_sanity_val_steps=0, **kw,
    )


def test_ema_state_roundtrip():
    ema = EMACallback(decay=0.99)
    ema.shadow = {"w": torch.randn(3, 3), "b": torch.ones(3)}
    ema2 = EMACallback(decay=0.5)
    ema2.load_state_dict(ema.state_dict())
    assert ema2.decay == 0.99
    assert set(ema2.shadow) == {"w", "b"}
    assert torch.allclose(ema2.shadow["w"], ema.shadow["w"])


def test_monitor_state_roundtrip():
    m = TrainingMonitorCallback()
    m.best_validity = 0.73
    m.history["val_loss"].extend([1.0, 0.9])
    m2 = TrainingMonitorCallback()
    m2.load_state_dict(m.state_dict())
    assert m2.best_validity == 0.73
    assert list(m2.history["val_loss"]) == [1.0, 0.9]


def test_checkpoint_contains_full_state(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    pl.seed_everything(0)
    m1 = _model()
    cb = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=0,
                         every_n_train_steps=3)
    _trainer(max_steps=6, enable_checkpointing=True,
             callbacks=[EMACallback(decay=0.9), cb]).fit(m1, train_dataloaders=_loader())

    last = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(last)
    ck = torch.load(last, map_location="cpu", weights_only=False)
    # step counter, optimizer moments, and EMA shadow are all in the checkpoint
    assert ck["global_step"] == 6
    assert ck["optimizer_states"] and ck["optimizer_states"][0]["state"]
    ema_keys = [k for k in ck.get("callbacks", {}) if "EMACallback" in k]
    assert ema_keys and ck["callbacks"][ema_keys[0]]["shadow"]


def test_resume_continues_not_restarts(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    # chunk 1: 6 steps -> last.ckpt (seed 0)
    pl.seed_everything(0)
    m1 = _model()
    cb = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=0,
                         every_n_train_steps=3)
    _trainer(max_steps=6, enable_checkpointing=True,
             callbacks=[EMACallback(decay=0.9), cb]).fit(m1, train_dataloaders=_loader())
    last = os.path.join(ckpt_dir, "last.ckpt")
    ck = torch.load(last, map_location="cpu", weights_only=False)
    saved_shadow = ck["callbacks"][[k for k in ck["callbacks"] if "EMACallback" in k][0]]["shadow"]
    a_key = next(iter(saved_shadow))

    # chunk 2: DIFFERENT seed (so a broken resume would reinit from a different
    # init), resume from last.ckpt with max_steps already reached -> zero extra
    # steps. A correct resume restores the EMA shadow from the checkpoint.
    pl.seed_everything(123)
    m2 = _model()
    fresh_param = dict(m2.named_parameters())[a_key].detach().clone()
    ema2 = EMACallback(decay=0.9)
    t2 = _trainer(max_steps=6, enable_checkpointing=False, callbacks=[ema2])
    t2.fit(m2, train_dataloaders=_loader(), ckpt_path=last)

    assert t2.global_step == 6                       # continued from the checkpoint
    assert ema2.shadow                               # EMA shadow restored (non-empty)
    # restored shadow == checkpoint's, and is NOT this run's fresh (seed-123) init
    assert torch.allclose(ema2.shadow[a_key].float(), saved_shadow[a_key].float(), atol=1e-6)
    assert not torch.allclose(ema2.shadow[a_key].float(), fresh_param.float(), atol=1e-4)
