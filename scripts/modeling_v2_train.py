#!/usr/bin/env python3
"""Train leak-free 5-fold baseline & v2 models + ensemble and run Test.csv inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import torch before GBDT C extensions to prevent macOS OpenMP initialization conflicts.
import torch
import torch.nn as nn
import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_mixtures_v1 import BANDS  # noqa: E402
from modeling_v2_features import (  # noqa: E402
    build_cnn_tensors,
    compute_physics_baseline_share,
    extract_features,
    extract_v1_combined96_features,
)

DATA_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v2"
MODELS_DIR_DEFAULT = ROOT / "models/v2"
REPORTS_DIR_DEFAULT = ROOT / "reports/modeling_v2"
SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v2_ensemble.csv"
TEST_CSV_DEFAULT = ROOT / "data/zindi/raw/Test.csv"
SAMPLE_SUB_DEFAULT = ROOT / "data/zindi/raw/sample_submission.csv"

TEST_COLS = [f"m{m:02d}_{b}" for m in range(1, 16) for b in BANDS]
LEARNED_V2_MODELS = ["RF_v2", "Linear", "HistGBM", "LightGBM", "XGBoost", "CatBoost", "CNN2D"]
ALL_BENCHMARK_MODELS = ["Constant_Mean", "Physics_q15", "V1_RF_MaxShift"] + LEARNED_V2_MODELS + ["Ensemble_v2"]


class AcousticResBlock(nn.Module):
    """Residual 2D convolutional block over (time, band) spectrogram grid."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.proj(x))


class AcousticCNN2D(nn.Module):
    """Mask-aware 2D spatiotemporal CNN predicting share_added_vessel and presence logit."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
        )
        self.block1 = AcousticResBlock(24, 40)
        self.block2 = AcousticResBlock(40, 64)
        self.drop = nn.Dropout(0.20)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.share_out = nn.Linear(64, 1)
        self.score_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = x[:, 3:4, :, :]
        h = self.block2(self.block1(self.stem(x)))
        h_masked = h * mask
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        avg_pool = h_masked.sum(dim=(2, 3)) / denom
        max_pool = (h_masked + (mask - 1.0) * 1e4).amax(dim=(2, 3))
        feat = self.head(self.drop(torch.cat([avg_pool, max_pool], dim=1)))
        share = torch.sigmoid(self.share_out(feat)).squeeze(-1)
        score_logit = self.score_out(feat).squeeze(-1)
        return share, score_logit


def seed_everything(seed: int = 42, torch_threads: int = 4) -> None:
    """Lock Python, NumPy, and PyTorch RNG state and deterministic CPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, torch_threads))


def _select_torch_device(device_pref: str = "cpu") -> torch.device:
    if device_pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    return torch.device("cpu")


def fit_predict_cnn_fold(
    T_tr: np.ndarray,
    y_share_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    T_es: np.ndarray,
    y_share_es: np.ndarray,
    y_pos_es: np.ndarray,
    eval_tensors: list[np.ndarray],
    seed: int,
    epochs: int = 24,
    batch_size: int = 128,
    lr: float = 1.8e-3,
    device_pref: str = "cpu",
) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]]]:
    """Train one 2D CNN fold strictly on internal splits with bit-exact determinism."""
    seed_everything(seed, torch_threads=4)
    device = _select_torch_device(device_pref=device_pref)
    model = AcousticCNN2D(in_channels=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    x_tr_t = torch.from_numpy(T_tr).to(device)
    y_s_t = torch.from_numpy(y_share_tr.astype(np.float32)).to(device)
    y_p_t = torch.from_numpy(y_pos_tr.astype(np.float32)).to(device)
    x_es_t = torch.from_numpy(T_es).to(device)
    y_s_es_t = torch.from_numpy(y_share_es.astype(np.float32)).to(device)
    y_p_es_t = torch.from_numpy(y_pos_es.astype(np.float32)).to(device)

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(seed)
    n_tr = len(T_tr)

    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            sh_p, sc_l = model(x_tr_t[idx])
            loss = ((sh_p - y_s_t[idx]) ** 2).mean() + 0.12 * bce(sc_l, y_p_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            sh_e, sc_e = model(x_es_t)
            val_loss = float(((sh_e - y_s_es_t) ** 2).mean().item() + 0.12 * bce(sc_e, y_p_es_t).item())
            if np.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    preds: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for arr in eval_tensors:
            sh_parts, sc_parts = [], []
            for s in range(0, len(arr), 512):
                xb = torch.from_numpy(arr[s : s + 512]).to(device)
                sh_o, sc_o = model(xb)
                sh_parts.append(sh_o.detach().cpu().numpy().astype(np.float64))
                sc_parts.append(torch.sigmoid(sc_o).detach().cpu().numpy().astype(np.float64))
            preds.append((np.concatenate(sh_parts), np.concatenate(sc_parts)))
    return best_state, preds


def fit_simplex_weights_rmse(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Solve non-negative convex weights minimizing MSE strictly on OOF predictions."""
    m = oof_matrix.shape[1]
    init = np.full(m, 1.0 / m)

    def _loss(w: np.ndarray) -> float:
        pred = np.clip(oof_matrix @ w, 0.0, 1.0)
        return float(np.mean((pred - y_true) ** 2))

    res = minimize(
        _loss,
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = np.maximum(res.x, 0.0)
    return w / np.maximum(w.sum(), 1e-12)


def fit_simplex_weights_logloss(oof_probs: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    """Solve non-negative convex weights on clipped logits minimizing binary cross-entropy on OOF."""
    m = oof_probs.shape[1]
    p_clip = np.clip(oof_probs, 1e-5, 1.0 - 1e-5)
    logits = np.log(p_clip / (1.0 - p_clip))
    init = np.full(m, 1.0 / m)

    def _loss(w: np.ndarray) -> float:
        comb_logit = logits @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(comb_logit, -20.0, 20.0)))
        return float(-np.mean(y_bin * np.log(np.clip(p, 1e-7, 1.0)) + (1 - y_bin) * np.log(np.clip(1.0 - p, 1e-7, 1.0))))

    res = minimize(
        _loss,
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = np.maximum(res.x, 0.0)
    return w / np.maximum(w.sum(), 1e-12)


def apply_logit_blend(probs_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine probability columns in logit space using simplex weights."""
    p_clip = np.clip(probs_matrix, 1e-5, 1.0 - 1e-5)
    logits = np.log(p_clip / (1.0 - p_clip)) @ weights
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0))), 0.0, 1.0)


def compute_metrics(split: str, model: str, y_share: np.ndarray, pred_share: np.ndarray, pred_score: np.ndarray) -> dict:
    """Compute competition and diagnostic metrics for one model on one split."""
    y_bin = (y_share > 0.0).astype(int)
    p_sh = np.clip(pred_share, 0.0, 1.0)
    p_sc = np.clip(pred_score, 0.0, 1.0)
    rmse = float(root_mean_squared_error(y_share, p_sh))
    zero_mask = y_share == 0.0
    zero_rmse = float(np.sqrt(np.mean(p_sh[zero_mask] ** 2))) if zero_mask.any() else 0.0
    auc = 0.5 if len(np.unique(y_bin)) < 2 else float(roc_auc_score(y_bin, p_sc))
    pr_auc = float(y_bin.mean()) if len(np.unique(y_bin)) < 2 else float(average_precision_score(y_bin, p_sc))
    return {
        "split": split,
        "model": model,
        "n": int(len(y_share)),
        "zindi_score_1_minus_rmse": float(1.0 - rmse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_share, p_sh)),
        "r2": float(r2_score(y_share, p_sh)),
        "zero_rmse": zero_rmse,
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }


def eval_constant_mean(y_tr: np.ndarray, y_va: np.ndarray, y_ho: np.ndarray, folds_tr: np.ndarray) -> dict:
    """Zero-skill Constant Mean baseline fit strictly inside train folds."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    y_pos = (y_tr > 0.0).astype(np.float64)
    for k in range(5):
        m = folds_tr == k
        oof_sh[m] = float(y_tr[~m].mean())
        oof_sc[m] = float(y_pos[~m].mean())
    va_sh = np.full(len(y_va), float(y_tr.mean()), dtype=np.float64)
    va_sc = np.full(len(y_va), float(y_pos.mean()), dtype=np.float64)
    ho_sh = np.full(len(y_ho), float(y_tr.mean()), dtype=np.float64)
    ho_sc = np.full(len(y_ho), float(y_pos.mean()), dtype=np.float64)
    return {
        "name": "Constant_Mean",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "elapsed_s": time.perf_counter() - t0,
    }


def eval_physics_q15(X_tr: np.ndarray, X_va: np.ndarray, X_ho: np.ndarray) -> dict:
    """Zero-training analytical Physics_q15 excess-share estimator."""
    t0 = time.perf_counter()
    tr_sh, tr_sc = compute_physics_baseline_share(X_tr, q=15.0)
    va_sh, va_sc = compute_physics_baseline_share(X_va, q=15.0)
    ho_sh, ho_sc = compute_physics_baseline_share(X_ho, q=15.0)
    return {
        "name": "Physics_q15",
        "oof_share": tr_sh,
        "oof_score": tr_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_random_forest(
    name: str,
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
    seed: int = 42,
) -> dict:
    """5-fold RandomForest regressor & classifier (used for V1_RF_MaxShift and RF_v2)."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        val_mask = folds_tr == k
        tr_mask = ~val_mask
        reg = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            max_samples=0.8,
            n_jobs=1,
            random_state=seed + k,
        )
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=5,
            max_features="sqrt",
            max_samples=0.8,
            n_jobs=1,
            random_state=seed + k,
        )
        reg.fit(F_tr[tr_mask], y_tr[tr_mask])
        clf.fit(F_tr[tr_mask], y_pos_tr[tr_mask])
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": name,
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_linear(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
) -> dict:
    """5-fold StandardScaler + Ridge / LogisticRegression."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        val_mask = folds_tr == k
        tr_mask = ~val_mask
        reg = make_pipeline(StandardScaler(), Ridge(alpha=30.0, solver="cholesky"))
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, solver="lbfgs", max_iter=1000, random_state=42 + k))
        reg.fit(F_tr[tr_mask], y_tr[tr_mask])
        clf.fit(F_tr[tr_mask], y_pos_tr[tr_mask])
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": "Linear",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_histgbm(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
) -> dict:
    """5-fold scikit-learn HistGradientBoosting regressor & classifier."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        val_mask = folds_tr == k
        tr_mask = ~val_mask
        reg = HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, learning_rate=0.05,
            l2_regularization=2.0, early_stopping=True, validation_fraction=0.12, random_state=42 + k,
        )
        clf = HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, learning_rate=0.05,
            l2_regularization=2.0, early_stopping=True, validation_fraction=0.12, random_state=42 + k,
        )
        reg.fit(F_tr[tr_mask], y_tr[tr_mask])
        clf.fit(F_tr[tr_mask], y_pos_tr[tr_mask])
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": "HistGBM",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_lightgbm(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
    n_jobs: int = 1,
) -> dict:
    """5-fold LightGBM with zero-leakage (k+1)%5 internal early stopping."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    gains = np.zeros(F_tr.shape[1], dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        es_fold = (k + 1) % 5
        val_mask = folds_tr == k
        es_mask = folds_tr == es_fold
        fit_mask = (~val_mask) & (~es_mask)
        reg = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            deterministic=True, force_col_wise=True, feature_fraction_seed=42 + k,
            bagging_seed=42 + k, data_random_seed=42 + k, n_jobs=n_jobs, random_state=42 + k, verbose=-1,
        )
        clf = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            deterministic=True, force_col_wise=True, feature_fraction_seed=42 + k,
            bagging_seed=42 + k, data_random_seed=42 + k, n_jobs=n_jobs, random_state=42 + k, verbose=-1,
        )
        reg.fit(
            F_tr[fit_mask], y_tr[fit_mask],
            eval_X=F_tr[es_mask], eval_y=y_tr[es_mask],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        clf.fit(
            F_tr[fit_mask], y_pos_tr[fit_mask],
            eval_X=F_tr[es_mask], eval_y=y_pos_tr[es_mask],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        imp = reg.booster_.feature_importance(importance_type="gain")
        gains += imp / max(float(imp.sum()), 1e-12) / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": "LightGBM",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "imp": gains,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_xgboost(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
    n_jobs: int = 1,
) -> dict:
    """5-fold XGBoost hist with zero-leakage (k+1)%5 internal early stopping."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    gains = np.zeros(F_tr.shape[1], dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        es_fold = (k + 1) % 5
        val_mask = folds_tr == k
        es_mask = folds_tr == es_fold
        fit_mask = (~val_mask) & (~es_mask)
        reg = xgb.XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=5, min_child_weight=12,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            tree_method="hist", early_stopping_rounds=50, n_jobs=n_jobs, random_state=42 + k,
        )
        clf = xgb.XGBClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=5, min_child_weight=12,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            tree_method="hist", eval_metric="logloss", early_stopping_rounds=50, n_jobs=n_jobs, random_state=42 + k,
        )
        reg.fit(F_tr[fit_mask], y_tr[fit_mask], eval_set=[(F_tr[es_mask], y_tr[es_mask])], verbose=False)
        clf.fit(F_tr[fit_mask], y_pos_tr[fit_mask], eval_set=[(F_tr[es_mask], y_pos_tr[es_mask])], verbose=False)
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        imp = reg.feature_importances_
        gains += imp / max(float(imp.sum()), 1e-12) / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": "XGBoost",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "imp": gains,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_catboost(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    F_va: np.ndarray,
    F_ho: np.ndarray,
    folds_tr: np.ndarray,
    thread_count: int = 4,
) -> dict:
    """5-fold CatBoost with zero-leakage (k+1)%5 internal early stopping."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(F_va), dtype=np.float64)
    va_sc = np.zeros(len(F_va), dtype=np.float64)
    ho_sh = np.zeros(len(F_ho), dtype=np.float64)
    ho_sc = np.zeros(len(F_ho), dtype=np.float64)
    imps = np.zeros(F_tr.shape[1], dtype=np.float64)
    regs, clfs = [], []

    for k in range(5):
        es_fold = (k + 1) % 5
        val_mask = folds_tr == k
        es_mask = folds_tr == es_fold
        fit_mask = (~val_mask) & (~es_mask)
        reg = cb.CatBoostRegressor(
            iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=6.0,
            thread_count=thread_count, random_seed=42 + k, verbose=0,
        )
        clf = cb.CatBoostClassifier(
            iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=6.0,
            thread_count=thread_count, random_seed=42 + k, verbose=0,
        )
        reg.fit(F_tr[fit_mask], y_tr[fit_mask], eval_set=(F_tr[es_mask], y_tr[es_mask]), early_stopping_rounds=50, verbose=False)
        clf.fit(F_tr[fit_mask], y_pos_tr[fit_mask], eval_set=(F_tr[es_mask], y_pos_tr[es_mask]), early_stopping_rounds=50, verbose=False)
        oof_sh[val_mask] = np.clip(reg.predict(F_tr[val_mask]), 0.0, 1.0)
        oof_sc[val_mask] = clf.predict_proba(F_tr[val_mask])[:, 1]
        va_sh += np.clip(reg.predict(F_va), 0.0, 1.0) / 5.0
        va_sc += clf.predict_proba(F_va)[:, 1] / 5.0
        ho_sh += np.clip(reg.predict(F_ho), 0.0, 1.0) / 5.0
        ho_sc += clf.predict_proba(F_ho)[:, 1] / 5.0
        imp = np.asarray(reg.get_feature_importance(), dtype=np.float64)
        imps += imp / max(float(imp.sum()), 1e-12) / 5.0
        regs.append(reg)
        clfs.append(clf)

    return {
        "name": "CatBoost",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_regs": regs,
        "fold_clfs": clfs,
        "imp": imps,
        "elapsed_s": time.perf_counter() - t0,
    }


def train_cv_cnn2d(
    T_tr: np.ndarray,
    y_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    T_va: np.ndarray,
    T_ho: np.ndarray,
    folds_tr: np.ndarray,
    epochs: int = 24,
    batch_size: int = 128,
    device_pref: str = "cpu",
) -> dict:
    """5-fold PyTorch AcousticCNN2D with zero-leakage (k+1)%5 early stopping."""
    t0 = time.perf_counter()
    oof_sh = np.zeros(len(y_tr), dtype=np.float64)
    oof_sc = np.zeros(len(y_tr), dtype=np.float64)
    va_sh = np.zeros(len(T_va), dtype=np.float64)
    va_sc = np.zeros(len(T_va), dtype=np.float64)
    ho_sh = np.zeros(len(T_ho), dtype=np.float64)
    ho_sc = np.zeros(len(T_ho), dtype=np.float64)
    states = []

    for k in range(5):
        es_fold = (k + 1) % 5
        val_mask = folds_tr == k
        es_mask = folds_tr == es_fold
        fit_mask = (~val_mask) & (~es_mask)
        state, preds = fit_predict_cnn_fold(
            T_tr[fit_mask], y_tr[fit_mask], y_pos_tr[fit_mask],
            T_tr[es_mask], y_tr[es_mask], y_pos_tr[es_mask],
            [T_tr[val_mask], T_va, T_ho],
            seed=100 + k,
            epochs=epochs,
            batch_size=batch_size,
            device_pref=device_pref,
        )
        oof_sh[val_mask], oof_sc[val_mask] = preds[0]
        va_sh += preds[1][0] / 5.0
        va_sc += preds[1][1] / 5.0
        ho_sh += preds[2][0] / 5.0
        ho_sc += preds[2][1] / 5.0
        states.append(state)

    return {
        "name": "CNN2D",
        "oof_share": oof_sh,
        "oof_score": oof_sc,
        "val_share": va_sh,
        "val_score": va_sc,
        "ho_share": ho_sh,
        "ho_score": ho_sc,
        "fold_states": states,
        "elapsed_s": time.perf_counter() - t0,
    }


def predict_from_bundle(bundle: dict, X_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure inference helper using saved 5-fold bundle with zero fitting."""
    F = extract_features(X_db)[bundle["feature_names"]].to_numpy(dtype=np.float64)
    T = build_cnn_tensors(X_db)
    n = len(X_db)
    model_names = bundle["meta_model_names"]
    share_preds = {m: np.zeros(n, dtype=np.float64) for m in model_names}
    score_preds = {m: np.zeros(n, dtype=np.float64) for m in model_names}
    k_folds = len(bundle["folds"])
    key_map = {
        "RF_v2": ("rf_reg", "rf_clf"),
        "Linear": ("linear_reg", "linear_clf"),
        "HistGBM": ("hist_reg", "hist_clf"),
        "LightGBM": ("lgb_reg", "lgb_clf"),
        "XGBoost": ("xgb_reg", "xgb_clf"),
        "CatBoost": ("cb_reg", "cb_clf"),
    }

    for fold_art in bundle["folds"]:
        for m_name, (r_key, c_key) in key_map.items():
            if m_name in share_preds and r_key in fold_art:
                share_preds[m_name] += np.clip(fold_art[r_key].predict(F), 0.0, 1.0) / k_folds
                score_preds[m_name] += fold_art[c_key].predict_proba(F)[:, 1] / k_folds

        cnn = AcousticCNN2D(in_channels=4)
        cnn.load_state_dict(fold_art["cnn_state"])
        cnn.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(T)
            sh_list, sc_list = [], []
            for s in range(0, n, 512):
                sh_o, sc_o = cnn(t_in[s : s + 512])
                sh_list.append(sh_o.cpu().numpy())
                sc_list.append(torch.sigmoid(sc_o).cpu().numpy())
            share_preds["CNN2D"] += np.concatenate(sh_list) / k_folds
            score_preds["CNN2D"] += np.concatenate(sc_list) / k_folds

    sh_mat = np.column_stack([share_preds[m] for m in model_names])
    sc_mat = np.column_stack([score_preds[m] for m in model_names])
    w_sh = np.array(bundle["weights_share"], dtype=np.float64)
    w_sc = np.array(bundle["weights_score"], dtype=np.float64)
    return np.clip(sh_mat @ w_sh, 0.0, 1.0), apply_logit_blend(sc_mat, w_sc)


def fit_ensemble_and_export(
    results: dict[str, dict],
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_ho: np.ndarray,
    feat_names: list[str],
    data_dir: Path = DATA_DIR_DEFAULT,
    models_dir: Path = MODELS_DIR_DEFAULT,
    reports_dir: Path = REPORTS_DIR_DEFAULT,
    submission_path: Path = SUBMISSION_DEFAULT,
    test_csv_path: Path = TEST_CSV_DEFAULT,
    sample_sub_path: Path = SAMPLE_SUB_DEFAULT,
) -> tuple[pd.DataFrame, dict, dict]:
    """Fit OOF simplex meta-weights strictly inside train, evaluate all 11 models, save reports & Test CSV."""
    t0 = time.perf_counter()
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    y_pos_tr = (y_tr > 0.0).astype(np.int64)
    oof_sh_mat = np.column_stack([results[m]["oof_share"] for m in LEARNED_V2_MODELS])
    oof_sc_mat = np.column_stack([results[m]["oof_score"] for m in LEARNED_V2_MODELS])
    w_share = fit_simplex_weights_rmse(oof_sh_mat, y_tr)
    w_score = fit_simplex_weights_logloss(oof_sc_mat, y_pos_tr)

    results["Ensemble_v2"] = {
        "name": "Ensemble_v2",
        "oof_share": np.clip(oof_sh_mat @ w_share, 0.0, 1.0),
        "oof_score": apply_logit_blend(oof_sc_mat, w_score),
        "val_share": np.clip(np.column_stack([results[m]["val_share"] for m in LEARNED_V2_MODELS]) @ w_share, 0.0, 1.0),
        "val_score": apply_logit_blend(np.column_stack([results[m]["val_score"] for m in LEARNED_V2_MODELS]), w_score),
        "ho_share": np.clip(np.column_stack([results[m]["ho_share"] for m in LEARNED_V2_MODELS]) @ w_share, 0.0, 1.0),
        "ho_score": apply_logit_blend(np.column_stack([results[m]["ho_score"] for m in LEARNED_V2_MODELS]), w_score),
        "elapsed_s": time.perf_counter() - t0,
    }

    metric_rows = []
    for m in ALL_BENCHMARK_MODELS:
        r = results[m]
        metric_rows.append(compute_metrics("train_oof", m, y_tr, r["oof_share"], r["oof_score"]))
        metric_rows.append(compute_metrics("val", m, y_va, r["val_share"], r["val_score"]))
        metric_rows.append(compute_metrics("holdout", m, y_ho, r["ho_share"], r["ho_score"]))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(reports_dir / "benchmark_metrics.csv", index=False)

    weights_payload = {
        "models": LEARNED_V2_MODELS,
        "share_weights": {m: float(w) for m, w in zip(LEARNED_V2_MODELS, w_share)},
        "score_weights": {m: float(w) for m, w in zip(LEARNED_V2_MODELS, w_score)},
    }
    (reports_dir / "ensemble_weights.json").write_text(json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8")

    lgb_g = results["LightGBM"]["imp"]
    xgb_g = results["XGBoost"]["imp"]
    cb_i = results["CatBoost"]["imp"]
    imp_df = pd.DataFrame(
        {
            "feature": feat_names,
            "lgb_gain_norm": lgb_g,
            "xgb_gain_norm": xgb_g,
            "cb_imp_norm": cb_i,
            "mean_norm_importance": (lgb_g + xgb_g + cb_i) / 3.0,
        }
    ).sort_values("mean_norm_importance", ascending=False)
    imp_df.to_csv(reports_dir / "feature_importance.csv", index=False)

    for sp_name, y_sp, key_prefix, out_name in [
        ("val", y_va, "val", "val_predictions.parquet"),
        ("holdout", y_ho, "ho", "holdout_predictions.parquet"),
    ]:
        prov = pd.read_csv(data_dir / f"{sp_name}_provenance.csv")
        prov["y_true"] = y_sp
        for m in ALL_BENCHMARK_MODELS:
            prov[f"pred_share_{m}"] = results[m][f"{key_prefix}_share"]
            prov[f"pred_score_{m}"] = results[m][f"{key_prefix}_score"]
        prov.to_parquet(reports_dir / out_name, index=False)

    st_rows = []
    for sp_name in ("val", "holdout"):
        df_p = pd.read_parquet(reports_dir / f"{sp_name}_predictions.parquet")
        for station, grp in df_p.groupby("station", sort=True):
            for m in ("V1_RF_MaxShift", "Physics_q15", "Ensemble_v2"):
                met = compute_metrics(
                    sp_name,
                    m,
                    grp["y_true"].to_numpy(),
                    grp[f"pred_share_{m}"].to_numpy(),
                    grp[f"pred_score_{m}"].to_numpy(),
                )
                met["station"] = station
                st_rows.append(met)
    pd.DataFrame(st_rows).to_csv(reports_dir / "station_breakdown.csv", index=False)

    fold_artifacts = []
    for k in range(5):
        fold_artifacts.append(
            {
                "fold": k,
                "rf_reg": results["RF_v2"]["fold_regs"][k],
                "rf_clf": results["RF_v2"]["fold_clfs"][k],
                "linear_reg": results["Linear"]["fold_regs"][k],
                "linear_clf": results["Linear"]["fold_clfs"][k],
                "hist_reg": results["HistGBM"]["fold_regs"][k],
                "hist_clf": results["HistGBM"]["fold_clfs"][k],
                "lgb_reg": results["LightGBM"]["fold_regs"][k],
                "lgb_clf": results["LightGBM"]["fold_clfs"][k],
                "xgb_reg": results["XGBoost"]["fold_regs"][k],
                "xgb_clf": results["XGBoost"]["fold_clfs"][k],
                "cb_reg": results["CatBoost"]["fold_regs"][k],
                "cb_clf": results["CatBoost"]["fold_clfs"][k],
                "cnn_state": results["CNN2D"]["fold_states"][k],
            }
        )

    bundle = {
        "feature_names": feat_names,
        "meta_model_names": LEARNED_V2_MODELS,
        "weights_share": w_share.tolist(),
        "weights_score": w_score.tolist(),
        "folds": fold_artifacts,
    }
    joblib.dump(bundle, models_dir / "bundle.joblib")

    test_df = pd.read_csv(test_csv_path)
    sample_df = pd.read_csv(sample_sub_path)
    assert list(test_df["ID"]) == list(sample_df["ID"]), "Test ID alignment mismatch"
    X_test = test_df[TEST_COLS].to_numpy(dtype=np.float32).reshape(len(test_df), 15, 25)
    test_share, test_score = predict_from_bundle(bundle, X_test)

    sub_df = pd.DataFrame(
        {
            "ID": sample_df["ID"],
            "share_added_vessel": np.clip(test_share, 0.0, 1.0),
            "score_added_vessel": np.clip(test_score, 0.0, 1.0),
        }
    )
    assert len(sub_df) == len(sample_df)
    assert list(sub_df.columns) == list(sample_df.columns)
    assert np.isfinite(sub_df["share_added_vessel"]).all() and np.isfinite(sub_df["score_added_vessel"]).all()
    assert sub_df["share_added_vessel"].between(0.0, 1.0).all()
    assert sub_df["score_added_vessel"].between(0.0, 1.0).all()
    sub_df.to_csv(submission_path, index=False)

    sanity = {
        "submission_path": str(submission_path.relative_to(ROOT)),
        "rows": int(len(sub_df)),
        "share_mean": float(sub_df["share_added_vessel"].mean()),
        "share_std": float(sub_df["share_added_vessel"].std()),
        "share_q05_q50_q95": [float(x) for x in np.quantile(sub_df["share_added_vessel"], [0.05, 0.50, 0.95])],
        "score_mean": float(sub_df["score_added_vessel"].mean()),
        "score_std": float(sub_df["score_added_vessel"].std()),
        "score_q05_q50_q95": [float(x) for x in np.quantile(sub_df["score_added_vessel"], [0.05, 0.50, 0.95])],
        "val_ensemble_score": float(metrics_df.loc[(metrics_df["split"] == "val") & (metrics_df["model"] == "Ensemble_v2"), "zindi_score_1_minus_rmse"].iloc[0]),
        "val_ensemble_auc": float(metrics_df.loc[(metrics_df["split"] == "val") & (metrics_df["model"] == "Ensemble_v2"), "roc_auc"].iloc[0]),
        "holdout_ensemble_score": float(metrics_df.loc[(metrics_df["split"] == "holdout") & (metrics_df["model"] == "Ensemble_v2"), "zindi_score_1_minus_rmse"].iloc[0]),
        "holdout_ensemble_auc": float(metrics_df.loc[(metrics_df["split"] == "holdout") & (metrics_df["model"] == "Ensemble_v2"), "roc_auc"].iloc[0]),
    }
    (reports_dir / "sanity_summary.json").write_text(json.dumps(sanity, indent=2) + "\n", encoding="utf-8")
    return metrics_df, weights_payload, sanity


def run_self_test() -> None:
    """Verify bit-exact repeat determinism across every model family + simplex weights."""
    seed_everything(42, torch_threads=4)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(240, 20)).astype(np.float64)
    y = np.clip(0.35 + 0.15 * X[:, 0] - 0.1 * X[:, 1] + rng.normal(0, 0.05, size=240), 0.0, 1.0)
    yb = (y > 0.35).astype(np.int64)
    X_eval = rng.normal(size=(40, 20)).astype(np.float64)
    T_tr = rng.normal(size=(240, 4, 15, 25)).astype(np.float32)
    T_tr[:, 3] = 1.0
    T_eval = rng.normal(size=(40, 4, 15, 25)).astype(np.float32)
    T_eval[:, 3] = 1.0

    def _fit_once(name: str) -> np.ndarray:
        seed_everything(42, torch_threads=4)
        if name == "Linear":
            r = make_pipeline(StandardScaler(), Ridge(alpha=30.0, solver="cholesky")).fit(X[:180], y[:180]).predict(X_eval)
            c = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, solver="lbfgs", max_iter=1000, random_state=42)).fit(X[:180], yb[:180]).predict_proba(X_eval)[:, 1]
        elif name == "RF":
            r = RandomForestRegressor(n_estimators=25, max_depth=8, max_features="sqrt", max_samples=0.8, n_jobs=1, random_state=42).fit(X[:180], y[:180]).predict(X_eval)
            c = RandomForestClassifier(n_estimators=25, max_depth=8, max_features="sqrt", max_samples=0.8, n_jobs=1, random_state=42).fit(X[:180], yb[:180]).predict_proba(X_eval)[:, 1]
        elif name == "HistGBM":
            r = HistGradientBoostingRegressor(max_iter=30, random_state=42).fit(X[:180], y[:180]).predict(X_eval)
            c = HistGradientBoostingClassifier(max_iter=30, random_state=42).fit(X[:180], yb[:180]).predict_proba(X_eval)[:, 1]
        elif name == "LightGBM":
            r = lgb.LGBMRegressor(n_estimators=30, deterministic=True, force_col_wise=True, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42, n_jobs=1, random_state=42, verbose=-1).fit(X[:180], y[:180], eval_X=X[180:], eval_y=y[180:], callbacks=[lgb.early_stopping(5, verbose=False)]).predict(X_eval)
            c = lgb.LGBMClassifier(n_estimators=30, deterministic=True, force_col_wise=True, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42, n_jobs=1, random_state=42, verbose=-1).fit(X[:180], yb[:180], eval_X=X[180:], eval_y=yb[180:], callbacks=[lgb.early_stopping(5, verbose=False)]).predict_proba(X_eval)[:, 1]
        elif name == "XGBoost":
            r = xgb.XGBRegressor(n_estimators=30, tree_method="hist", n_jobs=1, random_state=42, early_stopping_rounds=5).fit(X[:180], y[:180], eval_set=[(X[180:], y[180:])], verbose=False).predict(X_eval)
            c = xgb.XGBClassifier(n_estimators=30, tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=42, early_stopping_rounds=5).fit(X[:180], yb[:180], eval_set=[(X[180:], yb[180:])], verbose=False).predict_proba(X_eval)[:, 1]
        elif name == "CatBoost":
            r = cb.CatBoostRegressor(iterations=30, depth=4, thread_count=4, random_seed=42, verbose=0).fit(X[:180], y[:180], eval_set=(X[180:], y[180:]), early_stopping_rounds=5, verbose=False).predict(X_eval)
            c = cb.CatBoostClassifier(iterations=30, depth=4, thread_count=4, random_seed=42, verbose=0).fit(X[:180], yb[:180], eval_set=(X[180:], yb[180:]), early_stopping_rounds=5, verbose=False).predict_proba(X_eval)[:, 1]
        elif name == "CNN2D":
            _, preds = fit_predict_cnn_fold(T_tr[:180], y[:180], yb[:180], T_tr[180:], y[180:], yb[180:], [T_eval], seed=100, epochs=3, batch_size=64, device_pref="cpu")
            r, c = preds[0]
        else:
            raise ValueError(name)
        return np.concatenate([r.astype(np.float64), c.astype(np.float64)])

    digest_vecs = []
    for fam in ("Linear", "RF", "HistGBM", "LightGBM", "XGBoost", "CatBoost", "CNN2D"):
        v1 = _fit_once(fam)
        v2 = _fit_once(fam)
        diff = float(np.max(np.abs(v1 - v2)))
        assert diff == 0.0, f"Non-deterministic {fam}: max_abs_diff={diff}"
        digest_vecs.append(v1)

    oof = np.column_stack([rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)])
    y_s = 0.7 * oof[:, 0] + 0.3 * oof[:, 1]
    w = fit_simplex_weights_rmse(oof, y_s)
    assert np.isclose(w.sum(), 1.0, atol=1e-6) and w[0] > w[1]
    digest = hashlib.sha256(np.concatenate(digest_vecs).tobytes()).hexdigest()[:16]
    print(f"modeling_v2_train self-test: PASS (7 families bit-exact diff=0.0, digest={digest})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    parser.add_argument("--submission", type=Path, default=SUBMISSION_DEFAULT)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV_DEFAULT)
    parser.add_argument("--sample-sub", type=Path, default=SAMPLE_SUB_DEFAULT)
    parser.add_argument("--cnn-epochs", type=int, default=24)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    seed_everything(42, torch_threads=4)
    warnings.filterwarnings("ignore", category=UserWarning)
    tr_npz = np.load(args.data_dir / "train.npz")
    va_npz = np.load(args.data_dir / "val.npz")
    ho_npz = np.load(args.data_dir / "holdout.npz")

    X_tr, y_tr, folds_tr = tr_npz["X"], tr_npz["y"].astype(np.float64), tr_npz["fold"]
    X_va, y_va = va_npz["X"], va_npz["y"].astype(np.float64)
    X_ho, y_ho = ho_npz["X"], ho_npz["y"].astype(np.float64)
    y_pos_tr = (y_tr > 0.0).astype(np.int64)

    print("Extracting V1 (96) + V2 (138) features & CNN tensors...", flush=True)
    F_v1_tr = extract_v1_combined96_features(X_tr)
    F_v1_va = extract_v1_combined96_features(X_va)
    F_v1_ho = extract_v1_combined96_features(X_ho)

    F_tr_df = extract_features(X_tr)
    F_va_df = extract_features(X_va)
    F_ho_df = extract_features(X_ho)
    feat_names = list(F_tr_df.columns)
    F_tr, F_va, F_ho = F_tr_df.to_numpy(dtype=np.float64), F_va_df.to_numpy(dtype=np.float64), F_ho_df.to_numpy(dtype=np.float64)

    T_tr, T_va, T_ho = build_cnn_tensors(X_tr), build_cnn_tensors(X_va), build_cnn_tensors(X_ho)

    results = {
        "Constant_Mean": eval_constant_mean(y_tr, y_va, y_ho, folds_tr),
        "Physics_q15": eval_physics_q15(X_tr, X_va, X_ho),
        "V1_RF_MaxShift": train_cv_random_forest("V1_RF_MaxShift", F_v1_tr, y_tr, y_pos_tr, F_v1_va, F_v1_ho, folds_tr, seed=24),
        "RF_v2": train_cv_random_forest("RF_v2", F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr, seed=42),
        "Linear": train_cv_linear(F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr),
        "HistGBM": train_cv_histgbm(F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr),
        "LightGBM": train_cv_lightgbm(F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr, n_jobs=args.threads),
        "XGBoost": train_cv_xgboost(F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr, n_jobs=args.threads),
        "CatBoost": train_cv_catboost(F_tr, y_tr, y_pos_tr, F_va, F_ho, folds_tr, thread_count=4),
        "CNN2D": train_cv_cnn2d(T_tr, y_tr, y_pos_tr, T_va, T_ho, folds_tr, epochs=args.cnn_epochs, batch_size=128, device_pref=args.device),
    }

    metrics_df, weights_payload, sanity = fit_ensemble_and_export(
        results,
        y_tr,
        y_va,
        y_ho,
        feat_names,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        submission_path=args.submission,
        test_csv_path=args.test_csv,
        sample_sub_path=args.sample_sub,
    )
    print("\n=== BENCHMARK METRICS (Val & Unseen-Station Holdout) ===")
    print(metrics_df[metrics_df["split"].isin(["val", "holdout"])].to_string(index=False))
    print("\nEnsemble weights:", json.dumps(weights_payload))
    print("Submission sanity:", json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
