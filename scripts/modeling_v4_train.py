#!/usr/bin/env python3
"""Train leak-free, h(f)-invariant 5-fold Modeling v4 ensemble and audit against v2 & v3."""
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# Import torch before GBDT C extensions to prevent macOS OpenMP initialization conflicts.
import torch
import torch.nn as nn
import catboost as cb
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
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
from modeling_v2_train import predict_from_bundle as predict_from_v2_bundle  # noqa: E402
from modeling_v3_features import compute_robust_physics_share  # noqa: E402
from modeling_v3_train import (  # noqa: E402
    ExcessSpectralPrototypeTransformer,
    predict_from_bundle_v3,
)
from modeling_v4_features import build_cnn_tensors_v4, extract_features_v4  # noqa: E402

DATA_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v4"
V2_BUNDLE_DEFAULT = ROOT / "models/v2/bundle.joblib"
V3_BUNDLE_DEFAULT = ROOT / "models/v3/bundle_v3.joblib"
V2_SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v2_ensemble.csv"
V3_SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v3_ensemble.csv"
MODELS_DIR_DEFAULT = ROOT / "models/v4"
REPORTS_DIR_DEFAULT = ROOT / "reports/modeling_v4"
SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v4_ensemble.csv"
TEST_CSV_DEFAULT = ROOT / "data/zindi/raw/Test.csv"
SAMPLE_SUB_DEFAULT = ROOT / "data/zindi/raw/sample_submission.csv"

TEST_COLS = [f"m{m:02d}_{b}" for m in range(1, 16) for b in BANDS]
SHARE_META_CANDIDATES = [
    "Linear_v4",
    "RF_v4",
    "LightGBM_v4",
    "LightGBM_v4_Hurdle",
    "XGBoost_v4",
    "XGBoost_v4_Hurdle",
    "CatBoost_v4",
    "CatBoost_v4_Hurdle",
    "CNN2D_v4",
]
SCORE_META_CANDIDATES = [
    "Linear_v4",
    "RF_v4",
    "LightGBM_v4",
    "XGBoost_v4",
    "CatBoost_v4",
    "CNN2D_v4",
]


class SEBlock2D(nn.Module):
    """Channel Squeeze-and-Excitation over masked 2D spectrogram maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hid),
            nn.SiLU(),
            nn.Linear(hid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        s = (x * mask).sum(dim=(2, 3)) / denom
        w = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * w


class AcousticSEResBlock(nn.Module):
    """Residual 2D convolutional block with Squeeze-and-Excitation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock2D(out_ch)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)), mask)
        return self.act(out + self.proj(x))


class AcousticCNN2D_v4(nn.Module):
    """Fast sleek 7-channel SE-ResNet CNN (7->24->36->48) with masked temporal attention."""

    def __init__(self, in_channels: int = 7) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
        )
        self.block1 = AcousticSEResBlock(24, 24)
        self.block2 = AcousticSEResBlock(24, 36)
        self.block3 = AcousticSEResBlock(36, 48)
        self.time_att = nn.Sequential(
            nn.Conv1d(96, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.drop = nn.Dropout(0.20)
        self.head = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.score_out = nn.Linear(64, 1)
        self.cond_out = nn.Linear(64, 1)
        self.res_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = x[:, 6:7, :, :]
        min_obs = (mask.sum(dim=3, keepdim=False) > 0.5).float()

        h = self.stem(x)
        h = self.block1(h, mask)
        h = self.block2(h, mask)
        h = self.block3(h, mask)
        h_masked = h * mask

        denom_tf = mask.sum(dim=(2, 3)).clamp(min=1.0)
        g_avg = h_masked.sum(dim=(2, 3)) / denom_tf
        g_max = (h_masked + (mask - 1.0) * 1e4).amax(dim=(2, 3))

        f_avg = h_masked.sum(dim=3) / mask.sum(dim=3).clamp(min=1.0)
        f_max = (h_masked + (mask - 1.0) * 1e4).amax(dim=3)
        seq = torch.cat([f_avg, f_max], dim=1)  # (B, 96, 15)
        att_logits = self.time_att(seq) + (min_obs - 1.0) * 1e4
        att_w = torch.softmax(att_logits, dim=2)
        t_att = (seq * att_w).sum(dim=2)  # (B, 96)

        feat = self.head(self.drop(torch.cat([g_avg, g_max, t_att], dim=1)))  # (B, 192) -> (B, 64)
        score_logit = self.score_out(feat).squeeze(-1)
        cond_share = torch.sigmoid(self.cond_out(feat)).squeeze(-1)
        direct_res = 0.15 * torch.tanh(self.res_out(feat).squeeze(-1))
        share = torch.clamp(torch.sigmoid(score_logit) * cond_share + direct_res, 0.0, 1.0)
        return share, score_logit, cond_share


def seed_everything(seed: int = 42, torch_threads: int = 4) -> None:
    """Lock Python, NumPy, and PyTorch RNG state and deterministic CPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, torch_threads))


def _augment_cnn_batch_v4(xb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Apply label-preserving augmentations: time flip (-ch3), ch0-2 noise, & ch4 calibration jitter."""
    x_aug = xb.clone()
    mask = x_aug[:, 6:7, :, :]
    flip_mask = torch.from_numpy((rng.random(len(x_aug)) < 0.5).astype(bool))
    if flip_mask.any():
        x_aug[flip_mask] = torch.flip(x_aug[flip_mask], dims=[2])
        x_aug[flip_mask, 3, :, :] = -x_aug[flip_mask, 3, :, :]
    noise_02 = torch.from_numpy(rng.normal(0.0, 0.03, size=(len(x_aug), 3, 15, 25)).astype(np.float32))
    x_aug[:, :3, :, :] = (x_aug[:, :3, :, :] + noise_02) * mask
    cal_jitter = torch.from_numpy(rng.normal(0.0, 0.08, size=(len(x_aug), 1, 25)).astype(np.float32))
    x_aug[:, 4, :, :] = (x_aug[:, 4, :, :] + cal_jitter) * mask[:, 0, :, :]
    return x_aug


def fit_predict_cnn_fold_v4(
    T_tr: np.ndarray,
    y_share_tr: np.ndarray,
    y_pos_tr: np.ndarray,
    T_es: np.ndarray,
    y_share_es: np.ndarray,
    y_pos_es: np.ndarray,
    eval_tensors: list[np.ndarray],
    seed: int,
    epochs: int = 18,
    batch_size: int = 256,
    lr: float = 2.0e-3,
) -> tuple[dict, list[tuple[np.ndarray, np.ndarray]]]:
    """Train one fast AcousticCNN2D_v4 fold strictly on internal splits with bit-exact determinism."""
    seed_everything(seed, torch_threads=4)
    device = torch.device("cpu")
    model = AcousticCNN2D_v4(in_channels=7).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2.5e-3)
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
            xb = _augment_cnn_batch_v4(x_tr_t[idx], rng)
            yb_s = y_s_t[idx]
            yb_p = y_p_t[idx]
            opt.zero_grad()
            sh_p, sc_l, cond_s = model(xb)
            mse_all = ((sh_p - yb_s) ** 2).mean()
            bce_l = bce(sc_l, yb_p)
            pos_m = yb_p > 0.5
            mse_cond = ((cond_s[pos_m] - yb_s[pos_m]) ** 2).mean() if pos_m.any() else 0.0
            loss = mse_all + 0.14 * bce_l + 0.25 * mse_cond
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            sh_e, sc_e, cond_e = model(x_es_t)
            pos_em = y_p_es_t > 0.5
            mse_ec = ((cond_e[pos_em] - y_s_es_t[pos_em]) ** 2).mean().item() if pos_em.any() else 0.0
            val_loss = float(((sh_e - y_s_es_t) ** 2).mean().item() + 0.14 * bce(sc_e, y_p_es_t).item() + 0.25 * mse_ec)
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
                sh_o, sc_o, _ = model(xb)
                sh_parts.append(sh_o.detach().cpu().numpy().astype(np.float64))
                sc_parts.append(torch.sigmoid(sc_o).detach().cpu().numpy().astype(np.float64))
            preds.append((np.concatenate(sh_parts), np.concatenate(sc_parts)))
    return best_state, preds


def fit_simplex_weights_rmse(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    m = oof_matrix.shape[1]
    init = np.full(m, 1.0 / m)
    res = minimize(
        lambda w: float(np.mean((np.clip(oof_matrix @ w, 0.0, 1.0) - y_true) ** 2)),
        init,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = np.maximum(res.x, 0.0)
    return w / np.maximum(w.sum(), 1e-12)


def fit_simplex_weights_logloss(oof_probs: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    m = oof_probs.shape[1]
    p_clip = np.clip(oof_probs, 1e-5, 1.0 - 1e-5)
    logits = np.log(p_clip / (1.0 - p_clip))
    init = np.full(m, 1.0 / m)

    def _loss(w: np.ndarray) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(logits @ w, -20.0, 20.0)))
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


def apply_logit_blend(probs_matrix: np.ndarray, weights: np.ndarray, temp: float = 1.0) -> np.ndarray:
    p_clip = np.clip(probs_matrix, 1e-5, 1.0 - 1e-5)
    logits = (np.log(p_clip / (1.0 - p_clip)) @ weights) / max(temp, 1e-3)
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0))), 0.0, 1.0)


def fit_oof_temperature(probs: np.ndarray, y_bin: np.ndarray) -> float:
    p_clip = np.clip(probs, 1e-5, 1.0 - 1e-5)
    logits = np.log(p_clip / (1.0 - p_clip))
    res = minimize(
        lambda t: float(log_loss(y_bin, np.clip(1.0 / (1.0 + np.exp(-np.clip(logits / float(t[0]), -20.0, 20.0))), 1e-7, 1.0 - 1e-7))),
        np.array([1.0]),
        method="L-BFGS-B",
        bounds=[(0.5, 4.0)],
    )
    return float(res.x[0])


def compute_metrics(split: str, model: str, y_share: np.ndarray, pred_share: np.ndarray, pred_score: np.ndarray) -> dict:
    y_bin = (y_share > 0.0).astype(int)
    p_sh = np.clip(pred_share, 0.0, 1.0)
    p_sc = np.clip(pred_score, 0.0, 1.0)
    rmse = float(root_mean_squared_error(y_share, p_sh))
    zero_mask = y_share == 0.0
    zero_rmse = float(np.sqrt(np.mean(p_sh[zero_mask] ** 2))) if zero_mask.any() else 0.0
    auc = 0.5 if len(np.unique(y_bin)) < 2 else float(roc_auc_score(y_bin, p_sc))
    pr_auc = float(y_bin.mean()) if len(np.unique(y_bin)) < 2 else float(average_precision_score(y_bin, p_sc))
    ll = float(log_loss(y_bin, np.clip(p_sc, 1e-7, 1.0 - 1e-7))) if len(np.unique(y_bin)) == 2 else 0.0
    brier = float(brier_score_loss(y_bin, p_sc))
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
        "logloss": ll,
        "brier": brier,
    }


def predict_from_bundle_v4(bundle: dict, X_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure inference helper using saved 5-fold Modeling v4 bundle with zero fitting."""
    F = extract_features_v4(X_db)[bundle["feature_names"]].to_numpy(dtype=np.float64)
    T = build_cnn_tensors_v4(X_db)
    n = len(X_db)
    k_folds = len(bundle["folds"])

    share_cols = bundle["share_meta_names"]
    score_cols = bundle["score_meta_names"]
    share_preds = {m: np.zeros(n, dtype=np.float64) for m in share_cols}
    score_preds = {m: np.zeros(n, dtype=np.float64) for m in score_cols}

    for fold_art in bundle["folds"]:
        share_preds["Linear_v4"] += np.clip(fold_art["linear_reg"].predict(F), 0.0, 1.0) / k_folds
        score_preds["Linear_v4"] += fold_art["linear_clf"].predict_proba(F)[:, 1] / k_folds

        share_preds["RF_v4"] += np.clip(fold_art["rf_reg"].predict(F), 0.0, 1.0) / k_folds
        score_preds["RF_v4"] += fold_art["rf_clf"].predict_proba(F)[:, 1] / k_folds

        for fam, prefix in (("LightGBM_v4", "lgb"), ("XGBoost_v4", "xgb"), ("CatBoost_v4", "cb")):
            p_sc = fold_art[f"{prefix}_clf"].predict_proba(F)[:, 1]
            p_sh = np.clip(fold_art[f"{prefix}_reg"].predict(F), 0.0, 1.0)
            p_cond = np.clip(fold_art[f"{prefix}_cond"].predict(F), 0.0, 1.0)
            score_preds[fam] += p_sc / k_folds
            share_preds[fam] += p_sh / k_folds
            share_preds[f"{fam}_Hurdle"] += np.clip(p_sc * p_cond, 0.0, 1.0) / k_folds

        cnn = AcousticCNN2D_v4(in_channels=7)
        cnn.load_state_dict(fold_art["cnn_state"])
        cnn.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(T)
            sh_l, sc_l = [], []
            for s in range(0, n, 512):
                sh_o, sc_o, _ = cnn(t_in[s : s + 512])
                sh_l.append(sh_o.cpu().numpy())
                sc_l.append(torch.sigmoid(sc_o).cpu().numpy())
            share_preds["CNN2D_v4"] += np.concatenate(sh_l) / k_folds
            score_preds["CNN2D_v4"] += np.concatenate(sc_l) / k_folds

    sh_mat = np.column_stack([share_preds[m] for m in share_cols])
    sc_mat = np.column_stack([score_preds[m] for m in score_cols])
    w_sh = np.array(bundle["weights_share"], dtype=np.float64)
    w_sc = np.array(bundle["weights_score"], dtype=np.float64)
    temp = float(bundle.get("score_temperature", 1.0))
    return np.clip(sh_mat @ w_sh, 0.0, 1.0), apply_logit_blend(sc_mat, w_sc, temp=temp)


def _generate_plots(metrics_df: pd.DataFrame, audit_rows: list[dict], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sel = ["Ensemble_v2", "Ensemble_v3", "Linear_v4", "LightGBM_v4", "XGBoost_v4", "CatBoost_v4", "CNN2D_v4", "Ensemble_v4"]
    sub_vh = metrics_df[(metrics_df["split"] == "val_cal_hard") & (metrics_df["model"].isin(sel))].set_index("model").loc[sel]
    sub_ho = metrics_df[(metrics_df["split"] == "holdout") & (metrics_df["model"].isin(sel))].set_index("model").loc[sel]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    x = np.arange(len(sel))
    w = 0.36
    axes[0].bar(x - w / 2, sub_vh["roc_auc"], width=w, label="Cal-Hard Val (h(f)-warped)", color="#1f77b4")
    axes[0].bar(x + w / 2, sub_ho["roc_auc"], width=w, label="Unseen-Station Holdout", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sel, rotation=28, ha="right", fontsize=8.5)
    axes[0].set_ylim(0.50, 1.0)
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_title("v2 vs v3 vs v4: Calibrated Hard-Ship Val & Holdout AUC")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - w / 2, sub_vh["zindi_score_1_minus_rmse"], width=w, label="Cal-Hard Val (h(f)-warped)", color="#2ca02c")
    axes[1].bar(x + w / 2, sub_ho["zindi_score_1_minus_rmse"], width=w, label="Unseen-Station Holdout", color="#9467bd")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sel, rotation=28, ha="right", fontsize=8.5)
    axes[1].set_ylim(0.70, 0.96)
    axes[1].set_ylabel("Competition Score (1 - RMSE)")
    axes[1].set_title("v2 vs v3 vs v4: 1 - RMSE")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "v4_cal_hard_comparison.png", dpi=180)
    plt.close(fig)

    adf = pd.DataFrame(audit_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    gx = np.arange(len(adf))
    bw = 0.25
    axes[0].bar(gx - bw, adf["v2_score_med"], width=bw, label="v2", color="#d62728", alpha=0.85)
    axes[0].bar(gx, adf["v3_score_med"], width=bw, label="v3", color="#ff7f0e", alpha=0.85)
    axes[0].bar(gx + bw, adf["v4_score_med"], width=bw, label="v4", color="#1f77b4", alpha=0.90)
    axes[0].set_xticks(gx)
    axes[0].set_xticklabels(adf["subgroup"], rotation=22, ha="right", fontsize=8.2)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Median score_added_vessel")
    axes[0].set_title("Test.csv Subgroup Calibration (v2 vs v3 vs v4)")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(gx - bw, adf["v2_share_mean"], width=bw, label="v2", color="#ff9896", alpha=0.90)
    axes[1].bar(gx, adf["v3_share_mean"], width=bw, label="v3", color="#ffbb78", alpha=0.90)
    axes[1].bar(gx + bw, adf["v4_share_mean"], width=bw, label="v4", color="#2ca02c", alpha=0.90)
    axes[1].set_xticks(gx)
    axes[1].set_xticklabels(adf["subgroup"], rotation=22, ha="right", fontsize=8.2)
    axes[1].set_ylim(0.0, 0.55)
    axes[1].set_ylabel("Mean share_added_vessel")
    axes[1].set_title("Test.csv Subgroup Share Calibration (v2 vs v3 vs v4)")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "v4_test_subgroup_audit.png", dpi=180)
    plt.close(fig)


def run_self_test() -> None:
    """Verify bit-exact repeat determinism across all v4 model families + CNN2D_v4."""
    seed_everything(42, torch_threads=4)
    rng = np.random.default_rng(7)
    X_raw = rng.normal(90.0, 5.0, size=(120, 15, 25))
    X_raw[:15, 0, :] = np.nan
    F = extract_features_v4(X_raw).to_numpy(dtype=np.float64)
    T = build_cnn_tensors_v4(X_raw)
    y = rng.uniform(0.0, 0.7, size=120)
    y[:45] = 0.0
    yb = (y > 0.0).astype(np.int64)

    def _fit_once(name: str) -> np.ndarray:
        seed_everything(42, torch_threads=4)
        F_tr, F_ev = F[:90], F[90:]
        if name == "Linear_v4":
            r = make_pipeline(StandardScaler(), Ridge(alpha=80.0, solver="cholesky")).fit(F_tr, y[:90]).predict(F_ev)
            c = make_pipeline(StandardScaler(), LogisticRegression(C=0.20, solver="lbfgs", max_iter=1000, random_state=42)).fit(F_tr, yb[:90]).predict_proba(F_ev)[:, 1]
        elif name == "RF_v4":
            r = RandomForestRegressor(n_estimators=20, max_depth=8, max_features="sqrt", n_jobs=1, random_state=42).fit(F_tr, y[:90]).predict(F_ev)
            c = RandomForestClassifier(n_estimators=20, max_depth=8, max_features="sqrt", n_jobs=1, random_state=42).fit(F_tr, yb[:90]).predict_proba(F_ev)[:, 1]
        elif name == "LightGBM_v4":
            r = lgb.LGBMRegressor(n_estimators=25, deterministic=True, force_col_wise=True, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42, n_jobs=1, random_state=42, verbose=-1).fit(F_tr[:70], y[:70], eval_X=F_tr[70:], eval_y=y[70:90], callbacks=[lgb.early_stopping(5, verbose=False)]).predict(F_ev)
            c = lgb.LGBMClassifier(n_estimators=25, deterministic=True, force_col_wise=True, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42, n_jobs=1, random_state=42, verbose=-1).fit(F_tr[:70], yb[:70], eval_X=F_tr[70:], eval_y=yb[70:90], callbacks=[lgb.early_stopping(5, verbose=False)]).predict_proba(F_ev)[:, 1]
        elif name == "XGBoost_v4":
            r = xgb.XGBRegressor(n_estimators=25, tree_method="hist", n_jobs=1, random_state=42, early_stopping_rounds=5).fit(F_tr[:70], y[:70], eval_set=[(F_tr[70:], y[70:90])], verbose=False).predict(F_ev)
            c = xgb.XGBClassifier(n_estimators=25, tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=42, early_stopping_rounds=5).fit(F_tr[:70], yb[:70], eval_set=[(F_tr[70:], yb[70:90])], verbose=False).predict_proba(F_ev)[:, 1]
        elif name == "CatBoost_v4":
            r = cb.CatBoostRegressor(iterations=25, depth=4, thread_count=4, random_seed=42, verbose=0).fit(F_tr[:70], y[:70], eval_set=(F_tr[70:], y[70:90]), early_stopping_rounds=5, verbose=False).predict(F_ev)
            c = cb.CatBoostClassifier(iterations=25, depth=4, thread_count=4, random_seed=42, verbose=0).fit(F_tr[:70], yb[:70], eval_set=(F_tr[70:], yb[70:90]), early_stopping_rounds=5, verbose=False).predict_proba(F_ev)[:, 1]
        elif name == "CNN2D_v4":
            _, preds = fit_predict_cnn_fold_v4(T[:70], y[:70], yb[:70], T[70:90], y[70:90], yb[70:90], [T[90:]], seed=100, epochs=2, batch_size=32)
            r, c = preds[0]
        else:
            raise ValueError(name)
        return np.concatenate([r.astype(np.float64), c.astype(np.float64)])

    vecs = []
    for fam in ("Linear_v4", "RF_v4", "LightGBM_v4", "XGBoost_v4", "CatBoost_v4", "CNN2D_v4"):
        v1 = _fit_once(fam)
        v2 = _fit_once(fam)
        diff = float(np.max(np.abs(v1 - v2)))
        assert diff == 0.0, f"Non-deterministic {fam}: max_abs_diff={diff}"
        vecs.append(v1)
    dig = hashlib.sha256(np.concatenate(vecs).tobytes()).hexdigest()[:16]
    print(f"modeling_v4_train self-test: PASS (6 families bit-exact diff=0.0, digest={dig})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--v2-bundle", type=Path, default=V2_BUNDLE_DEFAULT)
    parser.add_argument("--v3-bundle", type=Path, default=V3_BUNDLE_DEFAULT)
    parser.add_argument("--v2-submission", type=Path, default=V2_SUBMISSION_DEFAULT)
    parser.add_argument("--v3-submission", type=Path, default=V3_SUBMISSION_DEFAULT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    parser.add_argument("--submission", type=Path, default=SUBMISSION_DEFAULT)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV_DEFAULT)
    parser.add_argument("--sample-sub", type=Path, default=SAMPLE_SUB_DEFAULT)
    parser.add_argument("--cnn-epochs", type=int, default=18)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    t_start = time.perf_counter()
    seed_everything(42, torch_threads=4)
    warnings.filterwarnings("ignore", category=UserWarning)

    if args.export_only:
        bundle_path = args.models_dir / "bundle_v4.joblib"
        bundle = joblib.load(bundle_path)
        metrics_df = pd.read_csv(args.reports_dir / "metrics_summary.csv")
        comp = pd.read_csv(args.reports_dir / "v2_v3_v4_comparison.csv")
        imp_df = pd.read_csv(args.reports_dir / "feature_importance.csv")
        weights_payload = json.loads((args.reports_dir / "ensemble_weights.json").read_text(encoding="utf-8"))
        _export_and_audit(bundle, bundle_path, metrics_df, comp, imp_df, weights_payload, args, t_start)
        return

    tr_npz = np.load(args.data_dir / "train.npz")
    vs_npz = np.load(args.data_dir / "val_standard.npz")
    vh_npz = np.load(args.data_dir / "val_cal_hard.npz")
    ho_npz = np.load(args.data_dir / "holdout.npz")

    X_tr, y_tr, folds_tr = tr_npz["X"], tr_npz["y"].astype(np.float64), tr_npz["fold"]
    X_vs, y_vs = vs_npz["X"], vs_npz["y"].astype(np.float64)
    X_vh, y_vh = vh_npz["X"], vh_npz["y"].astype(np.float64)
    X_ho, y_ho = ho_npz["X"], ho_npz["y"].astype(np.float64)
    y_pos_tr = (y_tr > 0.0).astype(np.int64)

    splits_meta = {
        "train_oof": (X_tr, y_tr),
        "val_standard": (X_vs, y_vs),
        "val_cal_hard": (X_vh, y_vh),
        "holdout": (X_ho, y_ho),
    }

    print("1/5 Extracting Modeling v4 h(f)-invariant features & 7-ch CNN tensors...", flush=True)
    F_tr_df = extract_features_v4(X_tr)
    F_vs_df = extract_features_v4(X_vs)
    F_vh_df = extract_features_v4(X_vh)
    F_ho_df = extract_features_v4(X_ho)
    feat_names = list(F_tr_df.columns)

    F_tr = F_tr_df.to_numpy(dtype=np.float64)
    F_vs = F_vs_df.to_numpy(dtype=np.float64)
    F_vh = F_vh_df.to_numpy(dtype=np.float64)
    F_ho = F_ho_df.to_numpy(dtype=np.float64)

    T_tr = build_cnn_tensors_v4(X_tr)
    T_vs = build_cnn_tensors_v4(X_vs)
    T_vh = build_cnn_tensors_v4(X_vh)
    T_ho = build_cnn_tensors_v4(X_ho)

    all_models = ["Ensemble_v2", "Ensemble_v3", "Constant_Mean", "Physics_Robust_q20"] + SHARE_META_CANDIDATES
    results: dict[str, dict[str, np.ndarray]] = {}
    for m in all_models:
        results[m] = {
            "oof_share": np.zeros(len(y_tr), dtype=np.float64),
            "oof_score": np.zeros(len(y_tr), dtype=np.float64),
            "val_standard_share": np.zeros(len(y_vs), dtype=np.float64),
            "val_standard_score": np.zeros(len(y_vs), dtype=np.float64),
            "val_cal_hard_share": np.zeros(len(y_vh), dtype=np.float64),
            "val_cal_hard_score": np.zeros(len(y_vh), dtype=np.float64),
            "holdout_share": np.zeros(len(y_ho), dtype=np.float64),
            "holdout_score": np.zeros(len(y_ho), dtype=np.float64),
        }

    # --- Step 2: Evaluate saved Ensemble_v2 & Ensemble_v3 (pure inference!) ---
    print("2/5 Evaluating saved Ensemble_v2 & Ensemble_v3 on v4 validation & holdout splits...", flush=True)
    if args.v2_bundle.is_file():
        b2 = joblib.load(args.v2_bundle)
        for sp_key, X_sp in (("val_standard", X_vs), ("val_cal_hard", X_vh), ("holdout", X_ho)):
            sh2, sc2 = predict_from_v2_bundle(b2, X_sp)
            results["Ensemble_v2"][f"{sp_key}_share"] = sh2
            results["Ensemble_v2"][f"{sp_key}_score"] = sc2

    if args.v3_bundle.is_file():
        sys.modules["__main__"].ExcessSpectralPrototypeTransformer = ExcessSpectralPrototypeTransformer
        b3 = joblib.load(args.v3_bundle)
        for sp_key, X_sp in (("val_standard", X_vs), ("val_cal_hard", X_vh), ("holdout", X_ho)):
            sh3, sc3 = predict_from_bundle_v3(b3, X_sp)
            results["Ensemble_v3"][f"{sp_key}_share"] = sh3
            results["Ensemble_v3"][f"{sp_key}_score"] = sc3

    # Constant & Physics baselines
    for k in range(5):
        m = folds_tr == k
        results["Constant_Mean"]["oof_share"][m] = float(y_tr[~m].mean())
        results["Constant_Mean"]["oof_score"][m] = float(y_pos_tr[~m].mean())
    for sp_key in ("val_standard", "val_cal_hard", "holdout"):
        results["Constant_Mean"][f"{sp_key}_share"][:] = float(y_tr.mean())
        results["Constant_Mean"][f"{sp_key}_score"][:] = float(y_pos_tr.mean())

    for sp_key, X_sp, pfx in (
        ("train_oof", X_tr, "oof"),
        ("val_standard", X_vs, "val_standard"),
        ("val_cal_hard", X_vh, "val_cal_hard"),
        ("holdout", X_ho, "holdout"),
    ):
        psh, psc = compute_robust_physics_share(X_sp, q=20.0)
        results["Physics_Robust_q20"][f"{pfx}_share"] = psh
        results["Physics_Robust_q20"][f"{pfx}_score"] = psc

    # --- Step 3: 5-Fold Training of All v4 Families ---
    print("3/5 Training 5-fold Linear, RF, LightGBM, XGBoost, CatBoost (+ Hurdle) & fast CNN2D_v4...", flush=True)
    fold_artifacts = []
    lgb_gains = np.zeros(len(feat_names), dtype=np.float64)
    xgb_gains = np.zeros(len(feat_names), dtype=np.float64)
    cb_imps = np.zeros(len(feat_names), dtype=np.float64)

    for k in range(5):
        t_fk = time.perf_counter()
        es_fold = (k + 1) % 5
        val_mask = folds_tr == k
        es_mask = folds_tr == es_fold
        fit_mask = (~val_mask) & (~es_mask)
        tr_mask = ~val_mask

        lin_reg = make_pipeline(StandardScaler(), Ridge(alpha=80.0, solver="cholesky")).fit(F_tr[tr_mask], y_tr[tr_mask])
        lin_clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.20, solver="lbfgs", max_iter=1200, random_state=42 + k)
        ).fit(F_tr[tr_mask], y_pos_tr[tr_mask])
        for pfx, Fm in (("oof", F_tr[val_mask]), ("val_standard", F_vs), ("val_cal_hard", F_vh), ("holdout", F_ho)):
            sh_p = np.clip(lin_reg.predict(Fm), 0.0, 1.0)
            sc_p = lin_clf.predict_proba(Fm)[:, 1]
            if pfx == "oof":
                results["Linear_v4"]["oof_share"][val_mask] = sh_p
                results["Linear_v4"]["oof_score"][val_mask] = sc_p
            else:
                results["Linear_v4"][f"{pfx}_share"] += sh_p / 5.0
                results["Linear_v4"][f"{pfx}_score"] += sc_p / 5.0

        rf_reg = RandomForestRegressor(
            n_estimators=160, max_depth=14, min_samples_leaf=5, max_features="sqrt", max_samples=0.8, n_jobs=1, random_state=42 + k
        ).fit(F_tr[tr_mask], y_tr[tr_mask])
        rf_clf = RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=5, max_features="sqrt", max_samples=0.8, n_jobs=1, random_state=42 + k
        ).fit(F_tr[tr_mask], y_pos_tr[tr_mask])
        for pfx, Fm in (("oof", F_tr[val_mask]), ("val_standard", F_vs), ("val_cal_hard", F_vh), ("holdout", F_ho)):
            sh_p = np.clip(rf_reg.predict(Fm), 0.0, 1.0)
            sc_p = rf_clf.predict_proba(Fm)[:, 1]
            if pfx == "oof":
                results["RF_v4"]["oof_share"][val_mask] = sh_p
                results["RF_v4"]["oof_score"][val_mask] = sc_p
            else:
                results["RF_v4"][f"{pfx}_share"] += sh_p / 5.0
                results["RF_v4"][f"{pfx}_score"] += sc_p / 5.0

        fit_pos = fit_mask & (y_pos_tr == 1)
        es_pos = es_mask & (y_pos_tr == 1)

        lgb_kw = dict(
            n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=35,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.5, reg_lambda=6.0,
            deterministic=True, force_col_wise=True, feature_fraction_seed=42 + k,
            bagging_seed=42 + k, data_random_seed=42 + k, n_jobs=args.threads, random_state=42 + k, verbose=-1,
        )
        lgb_reg = lgb.LGBMRegressor(**lgb_kw).fit(
            F_tr[fit_mask], y_tr[fit_mask], eval_X=F_tr[es_mask], eval_y=y_tr[es_mask], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_clf = lgb.LGBMClassifier(**lgb_kw).fit(
            F_tr[fit_mask], y_pos_tr[fit_mask], eval_X=F_tr[es_mask], eval_y=y_pos_tr[es_mask], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_cond = lgb.LGBMRegressor(**lgb_kw).fit(
            F_tr[fit_pos], y_tr[fit_pos], eval_X=F_tr[es_pos], eval_y=y_tr[es_pos], callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        imp = lgb_clf.booster_.feature_importance(importance_type="gain")
        lgb_gains += imp / max(float(imp.sum()), 1e-12) / 5.0

        for pfx, Fm in (("oof", F_tr[val_mask]), ("val_standard", F_vs), ("val_cal_hard", F_vh), ("holdout", F_ho)):
            sc_p = lgb_clf.predict_proba(Fm)[:, 1]
            sh_p = np.clip(lgb_reg.predict(Fm), 0.0, 1.0)
            hu_p = np.clip(sc_p * np.clip(lgb_cond.predict(Fm), 0.0, 1.0), 0.0, 1.0)
            if pfx == "oof":
                results["LightGBM_v4"]["oof_score"][val_mask] = sc_p
                results["LightGBM_v4"]["oof_share"][val_mask] = sh_p
                results["LightGBM_v4_Hurdle"]["oof_score"][val_mask] = sc_p
                results["LightGBM_v4_Hurdle"]["oof_share"][val_mask] = hu_p
            else:
                results["LightGBM_v4"][f"{pfx}_score"] += sc_p / 5.0
                results["LightGBM_v4"][f"{pfx}_share"] += sh_p / 5.0
                results["LightGBM_v4_Hurdle"][f"{pfx}_score"] += sc_p / 5.0
                results["LightGBM_v4_Hurdle"][f"{pfx}_share"] += hu_p / 5.0

        xgb_kw = dict(
            n_estimators=1200, learning_rate=0.03, max_depth=5, min_child_weight=15,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.5, reg_lambda=6.0,
            tree_method="hist", early_stopping_rounds=50, n_jobs=args.threads, random_state=42 + k,
        )
        xgb_reg = xgb.XGBRegressor(**xgb_kw).fit(F_tr[fit_mask], y_tr[fit_mask], eval_set=[(F_tr[es_mask], y_tr[es_mask])], verbose=False)
        xgb_clf = xgb.XGBClassifier(eval_metric="logloss", **xgb_kw).fit(F_tr[fit_mask], y_pos_tr[fit_mask], eval_set=[(F_tr[es_mask], y_pos_tr[es_mask])], verbose=False)
        xgb_cond = xgb.XGBRegressor(**xgb_kw).fit(F_tr[fit_pos], y_tr[fit_pos], eval_set=[(F_tr[es_pos], y_tr[es_pos])], verbose=False)
        ximp = xgb_clf.feature_importances_
        xgb_gains += ximp / max(float(ximp.sum()), 1e-12) / 5.0

        for pfx, Fm in (("oof", F_tr[val_mask]), ("val_standard", F_vs), ("val_cal_hard", F_vh), ("holdout", F_ho)):
            sc_p = xgb_clf.predict_proba(Fm)[:, 1]
            sh_p = np.clip(xgb_reg.predict(Fm), 0.0, 1.0)
            hu_p = np.clip(sc_p * np.clip(xgb_cond.predict(Fm), 0.0, 1.0), 0.0, 1.0)
            if pfx == "oof":
                results["XGBoost_v4"]["oof_score"][val_mask] = sc_p
                results["XGBoost_v4"]["oof_share"][val_mask] = sh_p
                results["XGBoost_v4_Hurdle"]["oof_score"][val_mask] = sc_p
                results["XGBoost_v4_Hurdle"]["oof_share"][val_mask] = hu_p
            else:
                results["XGBoost_v4"][f"{pfx}_score"] += sc_p / 5.0
                results["XGBoost_v4"][f"{pfx}_share"] += sh_p / 5.0
                results["XGBoost_v4_Hurdle"][f"{pfx}_score"] += sc_p / 5.0
                results["XGBoost_v4_Hurdle"][f"{pfx}_share"] += hu_p / 5.0

        cb_kw = dict(iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=8.0, thread_count=4, random_seed=42 + k, verbose=0)
        cb_reg = cb.CatBoostRegressor(**cb_kw).fit(F_tr[fit_mask], y_tr[fit_mask], eval_set=(F_tr[es_mask], y_tr[es_mask]), early_stopping_rounds=50, verbose=False)
        cb_clf = cb.CatBoostClassifier(**cb_kw).fit(F_tr[fit_mask], y_pos_tr[fit_mask], eval_set=(F_tr[es_mask], y_pos_tr[es_mask]), early_stopping_rounds=50, verbose=False)
        cb_cond = cb.CatBoostRegressor(**cb_kw).fit(F_tr[fit_pos], y_tr[fit_pos], eval_set=(F_tr[es_pos], y_tr[es_pos]), early_stopping_rounds=50, verbose=False)
        cimp = np.asarray(cb_clf.get_feature_importance(), dtype=np.float64)
        cb_imps += cimp / max(float(cimp.sum()), 1e-12) / 5.0

        for pfx, Fm in (("oof", F_tr[val_mask]), ("val_standard", F_vs), ("val_cal_hard", F_vh), ("holdout", F_ho)):
            sc_p = cb_clf.predict_proba(Fm)[:, 1]
            sh_p = np.clip(cb_reg.predict(Fm), 0.0, 1.0)
            hu_p = np.clip(sc_p * np.clip(cb_cond.predict(Fm), 0.0, 1.0), 0.0, 1.0)
            if pfx == "oof":
                results["CatBoost_v4"]["oof_score"][val_mask] = sc_p
                results["CatBoost_v4"]["oof_share"][val_mask] = sh_p
                results["CatBoost_v4_Hurdle"]["oof_score"][val_mask] = sc_p
                results["CatBoost_v4_Hurdle"]["oof_share"][val_mask] = hu_p
            else:
                results["CatBoost_v4"][f"{pfx}_score"] += sc_p / 5.0
                results["CatBoost_v4"][f"{pfx}_share"] += sh_p / 5.0
                results["CatBoost_v4_Hurdle"][f"{pfx}_score"] += sc_p / 5.0
                results["CatBoost_v4_Hurdle"][f"{pfx}_share"] += hu_p / 5.0

        cnn_state, cnn_preds = fit_predict_cnn_fold_v4(
            T_tr[fit_mask], y_tr[fit_mask], y_pos_tr[fit_mask],
            T_tr[es_mask], y_tr[es_mask], y_pos_tr[es_mask],
            [T_tr[val_mask], T_vs, T_vh, T_ho],
            seed=100 + k, epochs=args.cnn_epochs, batch_size=256, lr=2.0e-3,
        )
        results["CNN2D_v4"]["oof_share"][val_mask], results["CNN2D_v4"]["oof_score"][val_mask] = cnn_preds[0]
        for idx_sp, sp_key in enumerate(("val_standard", "val_cal_hard", "holdout"), start=1):
            results["CNN2D_v4"][f"{sp_key}_share"] += cnn_preds[idx_sp][0] / 5.0
            results["CNN2D_v4"][f"{sp_key}_score"] += cnn_preds[idx_sp][1] / 5.0

        fold_artifacts.append(
            {
                "fold": k,
                "linear_reg": lin_reg,
                "linear_clf": lin_clf,
                "rf_reg": rf_reg,
                "rf_clf": rf_clf,
                "lgb_reg": lgb_reg,
                "lgb_clf": lgb_clf,
                "lgb_cond": lgb_cond,
                "xgb_reg": xgb_reg,
                "xgb_clf": xgb_clf,
                "xgb_cond": xgb_cond,
                "cb_reg": cb_reg,
                "cb_clf": cb_clf,
                "cb_cond": cb_cond,
                "cnn_state": cnn_state,
            }
        )
        print(f"  Fold {k}/4 finished in {time.perf_counter() - t_fk:.1f}s", flush=True)

    # --- Step 4: Leak-Free OOF Ensembling & Temperature Calibration ---
    print("4/5 Fitting OOF simplex weights, exporting bundle_v4 & verifying...", flush=True)
    oof_sh_mat = np.column_stack([results[m]["oof_share"] for m in SHARE_META_CANDIDATES])
    oof_sc_mat = np.column_stack([results[m]["oof_score"] for m in SCORE_META_CANDIDATES])
    w_share = fit_simplex_weights_rmse(oof_sh_mat, y_tr)
    w_score = fit_simplex_weights_logloss(oof_sc_mat, y_pos_tr)

    oof_sc_raw = apply_logit_blend(oof_sc_mat, w_score, temp=1.0)
    score_temp = fit_oof_temperature(oof_sc_raw, y_pos_tr)

    results["Ensemble_v4"] = {
        "oof_share": np.clip(oof_sh_mat @ w_share, 0.0, 1.0),
        "oof_score": apply_logit_blend(oof_sc_mat, w_score, temp=score_temp),
    }
    for sp_key in ("val_standard", "val_cal_hard", "holdout"):
        sp_sh_mat = np.column_stack([results[m][f"{sp_key}_share"] for m in SHARE_META_CANDIDATES])
        sp_sc_mat = np.column_stack([results[m][f"{sp_key}_score"] for m in SCORE_META_CANDIDATES])
        results["Ensemble_v4"][f"{sp_key}_share"] = np.clip(sp_sh_mat @ w_share, 0.0, 1.0)
        results["Ensemble_v4"][f"{sp_key}_score"] = apply_logit_blend(sp_sc_mat, w_score, temp=score_temp)

    metric_rows = []
    for m in all_models + ["Ensemble_v4"]:
        for sp_key, (_, y_sp) in splits_meta.items():
            if m in ("Ensemble_v2", "Ensemble_v3") and sp_key == "train_oof":
                continue
            pfx = "oof" if sp_key == "train_oof" else sp_key
            metric_rows.append(compute_metrics(sp_key, m, y_sp, results[m][f"{pfx}_share"], results[m][f"{pfx}_score"]))

    metrics_df = pd.DataFrame(metric_rows)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.reports_dir / "metrics_summary.csv", index=False)

    comp = metrics_df[
        metrics_df["model"].isin(["Ensemble_v2", "Ensemble_v3", "Physics_Robust_q20", "Linear_v4", "RF_v4", "LightGBM_v4", "LightGBM_v4_Hurdle", "XGBoost_v4", "CatBoost_v4", "CNN2D_v4", "Ensemble_v4"])
        & metrics_df["split"].isin(["val_standard", "val_cal_hard", "holdout"])
    ].sort_values(["split", "roc_auc"], ascending=[True, False])
    comp.to_csv(args.reports_dir / "v2_v3_v4_comparison.csv", index=False)

    imp_df = pd.DataFrame(
        {
            "feature": feat_names,
            "lgb_clf_gain": lgb_gains,
            "xgb_clf_gain": xgb_gains,
            "cb_clf_imp": cb_imps,
            "mean_norm_importance": (lgb_gains + xgb_gains + cb_imps) / 3.0,
        }
    ).sort_values("mean_norm_importance", ascending=False)
    imp_df.to_csv(args.reports_dir / "feature_importance.csv", index=False)

    weights_payload = {
        "share_models": SHARE_META_CANDIDATES,
        "share_weights": {m: float(w) for m, w in zip(SHARE_META_CANDIDATES, w_share)},
        "score_models": SCORE_META_CANDIDATES,
        "score_weights": {m: float(w) for m, w in zip(SCORE_META_CANDIDATES, w_score)},
        "score_temperature": float(score_temp),
    }
    (args.reports_dir / "ensemble_weights.json").write_text(json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8")

    bundle = {
        "feature_names": feat_names,
        "share_meta_names": SHARE_META_CANDIDATES,
        "score_meta_names": SCORE_META_CANDIDATES,
        "weights_share": w_share.tolist(),
        "weights_score": w_score.tolist(),
        "score_temperature": float(score_temp),
        "folds": fold_artifacts,
    }
    bundle_path = args.models_dir / "bundle_v4.joblib"
    joblib.dump(bundle, bundle_path)
    _export_and_audit(bundle, bundle_path, metrics_df, comp, imp_df, weights_payload, args, t_start)


def _export_and_audit(
    bundle: dict,
    bundle_path: Path,
    metrics_df: pd.DataFrame,
    comp: pd.DataFrame,
    imp_df: pd.DataFrame,
    weights_payload: dict,
    args: argparse.Namespace,
    t_start: float,
) -> None:
    """Run Test.csv float64 inference, shift invariance, subgroup audit, and save submission/plots."""
    print("5/5 Generating candidate_v4_ensemble.csv & running invariance/audit checks...", flush=True)
    test_df = pd.read_csv(args.test_csv)
    sample_df = pd.read_csv(args.sample_sub)
    id_col = sample_df.columns[0]
    assert list(test_df[id_col]) == list(sample_df[id_col]), "Test ID alignment mismatch"
    X_test = test_df[TEST_COLS].to_numpy(dtype=np.float64).reshape(len(test_df), 15, 25)

    reloaded = joblib.load(bundle_path)
    mem_sh, mem_sc = predict_from_bundle_v4(bundle, X_test)
    test_share, test_score = predict_from_bundle_v4(reloaded, X_test)
    assert max(float(np.max(np.abs(mem_sh - test_share))), float(np.max(np.abs(mem_sc - test_score)))) < 1e-12

    sh50_a, sc50_a = predict_from_bundle_v4(reloaded, X_test[:50])
    sh50_b, sc50_b = predict_from_bundle_v4(reloaded, X_test[:50] + 37.5)
    shift_diff_sh = float(np.max(np.abs(sh50_a - sh50_b)))
    shift_diff_sc = float(np.max(np.abs(sc50_a - sc50_b)))
    shift_p95_sh = float(np.percentile(np.abs(sh50_a - sh50_b), 95))
    shift_p95_sc = float(np.percentile(np.abs(sc50_a - sc50_b), 95))
    assert shift_p95_sh < 1e-6 and shift_p95_sc < 1e-6
    assert shift_diff_sh < 5e-3 and shift_diff_sc < 5e-3, f"Shift invariance failed: {shift_diff_sh}, {shift_diff_sc}"

    X_ho50 = np.load(args.data_dir / "holdout.npz")["X"][:50].astype(np.float64)
    ho_sh_a, ho_sc_a = predict_from_bundle_v4(reloaded, X_ho50)
    ho_sh_b, ho_sc_b = predict_from_bundle_v4(reloaded, X_ho50 + 37.5)
    ho_shift_diff_sh = float(np.max(np.abs(ho_sh_a - ho_sh_b)))
    ho_shift_diff_sc = float(np.max(np.abs(ho_sc_a - ho_sc_b)))
    assert ho_shift_diff_sh < 1e-6 and ho_shift_diff_sc < 1e-6, f"Holdout continuous shift failed: {ho_shift_diff_sh}, {ho_shift_diff_sc}"

    sub_df = pd.DataFrame(
        {
            id_col: sample_df[id_col],
            "share_added_vessel": np.clip(test_share, 0.0, 1.0),
            "score_added_vessel": np.clip(test_score, 0.0, 1.0),
        }
    )
    assert len(sub_df) == len(sample_df) and list(sub_df.columns) == list(sample_df.columns)
    assert np.isfinite(sub_df[["share_added_vessel", "score_added_vessel"]].to_numpy()).all()
    assert sub_df["share_added_vessel"].between(0.0, 1.0).all()
    assert sub_df["score_added_vessel"].between(0.0, 1.0).all()
    sub_df.to_csv(args.submission, index=False)

    v2_sub = pd.read_csv(args.v2_submission) if args.v2_submission.is_file() else sub_df.copy()
    v3_sub = pd.read_csv(args.v3_submission) if args.v3_submission.is_file() else sub_df.copy()
    Xt_f = X_test
    obs = np.isfinite(Xt_f)
    min_obs = obs.all(axis=2)
    Xc = Xt_f - np.nanmedian(Xt_f, axis=(1, 2), keepdims=True)
    E = np.where(obs, np.power(10.0, np.clip(Xc, -60.0, 60.0) / 10.0), np.nan)
    mdb = 10.0 * np.log10(np.maximum(np.where(min_obs, np.nansum(E, axis=2), np.nan), 1e-30))
    mdb[~min_obs] = np.nan
    amx = np.nanargmax(mdb, axis=1)
    amn = np.nanargmin(mdb, axis=1)
    bb_rng = np.nanmax(mdb, axis=1) - np.nanmin(mdb, axis=1)
    q_d63 = np.array([Xc[i, amn[i], 23] - Xc[i, amn[i], 21] for i in range(len(Xt_f))])
    dhv = np.array(
        [
            (Xc[i, amx[i], 17:25].mean() - Xc[i, amx[i], 0:5].mean())
            - (Xc[i, amn[i], 17:25].mean() - Xc[i, amn[i], 0:5].mean())
            for i in range(len(Xt_f))
        ]
    )
    drop_db = np.array([np.nanmedian(mdb[i]) - np.nanmin(mdb[i]) for i in range(len(Xt_f))])

    subgroups = [
        ("m01_PRES & dhv>+10 (VTUAD)", min_obs[:, 0] & (dhv > 10.0)),
        ("m01_PRES & dhv<+2 (SA_VLF)", min_obs[:, 0] & (dhv < 2.0)),
        ("m01_MISS & d63>1.5 (Reef)", (~min_obs[:, 0]) & (q_d63 > 1.5)),
        ("glitch_drop (>15 dB)", drop_db > 15.0),
        ("low_rng (bb_rng<8 dB)", bb_rng < 8.0),
        ("ALL_1500", np.ones(len(Xt_f), dtype=bool)),
    ]
    audit_rows = []
    for name, mask in subgroups:
        audit_rows.append(
            {
                "subgroup": name,
                "n": int(mask.sum()),
                "v2_share_mean": float(v2_sub.loc[mask, "share_added_vessel"].mean()),
                "v2_score_med": float(v2_sub.loc[mask, "score_added_vessel"].median()),
                "v3_share_mean": float(v3_sub.loc[mask, "share_added_vessel"].mean()),
                "v3_score_med": float(v3_sub.loc[mask, "score_added_vessel"].median()),
                "v4_share_mean": float(sub_df.loc[mask, "share_added_vessel"].mean()),
                "v4_share_med": float(sub_df.loc[mask, "share_added_vessel"].median()),
                "v4_score_med": float(sub_df.loc[mask, "score_added_vessel"].median()),
                "v4_score_gt_095": float((sub_df.loc[mask, "score_added_vessel"] > 0.95).mean()),
            }
        )
    (args.reports_dir / "test_prediction_audit.json").write_text(json.dumps(audit_rows, indent=2) + "\n", encoding="utf-8")
    _generate_plots(metrics_df, audit_rows, args.reports_dir / "figures")

    sanity = {
        "elapsed_s": float(time.perf_counter() - t_start),
        "shift_invariance_test_max_diff_share": shift_diff_sh,
        "shift_invariance_test_max_diff_score": shift_diff_sc,
        "shift_invariance_test_p95_diff_share": shift_p95_sh,
        "shift_invariance_test_p95_diff_score": shift_p95_sc,
        "shift_invariance_holdout_max_diff_share": ho_shift_diff_sh,
        "shift_invariance_holdout_max_diff_score": ho_shift_diff_sc,
        "submission_path": str(args.submission.relative_to(ROOT)),
        "test_share_mean": float(sub_df["share_added_vessel"].mean()),
        "test_share_std": float(sub_df["share_added_vessel"].std()),
        "test_share_q10_q50_q90": [float(x) for x in np.quantile(sub_df["share_added_vessel"], [0.10, 0.50, 0.90])],
        "test_score_mean": float(sub_df["score_added_vessel"].mean()),
        "test_score_std": float(sub_df["score_added_vessel"].std()),
        "test_score_q10_q50_q90": [float(x) for x in np.quantile(sub_df["score_added_vessel"], [0.10, 0.50, 0.90])],
        "test_score_gt_095_frac": float((sub_df["score_added_vessel"] > 0.95).mean()),
        "corr_v4_v2_share": float(np.corrcoef(sub_df["share_added_vessel"], v2_sub["share_added_vessel"])[0, 1]),
        "corr_v4_v2_score": float(np.corrcoef(sub_df["score_added_vessel"], v2_sub["score_added_vessel"])[0, 1]),
    }
    (args.reports_dir / "sanity_summary.json").write_text(json.dumps(sanity, indent=2) + "\n", encoding="utf-8")

    print("\n=== V2 VS V3 VS V4 BENCHMARK SUMMARY ===")
    print(comp[["split", "model", "zindi_score_1_minus_rmse", "rmse", "roc_auc", "pr_auc", "zero_rmse"]].to_string(index=False))
    print("\nTop 15 Classifier Feature Importances:")
    print(imp_df.head(15)[["feature", "lgb_clf_gain", "xgb_clf_gain", "cb_clf_imp", "mean_norm_importance"]].to_string(index=False))
    print("\nEnsemble weights:", json.dumps(weights_payload))
    print("\nRead-only Test.csv Subgroup Audit (v2 vs v3 vs v4):")
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print("\nSanity Summary:", json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
