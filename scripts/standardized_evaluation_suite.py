#!/usr/bin/env python3
"""Standardized international acoustic & zero-inflated ML evaluation suite (PCEN, RPCA, JOMOPANS, CCC, ECE)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# Import torch before GBDTs on macOS
import torch  # noqa: F401
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from extract_external_decidecade import EXACT_CENTRES  # noqa: E402
from modeling_v2_train import predict_from_bundle as predict_v2  # noqa: E402
from modeling_v3_features import _prepare_clean_masks, compute_robust_physics_share  # noqa: E402
from modeling_v3_train import (  # noqa: E402
    ExcessSpectralPrototypeTransformer,
    predict_from_bundle_v3 as predict_v3,
)
from modeling_v4_train import predict_from_bundle_v4 as predict_v4  # noqa: E402

DATA_DIR_DEFAULT = ROOT / "data/synthetic/modeling_v4"
OUT_DIR_DEFAULT = ROOT / "reports/standards_benchmark"


def compute_pcen_map(
    X_db: np.ndarray,
    s: float = 0.35,
    alpha: float = 0.90,
    delta: float = 2.0,
    r: float = 0.50,
    eps: float = 1e-6,
) -> np.ndarray:
    """Shift-invariant Per-Channel Energy Normalization over a single (15, 25) dB matrix."""
    X = np.asarray(X_db, dtype=np.float64)
    obs = np.isfinite(X)
    win_med = float(np.nanmedian(X))
    Xc = X - win_med
    E = np.where(obs, np.power(10.0, np.clip(Xc, -60.0, 60.0) / 10.0), np.nan)
    e_med = max(float(np.nanmedian(E)), 1e-12)
    E_norm = np.where(obs, E / e_med, 0.0)

    M = np.zeros_like(E_norm)
    valid_t = np.flatnonzero(obs.all(axis=1))
    if len(valid_t) == 0:
        return np.zeros_like(X)
    m_prev = E_norm[valid_t[0]].copy()
    for t in range(15):
        if obs[t].all():
            m_prev = (1.0 - s) * m_prev + s * E_norm[t]
        M[t] = m_prev

    denom = np.power(eps + M, alpha)
    pcen = np.power(E_norm / denom + delta, r) - (delta ** r)
    pcen[~obs] = np.nan
    return pcen


def rpca_ialm_decompose(
    M: np.ndarray,
    lam: float | None = None,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Inexact ALM Robust PCA decomposing M = L + S (Candès et al., 2011)."""
    X = np.asarray(M, dtype=np.float64)
    m_rows, n_cols = X.shape
    if lam is None:
        lam = 1.0 / float(np.sqrt(max(m_rows, n_cols)))
    norm_two = float(np.linalg.norm(X, ord=2))
    norm_inf = float(np.linalg.norm(X, ord=np.inf)) / max(lam, 1e-12)
    dual_norm = max(norm_two, norm_inf, 1e-12)
    Y = X / dual_norm
    norm_f = max(float(np.linalg.norm(X, ord="fro")), 1e-12)
    mu = 1.25 / max(norm_two, 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    L = np.zeros_like(X)
    S = np.zeros_like(X)
    for _ in range(max_iter):
        # Soft-threshold update for sparse transient matrix S
        T_s = X - L + Y / mu
        S = np.sign(T_s) * np.maximum(np.abs(T_s) - lam / mu, 0.0)
        # Singular-value thresholding for low-rank stationary background L
        U, sv, Vt = np.linalg.svd(X - S + Y / mu, full_matrices=False)
        sv_thr = np.maximum(sv - 1.0 / mu, 0.0)
        svp = int((sv_thr > 0).sum())
        L = (U[:, :svp] * sv_thr[:svp]) @ Vt[:svp, :] if svp > 0 else np.zeros_like(X)

        Z = X - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)
        if float(np.linalg.norm(Z, ord="fro")) / norm_f < tol:
            break
    return L, S


def eval_rpca_unsupervised(X_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Run shift-invariant RPCA decomposition across (N, 15, 25) dB matrices."""
    Xc, _, min_obs, clean_min, E_lin, _ = _prepare_clean_masks(X_db)
    n = len(Xc)
    shares = np.zeros(n, dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)
    sparse_maps: list[np.ndarray] = []

    for i in range(n):
        c_idx = np.flatnonzero(clean_min[i])
        E_c = E_lin[i, c_idx]
        med_e = max(float(np.median(E_c)), 1e-12)
        E_rel = E_c / med_e
        M = np.log1p(E_rel)
        L, S = rpca_ialm_decompose(M)

        # Positive sparse transient excess energy fraction
        E_bg = np.expm1(np.clip(L, 0.0, 20.0))
        E_sp = np.maximum(E_rel - E_bg, 0.0)
        sh = float(np.clip(E_sp.sum() / max(float(E_rel.sum()), 1e-12), 0.0, 1.0))
        sp_ratio = float(np.sum(np.maximum(S, 0.0) ** 2) / max(float(np.sum(M ** 2)), 1e-12))
        sc = float(1.0 / (1.0 + np.exp(-10.0 * (sp_ratio - 0.08))))
        shares[i] = sh
        scores[i] = np.clip(sc, 0.0, 1.0)

        full_s = np.full((15, 25), np.nan, dtype=np.float64)
        full_s[c_idx] = S
        sparse_maps.append(full_s)
    return shares, scores, sparse_maps


def compute_brier_decomposition_and_ece(y_bin: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, float, float, float]:
    """Compute 10-bin Reliability, Resolution, Uncertainty, and ECE."""
    p_c = np.clip(p, 0.0, 1.0)
    y_b = y_bin.astype(np.float64)
    base_rate = float(y_b.mean())
    unc = base_rate * (1.0 - base_rate)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rel = 0.0
    res = 0.0
    ece = 0.0
    n = max(len(y_b), 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p_c >= lo) & (p_c <= hi if b == n_bins - 1 else p_c < hi)
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_mean = float(p_c[mask].mean())
        o_mean = float(y_b[mask].mean())
        w = nk / n
        rel += w * ((p_mean - o_mean) ** 2)
        res += w * ((o_mean - base_rate) ** 2)
        ece += w * abs(p_mean - o_mean)
    return float(rel), float(res), float(unc), float(ece)


def compute_standardized_metrics(
    split: str,
    model: str,
    y_share: np.ndarray,
    pred_share: np.ndarray,
    pred_score: np.ndarray,
) -> dict:
    """Compute 16+ standardized regression, acoustic excess, rank, and calibration metrics."""
    y = np.asarray(y_share, dtype=np.float64)
    yp = np.clip(np.asarray(pred_share, dtype=np.float64), 0.0, 1.0)
    yb = (y > 0.0).astype(int)
    ps = np.clip(np.asarray(pred_score, dtype=np.float64), 0.0, 1.0)

    rmse = float(root_mean_squared_error(y, yp))
    mae = float(mean_absolute_error(y, yp))
    medae = float(median_absolute_error(y, yp))

    # Lin's Concordance Correlation Coefficient (CCC)
    mu_y, mu_p = float(y.mean()), float(yp.mean())
    var_y, var_p = float(y.var()), float(yp.var())
    cov = float(np.mean((y - mu_y) * (yp - mu_p)))
    ccc_den = var_y + var_p + (mu_y - mu_p) ** 2
    lin_ccc = float((2.0 * cov) / ccc_den) if ccc_den > 1e-12 else 0.0

    nse_r2 = float(r2_score(y, yp)) if len(np.unique(y)) > 1 else 0.0
    sp_r = float(spearmanr(y, yp).statistic) if (np.std(y) > 1e-12 and np.std(yp) > 1e-12) else 0.0
    kt_r = float(kendalltau(y, yp).statistic) if (np.std(y) > 1e-12 and np.std(yp) > 1e-12) else 0.0
    if not np.isfinite(sp_r):
        sp_r = 0.0
    if not np.isfinite(kt_r):
        kt_r = 0.0

    def _reg_rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y[mask] - yp[mask]) ** 2))) if mask.any() else np.nan

    rmse_zero = _reg_rmse(y == 0.0)
    rmse_low = _reg_rmse((y > 0.0) & (y <= 0.10))
    rmse_mod = _reg_rmse((y > 0.10) & (y <= 0.40))
    rmse_high = _reg_rmse(y > 0.40)

    # JOMOPANS / MSFD D11C2 Excess Level L_ex = -10*log10(1 - share) MAE in dB
    lex_true = -10.0 * np.log10(1.0 - np.clip(y, 0.0, 0.99))
    lex_pred = -10.0 * np.log10(1.0 - np.clip(yp, 0.0, 0.99))
    jomopans_lex_mae_db = float(np.mean(np.abs(lex_true - lex_pred)))

    has_two = len(np.unique(yb)) == 2
    auc = float(roc_auc_score(yb, ps)) if has_two else 0.5
    pr_auc = float(average_precision_score(yb, ps)) if has_two else float(yb.mean())
    ll = float(log_loss(yb, np.clip(ps, 1e-7, 1.0 - 1e-7))) if has_two else 0.0
    brier = float(brier_score_loss(yb, ps))
    b_rel, b_res, b_unc, ece10 = compute_brier_decomposition_and_ece(yb, ps, n_bins=10)

    return {
        "split": split,
        "model": model,
        "n": int(len(y)),
        "zindi_score_1_minus_rmse": float(1.0 - rmse),
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "lin_ccc": lin_ccc,
        "nse_r2": nse_r2,
        "spearman_rho": sp_r,
        "kendall_tau": kt_r,
        "rmse_zero": rmse_zero,
        "rmse_low": rmse_low,
        "rmse_mod": rmse_mod,
        "rmse_high": rmse_high,
        "jomopans_lex_mae_db": jomopans_lex_mae_db,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "logloss": ll,
        "brier": brier,
        "brier_rel": b_rel,
        "brier_res": b_res,
        "brier_unc": b_unc,
        "ece_10": ece10,
    }


def _plot_fig1_calibration_and_agreement(
    metrics_df: pd.DataFrame,
    preds_vh: dict[str, tuple[np.ndarray, np.ndarray]],
    y_vh: np.ndarray,
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    yb = (y_vh > 0.0).astype(int)

    # Panel A: 10-bin Reliability Diagram on val_cal_hard
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, label="Perfect calibration")
    edges = np.linspace(0.0, 1.0, 11)
    for m, col, mk in (
        ("Ensemble_v2", "#d62728", "s"),
        ("Ensemble_v3", "#ff7f0e", "^"),
        ("Ensemble_v4", "#1f77b4", "o"),
    ):
        ps = np.clip(preds_vh[m][1], 0.0, 1.0)
        bx, by = [], []
        for b in range(10):
            mask = (ps >= edges[b]) & (ps <= edges[b + 1] if b == 9 else ps < edges[b + 1])
            if mask.sum() >= 5:
                bx.append(float(ps[mask].mean()))
                by.append(float(yb[mask].mean()))
        row = metrics_df[(metrics_df["split"] == "val_cal_hard") & (metrics_df["model"] == m)].iloc[0]
        ax.plot(bx, by, f"-{mk}", color=col, lw=1.8, ms=5, label=f"{m} (ECE={row['ece_10']:.3f}, BS={row['brier']:.3f})")
    ax.set_xlabel("Predicted score_added_vessel")
    ax.set_ylabel("Empirical vessel addition fraction")
    ax.set_title("(a) 10-Bin Reliability Diagram (Cal-Hard Val)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax.grid(alpha=0.25)

    # Panel B: Predicted vs True scatter for Ensemble_v4 on val_cal_hard
    ax = axes[1]
    yp4 = preds_vh["Ensemble_v4"][0]
    row4 = metrics_df[(metrics_df["split"] == "val_cal_hard") & (metrics_df["model"] == "Ensemble_v4")].iloc[0]
    ax.scatter(y_vh, yp4, s=9, alpha=0.35, color="#1f77b4", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1.1)
    ax.set_xlabel("Observed share_added_vessel")
    ax.set_ylabel("Ensemble_v4 Predicted Share")
    ax.set_title("(b) Agreement on Cal-Hard Val (Ensemble_v4)")
    ax.set_xlim(-0.02, 0.96)
    ax.set_ylim(-0.02, 0.96)
    txt = f"Lin CCC = {row4['lin_ccc']:.3f}\nNSE (R²) = {row4['nse_r2']:.3f}\nSpearman ρ = {row4['spearman_rho']:.3f}\n1 - RMSE = {row4['zindi_score_1_minus_rmse']:.4f}"
    ax.text(0.04, 0.72, txt, transform=ax.transAxes, fontsize=8.3, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.grid(alpha=0.25)

    # Panel C: Regime-stratified RMSE
    ax = axes[2]
    regimes = ["rmse_zero", "rmse_low", "rmse_mod", "rmse_high"]
    r_labels = ["Zero\n(y=0)", "Low\n(0, 0.10]", "Moderate\n(0.10, 0.40]", "High\n(>0.40)"]
    models = ["RPCA_Unsupervised", "Ensemble_v2", "Ensemble_v3", "Ensemble_v4"]
    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4"]
    x = np.arange(len(regimes))
    bw = 0.19
    sub = metrics_df[metrics_df["split"] == "val_cal_hard"].set_index("model")
    for idx_m, (m, col) in enumerate(zip(models, colors)):
        vals = [float(sub.loc[m, r]) for r in regimes]
        ax.bar(x + (idx_m - 1.5) * bw, vals, width=bw, color=col, label=m.replace("_Unsupervised", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(r_labels, fontsize=8.5)
    ax.set_ylim(0.0, 0.64)
    ax.set_ylabel("Regime RMSE (Lower is Better)")
    ax.set_title("(c) Regime-Stratified Share Error (Cal-Hard Val)")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _plot_fig2_boundary_stress(
    cal_df: pd.DataFrame,
    stratum_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    # Panel A: Hydrophone Calibration Tilt Stress Curve (-6 to +10 dB/dec on Holdout)
    ax = axes[0]
    for m, col, mk in (
        ("RPCA_Unsupervised", "#7f7f7f", "d"),
        ("Ensemble_v2", "#d62728", "s"),
        ("Ensemble_v3", "#ff7f0e", "^"),
        ("Ensemble_v4", "#1f77b4", "o"),
    ):
        sub = cal_df[cal_df["model"] == m].sort_values("tilt_db_per_dec")
        ax.plot(sub["tilt_db_per_dec"], sub["roc_auc"], f"-{mk}", color=col, lw=1.9, ms=5, label=m.replace("_Unsupervised", ""))
    ax.axvline(0.0, color="#555555", ls=":", lw=1.0)
    ax.set_xlabel("Injected Hydrophone Tilt h(f) (dB / decade)")
    ax.set_ylabel("Unseen-Station ROC AUC (Presence)")
    ax.set_title("(a) Cross-Hydrophone Tilt Stress Test (Holdout)")
    ax.set_ylim(0.45, 1.01)
    ax.legend(frameon=True, facecolor="white", edgecolor="#dddddd", fontsize=8.0, loc="center left")
    ax.grid(alpha=0.25)

    # Panel B: Per-Stratum RMSE across SA physical strata (val_cal_hard + val_standard)
    ax = axes[1]
    strata = ["strict_hf_ship", "reef_active", "glitch_drop", "vlf_surf", "ambient_normal"]
    s_labels = ["Strict SA\nShip", "Shrimp\nReef", "Glitch\nDrop", "VLF\nSurf", "Ambient\nNormal"]
    sx = np.arange(len(strata))
    bw = 0.25
    for idx_m, (m, col) in enumerate((("Ensemble_v2", "#d62728"), ("Ensemble_v3", "#ff7f0e"), ("Ensemble_v4", "#1f77b4"))):
        sub = stratum_df[(stratum_df["scope"] == "val_combined") & (stratum_df["model"] == m)].set_index("group")
        vals = [float(sub.loc[s, "rmse"]) if s in sub.index else 0.0 for s in strata]
        ax.bar(sx + (idx_m - 1.0) * bw, vals, width=bw, color=col, label=m)
    ax.set_xticks(sx)
    ax.set_xticklabels(s_labels, fontsize=8.2)
    ax.set_ylabel("Stratum Share RMSE (Lower is Better)")
    ax.set_title("(b) Boundary Stratum Error (Cal-Hard + Std Val)")
    ax.legend(frameon=False, fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)

    # Panel C: JOMOPANS L_excess MAE (dB) across models & splits
    ax = axes[2]
    splits = ["val_cal_hard", "val_standard", "holdout"]
    sp_labels = ["Cal-Hard Val", "Standard Val", "Station Holdout"]
    mx = np.arange(len(splits))
    models = ["RPCA_Unsupervised", "Ensemble_v2", "Ensemble_v3", "Ensemble_v4"]
    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4"]
    bw2 = 0.19
    for idx_m, (m, col) in enumerate(zip(models, colors)):
        sub = metrics_df[metrics_df["model"] == m].set_index("split")
        vals = [float(sub.loc[sp, "jomopans_lex_mae_db"]) for sp in splits]
        ax.bar(mx + (idx_m - 1.5) * bw2, vals, width=bw2, color=col, label=m.replace("_Unsupervised", ""))
    ax.set_xticks(mx)
    ax.set_xticklabels(sp_labels, fontsize=8.5)
    ax.set_ylabel("JOMOPANS Excess-Level MAE (dB)")
    ax.set_title("(c) Acoustic Excess-Level Error L_ex (MSFD D11C2)")
    ax.legend(frameon=False, fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _plot_fig3_signal_boundaries(
    examples: list[tuple[str, np.ndarray]],
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(13.2, 10.6))
    freqs_lbl = ["31.6", "100", "316", "1k", "3.16k", "7.94k"]
    freq_pos = [0, 5, 10, 15, 20, 24]

    for row_i, (title, X_win) in enumerate(examples):
        obs = np.isfinite(X_win)
        Xc = X_win - float(np.nanmedian(X_win))
        pcen = compute_pcen_map(X_win)
        _, _, sp_list = eval_rpca_unsupervised(X_win[None, :, :])
        S_map = sp_list[0]

        mats = [
            (Xc, "Centered dB (Xc)", "RdYlBu_r", -18.0, 18.0),
            (pcen, "h(f)-Invariant PCEN Contrast", "viridis", 0.0, float(np.nanpercentile(pcen, 98))),
            (S_map, "RPCA Sparse Transient S(t, f)", "magma", 0.0, max(float(np.nanmax(S_map)), 0.5)),
        ]
        for col_j, (M, col_title, cmap, vmin, vmax) in enumerate(mats):
            ax = axes[row_i, col_j]
            im = ax.imshow(M.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_yticks(freq_pos)
            ax.set_yticklabels(freqs_lbl, fontsize=7.5)
            ax.set_xticks([0, 4, 9, 14])
            ax.set_xticklabels(["m01", "m05", "m10", "m15"], fontsize=7.5)
            if row_i == 0:
                ax.set_title(col_title, fontsize=10.0, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(title, fontsize=8.4)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle("Representative South African Boundary Cases: Spectrogram vs. PCEN Contrast vs. RPCA Sparse Transients", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def run_self_test() -> None:
    """Verify PCEN & RPCA global shift invariance and metric boundary identities."""
    rng = np.random.default_rng(7)
    X = rng.normal(85.0, 5.0, size=(12, 15, 25))
    X[:4, 0, :] = np.nan
    X[4:8, 6, :] -= 35.0
    X_sh = X + 37.5

    p1 = compute_pcen_map(X[0])
    p2 = compute_pcen_map(X_sh[0])
    pcen_diff = float(np.nanmax(np.abs(p1 - p2)))
    assert pcen_diff < 1e-10, f"PCEN shift diff={pcen_diff}"

    s1, c1, _ = eval_rpca_unsupervised(X)
    s2, c2, _ = eval_rpca_unsupervised(X_sh)
    rpca_diff = max(float(np.max(np.abs(s1 - s2))), float(np.max(np.abs(c1 - c2))))
    assert rpca_diff < 1e-10, f"RPCA shift diff={rpca_diff}"

    # Metric identities
    y = np.array([0.0, 0.0, 0.05, 0.25, 0.65, 0.85])
    sc = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    met = compute_standardized_metrics("test", "oracle", y, y, sc)
    assert np.isclose(met["zindi_score_1_minus_rmse"], 1.0)
    assert np.isclose(met["lin_ccc"], 1.0)
    assert np.isclose(met["nse_r2"], 1.0)
    assert np.isclose(met["ece_10"], 0.0)
    assert np.isclose(met["brier_rel"] - met["brier_res"] + met["brier_unc"], met["brier"], atol=1e-10)
    print(f"standardized_evaluation_suite self-test: PASS (pcen_diff={pcen_diff:.2e}, rpca_diff={rpca_diff:.2e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    warnings.filterwarnings("ignore", category=UserWarning)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load v4 benchmark splits & provenance
    tr_y = np.load(args.data_dir / "train.npz")["y"].astype(np.float64)
    mu_tr = float(tr_y.mean())
    p_pos_tr = float((tr_y > 0).mean())

    splits = {}
    for sp in ("val_cal_hard", "val_standard", "holdout"):
        d = np.load(args.data_dir / f"{sp}.npz")
        prov = pd.read_csv(args.data_dir / f"{sp}_provenance.csv")
        splits[sp] = (d["X"].astype(np.float64), d["y"].astype(np.float64), prov)

    # Load saved model bundles (infer only)
    b2 = joblib.load(ROOT / "models/v2/bundle.joblib")
    sys.modules["__main__"].ExcessSpectralPrototypeTransformer = ExcessSpectralPrototypeTransformer
    b3 = joblib.load(ROOT / "models/v3/bundle_v3.joblib")
    b4 = joblib.load(ROOT / "models/v4/bundle_v4.joblib")

    all_preds: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {sp: {} for sp in splits}
    metric_rows = []

    print("Running 16-metric + JOMOPANS + RPCA evaluation across val_cal_hard, val_standard, holdout...", flush=True)
    for sp, (X_sp, y_sp, _) in splits.items():
        all_preds[sp]["Constant_Mean"] = (np.full(len(y_sp), mu_tr), np.full(len(y_sp), p_pos_tr))
        all_preds[sp]["Physics_Robust_q20"] = compute_robust_physics_share(X_sp, q=20.0)
        sh_rpca, sc_rpca, _ = eval_rpca_unsupervised(X_sp)
        all_preds[sp]["RPCA_Unsupervised"] = (sh_rpca, sc_rpca)
        all_preds[sp]["Ensemble_v2"] = predict_v2(b2, X_sp)
        all_preds[sp]["Ensemble_v3"] = predict_v3(b3, X_sp)
        all_preds[sp]["Ensemble_v4"] = predict_v4(b4, X_sp)

        for m, (p_sh, p_sc) in all_preds[sp].items():
            metric_rows.append(compute_standardized_metrics(sp, m, y_sp, p_sh, p_sc))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out_dir / "standardized_metrics_full.csv", index=False)

    # Per-stratum & per-station breakdown
    stratum_rows = []
    # Combined val_cal_hard + val_standard by physical stratum
    y_comb = np.concatenate([splits["val_cal_hard"][1], splits["val_standard"][1]])
    st_comb = np.concatenate([splits["val_cal_hard"][2]["bg_stratum"].to_numpy(), splits["val_standard"][2]["bg_stratum"].to_numpy()])
    for m in ("RPCA_Unsupervised", "Ensemble_v2", "Ensemble_v3", "Ensemble_v4"):
        p_sh_c = np.concatenate([all_preds["val_cal_hard"][m][0], all_preds["val_standard"][m][0]])
        p_sc_c = np.concatenate([all_preds["val_cal_hard"][m][1], all_preds["val_standard"][m][1]])
        for grp_name in pd.unique(st_comb):
            mask = st_comb == grp_name
            row = compute_standardized_metrics("val_combined", m, y_comb[mask], p_sh_c[mask], p_sc_c[mask])
            row["scope"] = "val_combined"
            row["group"] = str(grp_name)
            stratum_rows.append(row)

    # Per-station on holdout
    _, y_ho, prov_ho = splits["holdout"]
    ho_stations = prov_ho["station"].to_numpy()
    for m in ("RPCA_Unsupervised", "Ensemble_v2", "Ensemble_v3", "Ensemble_v4"):
        p_sh_h, p_sc_h = all_preds["holdout"][m]
        for st_name in sorted(pd.unique(ho_stations)):
            mask = ho_stations == st_name
            row = compute_standardized_metrics("holdout_station", m, y_ho[mask], p_sh_h[mask], p_sc_h[mask])
            row["scope"] = "holdout_station"
            row["group"] = str(st_name)
            stratum_rows.append(row)

    stratum_df = pd.DataFrame(stratum_rows)
    stratum_df.to_csv(args.out_dir / "stratum_breakdown.csv", index=False)

    # Hydrophone calibration-tilt stress curve (-6 to +10 dB/dec on holdout)
    print("Running hydrophone calibration-tilt boundary curve (-6 to +10 dB/dec on Holdout)...", flush=True)
    X_ho, y_ho, _ = splits["holdout"]
    yb_ho = (y_ho > 0.0).astype(int)
    zero_ho = y_ho == 0.0
    log_f_norm = np.log10(EXACT_CENTRES / 500.0)[None, None, :]
    cal_rows = []
    for tilt in (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        X_warped = X_ho + tilt * log_f_norm
        eval_map = {
            "RPCA_Unsupervised": eval_rpca_unsupervised(X_warped)[:2],
            "Ensemble_v2": predict_v2(b2, X_warped),
            "Ensemble_v3": predict_v3(b3, X_warped),
            "Ensemble_v4": predict_v4(b4, X_warped),
        }
        for m, (sh_w, sc_w) in eval_map.items():
            auc_w = float(roc_auc_score(yb_ho, sc_w))
            zrmse_w = float(np.sqrt(np.mean(sh_w[zero_ho] ** 2)))
            ll_w = float(log_loss(yb_ho, np.clip(sc_w, 1e-7, 1.0 - 1e-7)))
            cal_rows.append(
                {
                    "tilt_db_per_dec": float(tilt),
                    "model": m,
                    "roc_auc": auc_w,
                    "zero_rmse": zrmse_w,
                    "logloss": ll_w,
                }
            )
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(args.out_dir / "calibration_shift_curve.csv", index=False)

    # Generate 3 publication-grade 300 DPI figures
    print("Rendering 300 DPI standardized figures...", flush=True)
    _plot_fig1_calibration_and_agreement(metrics_df, all_preds["val_cal_hard"], splits["val_cal_hard"][1], fig_dir / "fig1_standards_calibration_and_agreement.png")
    _plot_fig2_boundary_stress(cal_df, stratum_df, metrics_df, fig_dir / "fig2_boundary_stress_and_calibration_shift.png")

    # Select 4 real acoustic boundary exemplars from unwarped val_standard / holdout
    X_vs, y_vs, p_vs = splits["val_standard"]
    unwarped = ~p_vs["cal_warped"].to_numpy()
    idx_reef = int(np.flatnonzero(unwarped & (y_vs == 0) & (p_vs["bg_stratum"].to_numpy() == "reef_active"))[0])
    idx_ship = int(np.flatnonzero(unwarped & (y_vs == 0) & (p_vs["bg_stratum"].to_numpy() == "strict_hf_ship"))[0])
    idx_glitch = int(np.flatnonzero(unwarped & (y_vs == 0) & (p_vs["bg_stratum"].to_numpy() == "glitch_drop"))[0])
    idx_vtuad = int(np.flatnonzero(unwarped & (y_vs >= 0.25) & (p_vs["trajectory_mode"].to_numpy() == "short_block_vtuad"))[0])
    exemplars = [
        (f"Shrimp Reef ({p_vs.loc[idx_reef, 'station']})\ny = 0.0 (Pure Background)", X_vs[idx_reef]),
        (f"Real SA Vessel ({p_vs.loc[idx_ship, 'station']})\ny = 0.0 (Hard Negative)", X_vs[idx_ship]),
        (f"Dropout Glitch ({p_vs.loc[idx_glitch, 'station']})\ny = 0.0 (Hard Negative)", X_vs[idx_glitch]),
        (f"Added VTUAD Block\ny = {y_vs[idx_vtuad]:.3f} (Controlled Mixture)", X_vs[idx_vtuad]),
    ]
    _plot_fig3_signal_boundaries(exemplars, fig_dir / "fig3_acoustic_signal_boundaries_decomposition.png")

    summary = {
        "val_cal_hard_summary": metrics_df[metrics_df["split"] == "val_cal_hard"][
            ["model", "zindi_score_1_minus_rmse", "lin_ccc", "nse_r2", "spearman_rho", "jomopans_lex_mae_db", "roc_auc", "ece_10", "brier"]
        ].to_dict(orient="records"),
        "tilt_stress_plus8": cal_df[cal_df["tilt_db_per_dec"] == 8.0].to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== STANDARDIZED METRICS (val_cal_hard) ===")
    print(
        metrics_df[metrics_df["split"] == "val_cal_hard"][
            ["model", "zindi_score_1_minus_rmse", "lin_ccc", "nse_r2", "spearman_rho", "jomopans_lex_mae_db", "roc_auc", "ece_10", "brier_rel", "brier_res"]
        ].to_string(index=False)
    )
    print("\n=== HYDROPHONE TILT STRESS (+8 dB/dec on Unseen Holdout) ===")
    print(cal_df[cal_df["tilt_db_per_dec"] == 8.0].to_string(index=False))


if __name__ == "__main__":
    main()
