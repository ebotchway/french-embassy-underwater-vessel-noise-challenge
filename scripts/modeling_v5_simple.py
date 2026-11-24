#!/usr/bin/env python3
"""Simplified 100% h(f)-invariant Modeling v5-simple pipeline (42 invariant features, 3 simple models)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# Import torch before GBDT extensions on macOS
import torch  # noqa: F401
import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

from build_synthetic_mixtures_v1 import BANDS, SHAPE  # noqa: E402
from extract_external_decidecade import EXACT_CENTRES  # noqa: E402
from modeling_v3_simulator import (  # noqa: E402
    REF_DB,
    assign_disjoint_vessel_sources_v3,
    db_to_lin_fixed_ref,
    read_windows_fast,
)
from modeling_v4_simulator import (  # noqa: E402
    BG_MANIFEST_DEFAULT,
    HIGH_BAND_ATTEN_WEIGHT,
    HMS_PARQUET_DEFAULT,
    LOG10_F_NORM,
    NORM_BAND_COORD,
    VTUAD_PARQUET_DEFAULT,
    _build_stratum_queues,
    _pop_from_stratum,
    _sample_vessel_from_pool,
    apply_hydrophone_calibration,
    assign_balanced_run_folds,
    profile_backgrounds_v4,
    sample_minute_mask_v4,
    sample_requested_share_v4,
)
from standardized_evaluation_suite import rpca_ialm_decompose  # noqa: E402

OUT_DATA_DEFAULT = ROOT / "data/synthetic/modeling_v5_simple"
MODELS_DIR_DEFAULT = ROOT / "models/v5_simple"
REPORTS_DIR_DEFAULT = ROOT / "reports/modeling_v5_simple"
SUBMISSION_DEFAULT = ROOT / "submissions/candidate_v5_simple.csv"
TEST_CSV_DEFAULT = ROOT / "data/zindi/raw/Test.csv"
SAMPLE_SUB_DEFAULT = ROOT / "data/zindi/raw/sample_submission.csv"

TEST_COLS = [f"m{m:02d}_{b}" for m in range(1, 16) for b in BANDS]

ZONES = {
    "vlf": slice(0, 6),
    "lf": slice(6, 11),
    "mf": slice(11, 17),
    "hf": slice(17, 25),
    "bb": slice(0, 25),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lag1_autocorr(series: np.ndarray) -> float:
    if len(series) < 3 or float(np.std(series)) < 1e-8:
        return 0.0
    a, b = series[:-1], series[1:]
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(c) else c


def _per_band_pcen_invariant(X_clean: np.ndarray) -> np.ndarray:
    """Compute strictly per-band median-centered PCEN (algebraically h(f) & gain invariant)."""
    med_f = np.median(X_clean, axis=0, keepdims=True)
    X_bc = X_clean - med_f
    E_rel = np.power(10.0, np.clip(X_bc, -40.0, 40.0) / 10.0)

    s, alpha, delta, r, eps = 0.35, 0.90, 2.0, 0.50, 1e-6
    M = np.zeros_like(E_rel)
    m_prev = E_rel[0].copy()
    M[0] = m_prev
    for t in range(1, E_rel.shape[0]):
        m_prev = (1.0 - s) * m_prev + s * E_rel[t]
        M[t] = m_prev
    denom = np.power(eps + M, alpha)
    return np.power(E_rel / denom + delta, r) - (delta ** r)


def _extract_invariant_clean_minutes(X_obs: np.ndarray) -> np.ndarray:
    """Detect glitch-free minutes using strictly h(f)-invariant per-band median deviations."""
    t_cnt = X_obs.shape[0]
    if t_cnt < 6:
        return X_obs
    med_f = np.median(X_obs, axis=0, keepdims=True)
    # Center per band first so any additive h(f) + c cancels before dropout filtering
    dev_t = np.median(X_obs - med_f, axis=1)
    keep = dev_t >= -14.0
    if int(keep.sum()) >= 4:
        return X_obs[keep]
    top4 = np.argsort(dev_t)[-4:]
    mask = np.zeros(t_cnt, dtype=bool)
    mask[top4] = True
    return X_obs[mask]


def extract_features_v5_simple(X_db: np.ndarray) -> pd.DataFrame:
    """Extract 42 purely h(f)-invariant temporal excess, per-band PCEN, and RPCA features."""
    X = np.asarray(X_db, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (15, 25):
        raise ValueError(f"Expected (N, 15, 25), got {X.shape}")

    min_obs = np.isfinite(X).all(axis=2)
    assert min_obs.any(axis=1).all(), "Window with zero complete observed minutes"
    n = len(X)

    col_names: list[str] = []
    for z in ("vlf", "lf", "mf", "hf", "bb"):
        col_names.extend([f"r_{z}_max", f"r_{z}_mean", f"r_{z}_std", f"r_{z}_lag1"])
    col_names.extend([
        "r_hf_minus_vlf_max",
        "r_mf_minus_vlf_max",
        "r_lf_minus_vlf_max",
        "r_frac_gt_3db",
        "r_frac_gt_6db",
        "r_frac_gt_10db",
        "r_svd_rank1_ratio",
        "r_smooth_step_max",
        "r_drop_glitch_db",
        "r_peak_width_frac",
        "pcen_mean",
        "pcen_p90",
        "pcen_max",
        "pcen_std",
        "pcen_mf_hf_mean",
        "pcen_vlf_mean",
        "rpca_sparse_energy_frac",
        "rpca_lowrank_energy_frac",
        "rpca_sparse_max",
        "rpca_sparse_p95",
        "rpca_lowrank_std",
        "rpca_s_over_l_ratio",
    ])
    assert len(col_names) == 42

    feats = np.zeros((n, 42), dtype=np.float64)
    for i in range(n):
        X_obs = X[i, min_obs[i], :]
        X_cl = _extract_invariant_clean_minutes(X_obs)
        q20_f = np.percentile(X_cl, 20.0, axis=0, keepdims=True)
        R_cl = X_cl - q20_f

        row_vals: list[float] = []
        zone_maxes: dict[str, float] = {}
        r_bb = R_cl[:, ZONES["bb"]].mean(axis=1)
        for z_name, sl in ZONES.items():
            rz = R_cl[:, sl].mean(axis=1)
            z_max = float(np.max(rz))
            zone_maxes[z_name] = z_max
            row_vals.extend([
                z_max,
                float(np.mean(rz)),
                float(np.std(rz)),
                _lag1_autocorr(rz),
            ])

        # Group 2: Temporal morphology & cross-band excess structure (10)
        R_pos = np.maximum(R_cl, 0.0)
        sv = np.linalg.svd(R_pos, compute_uv=False)
        tot_sv2 = max(float(np.sum(sv ** 2)), 1e-12)
        rank1_ratio = float((sv[0] ** 2) / tot_sv2) if len(sv) > 0 else 0.0
        smooth_step = float(np.max(np.abs(np.diff(r_bb)))) if len(r_bb) >= 2 else 0.0
        drop_glitch = float(np.median(r_bb) - np.min(r_bb))
        r_bb_pk = float(np.max(r_bb))
        peak_width = float(np.mean(1.0 / (1.0 + np.exp(-np.clip((r_bb - (r_bb_pk - 6.0)) / 0.5, -40.0, 40.0)))))
        row_vals.extend([
            zone_maxes["hf"] - zone_maxes["vlf"],
            zone_maxes["mf"] - zone_maxes["vlf"],
            zone_maxes["lf"] - zone_maxes["vlf"],
            float(np.mean(1.0 / (1.0 + np.exp(-np.clip((R_cl - 3.0) / 0.5, -40.0, 40.0))))),
            float(np.mean(1.0 / (1.0 + np.exp(-np.clip((R_cl - 6.0) / 0.5, -40.0, 40.0))))),
            float(np.mean(1.0 / (1.0 + np.exp(-np.clip((R_cl - 10.0) / 0.5, -40.0, 40.0))))),
            rank1_ratio,
            smooth_step,
            drop_glitch,
            peak_width,
        ])

        # Group 3: Strictly per-band centered PCEN (6)
        pcen = _per_band_pcen_invariant(X_cl)
        row_vals.extend([
            float(np.mean(pcen)),
            float(np.percentile(pcen, 90.0)),
            float(np.max(pcen)),
            float(np.std(pcen)),
            float(np.mean(pcen[:, 11:25])),
            float(np.mean(pcen[:, 0:6])),
        ])

        # Group 4: R_cl RPCA low-rank vs sparse decomposition (6)
        M = np.log1p(R_pos)
        L, S = rpca_ialm_decompose(M, max_iter=30, tol=1e-4)
        e_m2 = max(float(np.sum(M ** 2)), 1e-12)
        e_s2 = float(np.sum(np.maximum(S, 0.0) ** 2))
        e_l2 = float(np.sum(np.maximum(L, 0.0) ** 2))
        row_vals.extend([
            e_s2 / e_m2,
            e_l2 / e_m2,
            float(np.max(S)),
            float(np.percentile(S, 95.0)),
            float(np.std(L)),
            e_s2 / max(e_l2, 1e-12),
        ])

        feats[i] = row_vals

    df = pd.DataFrame(feats, columns=col_names)
    assert np.isfinite(df.to_numpy()).all(), "Non-finite value in extract_features_v5_simple"
    return df


def _apply_broadband_vessel_warp_v5(V: np.ndarray, active_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply wide broadband LF/MF/HF spectral tilt (-12 to +6 dB/dec) to diversify ship types."""
    out = V.copy()
    if not active_mask.any():
        return out
    tilt = float(rng.uniform(-12.0, 6.0))
    ripple_amp = float(rng.uniform(0.0, 2.5))
    ripple_freq = float(rng.uniform(0.5, 2.0))
    ripple_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    warp_db = tilt * LOG10_F_NORM + ripple_amp * np.cos(2.0 * np.pi * ripple_freq * NORM_BAND_COORD + ripple_phase)
    jitter_db = rng.normal(0.0, 0.65, size=(15, 25))
    factor = np.power(10.0, (warp_db[None, :] + jitter_db) / 10.0)
    out[active_mask] *= factor[active_mask]
    return np.maximum(out, 0.0)


def synthesize_vessel_trajectory_v5(
    source_matrix: np.ndarray,
    rel_power: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int]:
    """Synthesize smooth broadband CPA & slow shipping trajectories (only 10% short blocks)."""
    n_avail = source_matrix.shape[0]
    t_idx = np.arange(15, dtype=np.float64)
    u = float(rng.random())

    if u < 0.10:
        dur = int(rng.integers(2, 6))
        if n_avail >= dur:
            src_start = int(rng.integers(0, n_avail - dur + 1))
            block = source_matrix[src_start : src_start + dur] * rel_power[src_start : src_start + dur, None]
        elif n_avail == 1:
            block = np.repeat(source_matrix, dur, axis=0)
        else:
            idxs = np.clip(np.round(np.linspace(0, n_avail - 1, dur)).astype(int), 0, n_avail - 1)
            block = source_matrix[idxs] * rel_power[idxs, None]
        ins_start = int(rng.integers(1, max(1, 14 - dur) + 1))
        V = np.zeros(SHAPE, dtype=np.float64)
        V[ins_start : ins_start + dur] = block
        V = _apply_broadband_vessel_warp_v5(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
        return V, "short_block", dur

    if n_avail == 1:
        base = np.repeat(source_matrix, 15, axis=0)
    else:
        start = int(rng.integers(0, n_avail))
        step_dir = 1 if rng.random() < 0.5 else -1
        idxs = np.clip(start + step_dir * np.arange(15), 0, n_avail - 1)
        base = source_matrix[idxs].copy()
    row_tot = np.maximum(base.sum(axis=1, keepdims=True), 1e-30)
    base_shape = base / row_tot * float(row_tot.mean())

    if u < 0.65:
        mode = "smooth_cpa_bell"
        t_cpa = float(rng.uniform(1.5, 12.5))
        width = float(rng.uniform(1.2, 4.5))
        exponent = float(rng.uniform(0.85, 1.45))
        dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / width) ** 2)
        env = np.power(dist_rel, -2.0 * exponent)
        hf_atten = -float(rng.uniform(0.5, 3.0)) * (dist_rel[:, None] - 1.0) * HIGH_BAND_ATTEN_WEIGHT[None, :]
        V = base_shape * (env[:, None] * np.power(10.0, hf_atten / 10.0))
        dur = int((env > 0.1 * env.max()).sum())
    elif u < 0.90:
        mode = "asym_or_entry_exit"
        if rng.random() < 0.6:
            t_cpa = float(rng.uniform(2.0, 12.0))
            w_l = float(rng.uniform(1.0, 4.0))
            w_r = float(rng.uniform(1.0, 4.0))
            widths = np.where(t_idx <= t_cpa, w_l, w_r)
            dist_rel = np.sqrt(1.0 + ((t_idx - t_cpa) / widths) ** 2)
            env = np.power(dist_rel, -2.0 * float(rng.uniform(0.9, 1.35)))
            V = base_shape * env[:, None]
            dur = int((env > 0.1 * env.max()).sum())
        else:
            dur = int(rng.integers(5, 14))
            ins_start = int(rng.integers(0, 16 - dur))
            env = np.zeros(15, dtype=np.float64)
            sub = np.linspace(-1.0, 1.0, dur)
            env[ins_start : ins_start + dur] = 0.25 + 0.75 * np.cos(0.45 * np.pi * sub) ** 2
            V = base_shape * env[:, None]
    else:
        mode = "slow_cruise_drift"
        dur = 15
        steps = rng.normal(0.0, 0.45, size=15)
        drift_db = np.clip(np.cumsum(steps) - np.mean(np.cumsum(steps)), -3.0, 3.0)
        V = base_shape * np.power(10.0, drift_db[:, None] / 10.0)

    V = _apply_broadband_vessel_warp_v5(V, active_mask=(V.sum(axis=1) > 0), rng=rng)
    return V, mode, dur


def _pick_stratum(probs: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(probs.keys())
    w = np.array([probs[k] for k in keys], dtype=np.float64)
    w /= w.sum()
    return str(rng.choice(keys, p=w))


def _build_split_v5(
    split_name: str,
    bg_pool: pd.DataFrame,
    vessel_pool: dict,
    n: int,
    zero_prob: float,
    cal_prob: float,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict:
    # Fix #1: NEVER label real SA passing ships (strict_hf_ship) as y=0 negatives!
    neg_probs = {"ambient_normal": 0.40, "vlf_surf": 0.25, "reef_active": 0.25, "glitch_drop": 0.10}
    pos_probs = {"ambient_normal": 0.55, "vlf_surf": 0.20, "reef_active": 0.18, "glitch_drop": 0.07}

    chosen: list[tuple[int, int, float]] = []
    used: set[int] = set()
    if split_name == "train":
        bg_folds = assign_balanced_run_folds(bg_pool, n_folds=5)
        per_fold = n // 5
        rem = n % 5
        for f in range(5):
            fp = bg_folds[bg_folds["cv_fold"] == f]
            sq, aq = _build_stratum_queues(fp, rng)
            sq_neg = {k: ([] if k == "strict_hf_ship" else v) for k, v in sq.items()}
            take = per_fold + (1 if f < rem else 0)
            qs = [sample_requested_share_v4(rng, zero_prob=zero_prob) for _ in range(take)]
            for q in sorted(qs):  # pop q == 0.0 negatives first so strict_hf_ship is never needed
                st = _pick_stratum(neg_probs if q == 0.0 else pos_probs, rng)
                target_sq = sq_neg if q == 0.0 else sq
                chosen.append((_pop_from_stratum(target_sq, st, aq, used), f, q))
        rng.shuffle(chosen)
    else:
        sq, aq = _build_stratum_queues(bg_pool, rng)
        sq_neg = {k: ([] if k == "strict_hf_ship" else v) for k, v in sq.items()}
        qs = [sample_requested_share_v4(rng, zero_prob=zero_prob) for _ in range(n)]
        for q in sorted(qs):  # pop q == 0.0 negatives first so strict_hf_ship is never needed
            st = _pick_stratum(neg_probs if q == 0.0 else pos_probs, rng)
            target_sq = sq_neg if q == 0.0 else sq
            chosen.append((_pop_from_stratum(target_sq, st, aq, used), -1, q))
        rng.shuffle(chosen)

    bg_indices = [c[0] for c in chosen]
    bg_mats = read_windows_fast(bg_pool, bg_indices)

    X = np.empty((n, 15, 25), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    folds = np.array([c[1] for c in chosen], dtype=np.int32)
    meta = []

    for i, (bg_idx, fold_i, q) in enumerate(chosen):
        rec = bg_pool.loc[bg_idx]
        B = db_to_lin_fixed_ref(bg_mats[i])
        if q == 0.0:
            V = np.zeros(SHAPE, dtype=np.float64)
            mode, dur, v_gid = "background_only", 0, ""
        else:
            src = _sample_vessel_from_pool(vessel_pool, fold_i if split_name == "train" else None, hms_prob=0.14, rng=rng)
            V_raw, mode, dur = synthesize_vessel_trajectory_v5(src["matrix"], src["rel_power"], rng)
            vs = float(V_raw.sum())
            alpha = 0.0 if vs <= 0.0 else (q * float(B.sum())) / ((1.0 - q) * vs)
            V = alpha * V_raw
            v_gid = src["group_id"]

        B_cal, V_cal, cal_app = apply_hydrophone_calibration(B, V, rng, prob=cal_prob)
        M = B_cal + V_cal
        obs_min = sample_minute_mask_v4(rng, str(rec.stratum))
        obs_cell = np.repeat(obs_min[:, None], 25, axis=1)

        m_obs = float(M[obs_cell].sum())
        v_obs = float(V_cal[obs_cell].sum())
        y_obs = 0.0 if (q == 0.0 or m_obs <= 0.0) else float(np.clip(v_obs / m_obs, 0.0, 1.0))

        M_db = REF_DB + 10.0 * np.log10(np.maximum(M, 1e-30))
        M_db[~obs_cell] = np.nan
        X[i] = M_db.astype(np.float32)
        y[i] = np.float32(y_obs)
        meta.append({
            "example": i,
            "split": split_name,
            "fold": int(fold_i),
            "station": str(rec.station),
            "bg_stratum": str(rec.stratum),
            "cal_warped": bool(cal_app),
            "bg_run_id": str(rec.run_id),
            "vessel_group_id": v_gid,
            "trajectory_mode": mode,
            "observed_share": y_obs,
        })

    assert not np.any(pd.DataFrame(meta).loc[y == 0, "bg_stratum"] == "strict_hf_ship"), (
        "strict_hf_ship leaked into y=0 negatives!"
    )
    npz_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(npz_path, X=X, y=y, fold=folds)
    pd.DataFrame(meta).to_csv(out_dir / f"{split_name}_provenance.csv", index=False)
    return {
        "split": split_name,
        "rows": n,
        "zero_fraction": float((y == 0.0).mean()),
        "mean_share": float(y.mean()),
        "cal_warped_fraction": float(np.mean([m["cal_warped"] for m in meta])),
    }


def _fit_simplex_weights(P: np.ndarray, y_true: np.ndarray, mode: str) -> np.ndarray:
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def _loss(w: np.ndarray) -> float:
        pred = np.clip(P @ w, 1e-6, 1.0 - 1e-6)
        if mode == "rmse":
            return float(np.sqrt(np.mean((pred - y_true) ** 2)))
        return float(log_loss(y_true, pred))

    res = minimize(
        _loss,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
    )
    w = np.clip(res.x if res.success else w0, 0.0, 1.0)
    return w / max(float(w.sum()), 1e-12)


def _fit_temperature(prob: np.ndarray, y_bin: np.ndarray) -> float:
    p = np.clip(prob, 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))
    res = minimize(
        lambda t: float(log_loss(y_bin, 1.0 / (1.0 + np.exp(-logits / float(t[0]))))),
        x0=np.array([1.0]),
        bounds=[(0.25, 4.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0]) if res.success else 1.0


def predict_from_bundle_v5_simple(bundle: dict, X_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic inference for v5-simple (algebraically invariant to any static h(f) + constant shift)."""
    X_feat = extract_features_v5_simple(X_db)[bundle["feature_names"]]
    sh_mat = np.zeros((len(X_feat), 3), dtype=np.float64)
    sc_mat = np.zeros((len(X_feat), 3), dtype=np.float64)
    for fi, m_dict in enumerate(bundle["folds"]):
        for j, name in enumerate(["Linear_v5", "LightGBM_v5", "CatBoost_v5"]):
            sh_mat[:, j] += np.clip(m_dict["share"][name].predict(X_feat), 0.0, 1.0) / 5.0
            sc_mat[:, j] += np.clip(m_dict["score"][name].predict_proba(X_feat)[:, 1], 1e-6, 1.0 - 1e-6) / 5.0

    sh_w = np.asarray(bundle["share_weights_arr"], dtype=np.float64)
    sc_w = np.asarray(bundle["score_weights_arr"], dtype=np.float64)
    share = np.clip(sh_mat @ sh_w, 0.0, 1.0)
    raw_sc = np.clip(sc_mat @ sc_w, 1e-6, 1.0 - 1e-6)
    logits = np.log(raw_sc / (1.0 - raw_sc))
    T = float(bundle["score_temperature"])
    score = np.clip(1.0 / (1.0 + np.exp(-logits / T)), 0.0, 1.0)
    return share, score


def _eval_split(y: np.ndarray, sh: np.ndarray, sc: np.ndarray, split: str, model: str) -> dict:
    yb = (y > 0.0).astype(int)
    z0 = y == 0.0
    rmse = float(root_mean_squared_error(y, sh))
    return {
        "split": split,
        "model": model,
        "n": int(len(y)),
        "zindi_1_minus_rmse": float(1.0 - rmse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, sh)),
        "zero_rmse": float(root_mean_squared_error(y[z0], sh[z0])) if z0.any() else np.nan,
        "roc_auc": float(roc_auc_score(yb, sc)) if len(np.unique(yb)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(yb, sc)) if yb.any() else np.nan,
        "logloss": float(log_loss(yb, np.clip(sc, 1e-6, 1.0 - 1e-6))) if len(np.unique(yb)) == 2 else np.nan,
        "brier": float(brier_score_loss(yb, sc)) if len(np.unique(yb)) == 2 else np.nan,
    }


def run_self_test() -> None:
    """Verify exact h(f) + constant-gain + NaN-mask invariance across all 42 features."""
    rng = np.random.default_rng(42)
    X = rng.uniform(65.0, 95.0, size=(8, 15, 25))
    X[:, 0, :] = np.nan
    X[2:5, 14, :] = np.nan
    X[0, :, 10:] += 12.0 * np.exp(-0.5 * ((np.arange(15) - 7) / 2.0) ** 2)[:, None]

    # Inject near-threshold dropout glitch to verify glitch filtering itself never depends on h(f)
    X[1, 5, :] -= 13.95
    X[2, 5, :] -= 14.05

    h_f = 10.0 * LOG10_F_NORM + 4.0 * np.cos(2.0 * np.pi * NORM_BAND_COORD * 1.3 + 0.5) + 27.5
    X_warp = X + h_f[None, None, :]

    f0 = extract_features_v5_simple(X)
    f1 = extract_features_v5_simple(X_warp)
    assert f0.shape == (8, 42)
    max_diff = float(np.max(np.abs(f0.to_numpy() - f1.to_numpy())))
    assert max_diff <= 1e-11, f"v5_simple features not h(f)-invariant: max|diff|={max_diff}"
    print(f"PASS modeling_v5_simple self-test (42 exact h(f)-inv features, max|diff|={max_diff:.2e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=OUT_DATA_DEFAULT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    parser.add_argument("--submission-out", type=Path, default=SUBMISSION_DEFAULT)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    t0 = time.perf_counter()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build leak-free broadband v5_simple splits
    print("[1/4] Building leak-free v5_simple splits (zero strict_hf_ship in y=0, broadband CPA warps)...", flush=True)
    bg_all = profile_backgrounds_v4(pd.read_csv(BG_MANIFEST_DEFAULT))
    vt_df = pd.read_parquet(VTUAD_PARQUET_DEFAULT)
    hms_df = pd.read_parquet(HMS_PARQUET_DEFAULT)
    tr_v, va_v, ho_v = assign_disjoint_vessel_sources_v3(vt_df, hms_df, seed=42)
    rng = np.random.default_rng(42)

    bg_train = bg_all[bg_all["split"] == "train"].copy()
    bg_val = bg_all[bg_all["split"] == "validation"].copy()
    bg_ho = bg_all[bg_all["split"] == "station_holdout"].copy()

    sim_manifest = {
        "train": _build_split_v5("train", bg_train, tr_v, 8000, 0.50, 0.75, rng, args.data_dir),
        "val": _build_split_v5("val", bg_val, va_v, 1500, 0.50, 1.00, rng, args.data_dir),
        "holdout": _build_split_v5("holdout", bg_ho, ho_v, 541, 0.50, 0.00, rng, args.data_dir),
    }

    # 2. Extract 42 purely h(f)-invariant features
    print("[2/4] Extracting 42 purely h(f)-invariant features...", flush=True)
    tr_z = np.load(args.data_dir / "train.npz")
    va_z = np.load(args.data_dir / "val.npz")
    ho_z = np.load(args.data_dir / "holdout.npz")

    X_tr_df = extract_features_v5_simple(tr_z["X"])
    X_va_df = extract_features_v5_simple(va_z["X"])
    X_ho_df = extract_features_v5_simple(ho_z["X"])
    feat_names = list(X_tr_df.columns)

    y_tr, folds_tr = tr_z["y"].astype(np.float64), tr_z["fold"]
    y_va, y_ho = va_z["y"].astype(np.float64), ho_z["y"].astype(np.float64)
    yb_tr = (y_tr > 0.0).astype(int)

    # 3. Train 3 simple regularized models across 5 folds
    print("[3/4] Training 5-fold Linear_v5, shallow LightGBM_v5, and shallow CatBoost_v5...", flush=True)
    model_names = ["Linear_v5", "LightGBM_v5", "CatBoost_v5"]
    oof_sh = np.zeros((len(y_tr), 3), dtype=np.float64)
    oof_sc = np.zeros((len(y_tr), 3), dtype=np.float64)
    fold_bundles = []

    for f in range(5):
        tr_m = folds_tr != f
        va_m = folds_tr == f
        Xf_tr, Xf_va = X_tr_df.iloc[tr_m], X_tr_df.iloc[va_m]
        yf_tr, ybf_tr = y_tr[tr_m], yb_tr[tr_m]

        lin_sh = make_pipeline(StandardScaler(), Ridge(alpha=100.0, random_state=42 + f))
        lin_sc = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=1000, random_state=42 + f))

        lgb_sh = lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.04, num_leaves=15, max_depth=4,
            min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=10.0, random_state=42 + f, n_jobs=1, verbosity=-1,
        )
        lgb_sc = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.04, num_leaves=15, max_depth=4,
            min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=10.0, random_state=42 + f, n_jobs=1, verbosity=-1,
        )

        cb_sh = cb.CatBoostRegressor(
            iterations=350, learning_rate=0.04, depth=4, l2_leaf_reg=12.0,
            random_seed=42 + f, thread_count=1, verbose=False,
        )
        cb_sc = cb.CatBoostClassifier(
            iterations=350, learning_rate=0.04, depth=4, l2_leaf_reg=12.0,
            random_seed=42 + f, thread_count=1, verbose=False,
        )

        sh_models = {"Linear_v5": lin_sh, "LightGBM_v5": lgb_sh, "CatBoost_v5": cb_sh}
        sc_models = {"Linear_v5": lin_sc, "LightGBM_v5": lgb_sc, "CatBoost_v5": cb_sc}
        for j, mname in enumerate(model_names):
            sh_models[mname].fit(Xf_tr, yf_tr)
            sc_models[mname].fit(Xf_tr, ybf_tr)
            oof_sh[va_m, j] = np.clip(sh_models[mname].predict(Xf_va), 0.0, 1.0)
            oof_sc[va_m, j] = np.clip(sc_models[mname].predict_proba(Xf_va)[:, 1], 1e-6, 1.0 - 1e-6)

        fold_bundles.append({"share": sh_models, "score": sc_models})

    sh_w = _fit_simplex_weights(oof_sh, y_tr, "rmse")
    sc_w = _fit_simplex_weights(oof_sc, yb_tr, "logloss")
    oof_sc_blend = np.clip(oof_sc @ sc_w, 1e-6, 1.0 - 1e-6)
    T_temp = _fit_temperature(oof_sc_blend, yb_tr)

    bundle = {
        "feature_names": feat_names,
        "folds": fold_bundles,
        "share_weights": {m: float(w) for m, w in zip(model_names, sh_w)},
        "score_weights": {m: float(w) for m, w in zip(model_names, sc_w)},
        "share_weights_arr": sh_w.tolist(),
        "score_weights_arr": sc_w.tolist(),
        "score_temperature": T_temp,
    }
    bundle_path = args.models_dir / "bundle_v5_simple.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    bundle_reloaded = joblib.load(bundle_path)

    # 4. Evaluate across v5_simple splits, v4 splits, real unmixed SA strata, and Test.csv
    print("[4/4] Benchmarking invariance, v4 stress splits, real SA strata, and Test.csv...", flush=True)
    rows_eval = []
    for sp_name, X_arr, y_arr in [
        ("v5_val_100pct_cal", va_z["X"], y_va),
        ("v5_holdout", ho_z["X"], y_ho),
    ]:
        sh, sc = predict_from_bundle_v5_simple(bundle_reloaded, X_arr)
        rows_eval.append(_eval_split(y_arr, sh, sc, sp_name, "Ensemble_v5_Simple"))

    for v4_sp in ["val_cal_hard", "val_standard", "holdout"]:
        z4 = np.load(ROOT / f"data/synthetic/modeling_v4/{v4_sp}.npz")
        sh, sc = predict_from_bundle_v5_simple(bundle_reloaded, z4["X"])
        rows_eval.append(_eval_split(z4["y"].astype(np.float64), sh, sc, f"v4_{v4_sp}", "Ensemble_v5_Simple"))

    # Exact +10 dB/dec + 4 dB ripple + 27.5 dB warp check on holdout (features <= 1e-11, p95 <= 1e-12)
    warp_vec = (10.0 * LOG10_F_NORM + 4.0 * np.cos(2.0 * np.pi * NORM_BAND_COORD * 1.3 + 0.5) + 27.5)[None, None, :]
    ho_X64 = ho_z["X"].astype(np.float64)
    f_ho0 = extract_features_v5_simple(ho_X64).to_numpy()
    f_how = extract_features_v5_simple(ho_X64 + warp_vec).to_numpy()
    ho_feat_diff = float(np.max(np.abs(f_ho0 - f_how)))
    sh_h0, sc_h0 = predict_from_bundle_v5_simple(bundle_reloaded, ho_X64)
    sh_hw, sc_hw = predict_from_bundle_v5_simple(bundle_reloaded, ho_X64 + warp_vec)
    ho_warp_sh_diff = float(np.max(np.abs(sh_h0 - sh_hw)))
    ho_warp_sc_diff = float(np.max(np.abs(sc_h0 - sc_hw)))
    ho_warp_sh_p95 = float(np.percentile(np.abs(sh_h0 - sh_hw), 95.0))
    assert ho_feat_diff <= 1e-11 and ho_warp_sh_p95 <= 1e-12 and ho_warp_sh_diff <= 5e-3 and ho_warp_sc_diff <= 5e-3, (
        f"Holdout warp diff too large: feat={ho_feat_diff}, share={ho_warp_sh_diff}, score={ho_warp_sc_diff}"
    )

    # Real unmixed SA strata response (check Fish Hoek + Sea Point unmixed y=0 & Dev validation sample)
    z4_ho = np.load(ROOT / "data/synthetic/modeling_v4/holdout.npz")
    p4_ho = pd.read_csv(ROOT / "data/synthetic/modeling_v4/holdout_provenance.csv")
    m0 = z4_ho["y"] == 0
    sh_real0, sc_real0 = predict_from_bundle_v5_simple(bundle_reloaded, z4_ho["X"][m0])
    real0_df = p4_ho[m0].copy()
    real0_df["v5_share"] = sh_real0
    real0_df["v5_score"] = sc_real0
    real_strata_means = real0_df.groupby("bg_stratum")[["v5_share", "v5_score"]].mean().to_dict("index")

    # Test.csv strictly inference-only
    test_df = pd.read_csv(TEST_CSV_DEFAULT)
    sample_sub = pd.read_csv(SAMPLE_SUB_DEFAULT)
    assert (test_df["ID"] == sample_sub["ID"]).all()
    X_test = test_df[TEST_COLS].to_numpy(dtype=np.float64).reshape(-1, 15, 25)
    f_t0 = extract_features_v5_simple(X_test).to_numpy()
    f_tw = extract_features_v5_simple(X_test + warp_vec).to_numpy()
    test_feat_diff = float(np.max(np.abs(f_t0 - f_tw)))
    sh_test, sc_test = predict_from_bundle_v5_simple(bundle_reloaded, X_test)
    sh_test_w, sc_test_w = predict_from_bundle_v5_simple(bundle_reloaded, X_test + warp_vec)
    test_warp_sh_diff = float(np.max(np.abs(sh_test - sh_test_w)))
    test_warp_sc_diff = float(np.max(np.abs(sc_test - sc_test_w)))
    test_warp_sh_p95 = float(np.percentile(np.abs(sh_test - sh_test_w), 95.0))
    assert test_feat_diff <= 1e-11 and test_warp_sh_p95 <= 1e-12 and test_warp_sh_diff <= 5e-3 and test_warp_sc_diff <= 5e-3, (
        f"Test.csv warp diff too large: feat={test_feat_diff}, share={test_warp_sh_diff}, score={test_warp_sc_diff}"
    )

    sub_out = pd.DataFrame({
        "ID": sample_sub["ID"],
        "share_added_vessel": sh_test,
        "score_added_vessel": sc_test,
    })
    assert list(sub_out.columns) == list(sample_sub.columns) and len(sub_out) == 1500
    assert np.isfinite(sh_test).all() and np.isfinite(sc_test).all()
    assert sub_out["share_added_vessel"].between(0.0, 1.0).all()
    assert sub_out["score_added_vessel"].between(0.0, 1.0).all()
    sub_out.to_csv(args.submission_out, index=False)

    sub_v4 = pd.read_csv(ROOT / "submissions/candidate_v4_ensemble.csv")
    sub_v2 = pd.read_csv(ROOT / "submissions/candidate_v2_ensemble.csv")

    eval_df = pd.DataFrame(rows_eval)
    eval_df.to_csv(args.reports_dir / "comparison_metrics.csv", index=False)

    imp_lgb = np.mean([f["share"]["LightGBM_v5"].feature_importances_ for f in fold_bundles], axis=0)
    imp_df = pd.DataFrame({"feature": feat_names, "lgb_gain_count": imp_lgb}).sort_values("lgb_gain_count", ascending=False)
    imp_df.to_csv(args.reports_dir / "feature_importance.csv", index=False)

    summary = {
        "elapsed_s": float(time.perf_counter() - t0),
        "sim_manifest": sim_manifest,
        "share_weights": bundle["share_weights"],
        "score_weights": bundle["score_weights"],
        "score_temperature": T_temp,
        "holdout_warp_max_diff_share": ho_warp_sh_diff,
        "holdout_warp_max_diff_score": ho_warp_sc_diff,
        "test_warp_max_diff_share": test_warp_sh_diff,
        "test_warp_max_diff_score": test_warp_sc_diff,
        "test_share_mean": float(sh_test.mean()),
        "test_share_std": float(sh_test.std()),
        "test_share_q10_q50_q90": [float(np.quantile(sh_test, q)) for q in (0.1, 0.5, 0.9)],
        "test_score_mean": float(sc_test.mean()),
        "test_score_std": float(sc_test.std()),
        "test_score_q10_q50_q90": [float(np.quantile(sc_test, q)) for q in (0.1, 0.5, 0.9)],
        "test_score_gt_095_frac": float((sc_test > 0.95).mean()),
        "corr_v5_v4_share": float(np.corrcoef(sh_test, sub_v4["share_added_vessel"])[0, 1]),
        "corr_v5_v4_score": float(np.corrcoef(sc_test, sub_v4["score_added_vessel"])[0, 1]),
        "corr_v5_v2_share": float(np.corrcoef(sh_test, sub_v2["share_added_vessel"])[0, 1]),
        "real_sa_holdout_y0_strata": real_strata_means,
        "submission_sha256": _sha256(args.submission_out),
    }
    (args.reports_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== v5_simple Evaluation Metrics ===")
    print(eval_df.round(4).to_string(index=False))
    print("\n=== Top-10 v5_simple Features ===")
    print(imp_df.head(10).to_string(index=False))
    print("\n=== Test.csv & Real SA Unmixed Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
