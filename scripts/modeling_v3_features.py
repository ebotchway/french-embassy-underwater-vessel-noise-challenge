#!/usr/bin/env python3
"""Physics-informed, glitch-robust, shift-invariant tabular & 7-channel CNN tensor extractors (v3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from extract_external_decidecade import EXACT_CENTRES  # noqa: E402

QUANTILES = (10, 15, 20, 25, 35, 50)
SUB_BANDS = {
    "vlf": slice(0, 6),    # 31.6 - 100 Hz
    "lf": slice(6, 13),    # 125.9 - 501.2 Hz
    "mf": slice(13, 19),   # 631 - 1995.3 Hz
    "hf": slice(19, 25),   # 2511.9 - 7943.3 Hz
}
LOG10_F_C = np.log10(EXACT_CENTRES) - np.log10(EXACT_CENTRES).mean()
LOG10_F_VAR = float(np.sum(LOG10_F_C ** 2))
FREQ_COORD = np.linspace(-1.0, 1.0, 25, dtype=np.float64)


def _safe_corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a_c * b_c) / denom)


def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_n = np.maximum(p, 0.0)
    q_n = np.maximum(q, 0.0)
    p_n = p_n / max(float(p_n.sum()), 1e-12)
    q_n = q_n / max(float(q_n.sum()), 1e-12)
    m = 0.5 * (p_n + q_n)
    kl_pm = np.sum(np.where(p_n > 1e-12, p_n * np.log2(np.maximum(p_n, 1e-12) / np.maximum(m, 1e-12)), 0.0))
    kl_qm = np.sum(np.where(q_n > 1e-12, q_n * np.log2(np.maximum(q_n, 1e-12) / np.maximum(m, 1e-12)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def _prepare_clean_masks(X_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center X_db cleanly and build glitch-cleaned minute masks."""
    X = np.asarray(X_db, dtype=np.float64)
    n = X.shape[0]
    obs = np.isfinite(X)
    min_obs = obs.all(axis=2)  # (N, 15)
    assert min_obs.any(axis=1).all(), "Window with zero observed minutes"

    win_med = np.nanmedian(X, axis=(1, 2), keepdims=True)
    Xc = X - win_med
    E_lin = np.where(obs, np.power(10.0, np.clip(Xc, -60.0, 60.0) / 10.0), np.nan)
    min_tot_e = np.where(min_obs, np.nansum(E_lin, axis=2), np.nan)
    min_tot_db = 10.0 * np.log10(np.maximum(min_tot_e, 1e-30))
    min_tot_db[~min_obs] = np.nan

    clean_min = np.zeros_like(min_obs)
    med_min_db = np.nanmedian(min_tot_db, axis=1)
    for i in range(n):
        valid_i = min_obs[i]
        cnt = int(valid_i.sum())
        if cnt < 6:
            clean_min[i] = valid_i
        else:
            cand = valid_i & (min_tot_db[i] >= med_min_db[i] - 14.0)
            if int(cand.sum()) >= 4:
                clean_min[i] = cand
            else:
                # Keep top-4 valid minutes cleanly
                order = np.argsort(np.where(valid_i, min_tot_db[i], -np.inf))
                clean_min[i, order[-4:]] = True
    return Xc, obs, min_obs, clean_min, E_lin, min_tot_db


def compute_robust_physics_share(X_db: np.ndarray, q: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Zero-training glitch-robust analytical excess share & presence score (strictly shift-invariant)."""
    Xc, obs, min_obs, clean_min, E_lin, min_tot_db = _prepare_clean_masks(X_db)
    n = Xc.shape[0]
    shares = np.zeros(n, dtype=np.float64)
    delta_max = np.zeros(n, dtype=np.float64)
    eps = 1e-30
    for i in range(n):
        c_idx = np.flatnonzero(clean_min[i])
        base_db = np.percentile(Xc[i, c_idx], q, axis=0)
        base_lin = np.power(10.0, np.clip(base_db, -60.0, 60.0) / 10.0)
        # Compute share over ALL observed cells using glitch-clean baseline
        v_idx = np.flatnonzero(min_obs[i])
        ex = np.maximum(E_lin[i, v_idx] - base_lin[None, :], 0.0)
        tot = float(E_lin[i, v_idx].sum())
        shares[i] = float(np.clip(ex.sum() / max(tot, eps), 0.0, 1.0))
        c_db = min_tot_db[i, c_idx]
        delta_max[i] = float(np.percentile(c_db, 95) - np.percentile(c_db, q))
    score = 1.0 / (1.0 + np.exp(-0.45 * (delta_max - 4.0)))
    return shares, np.clip(score, 0.0, 1.0)


def _series_morphology(series_db: np.ndarray, idx_clean: np.ndarray) -> dict[str, float]:
    """Compute CPA envelope width, interiority, concavity, lag-1 corr, and sharpness for one time series."""
    v = series_db[idx_clean]
    if len(v) < 2:
        return {
            "w6": float(len(v)),
            "w3": float(len(v)),
            "interior": 0.0,
            "prom": 0.0,
            "range": 0.0,
            "quad_r2": 0.0,
            "concav": 0.0,
            "lag1": 0.0,
            "asym": 0.0,
            "sharp": 0.0,
            "argmax": 7.0,
        }
    vmax = float(v.max())
    vmin = float(v.min())
    rel = v - float(np.median(v))
    amx_rel = int(np.argmax(v))
    t_peak = int(idx_clean[amx_rel])
    interior = float(1 <= t_peak <= 13)
    w6 = float((v >= vmax - 6.0).sum())
    w3 = float((v >= vmax - 3.0).sum())
    edge_mean = 0.5 * (rel[0] + rel[-1])
    prom = float(rel.max() - edge_mean)
    rng = float(vmax - vmin)

    t_norm = (idx_clean.astype(np.float64) - 7.0) / 7.0
    if len(t_norm) >= 4:
        coeffs = np.polyfit(t_norm, rel, deg=2)
        pred = np.polyval(coeffs, t_norm)
        ss_res = float(np.sum((rel - pred) ** 2))
        ss_tot = float(np.sum((rel - rel.mean()) ** 2))
        quad_r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), -1.0, 1.0)) if ss_tot > 1e-9 else 0.0
        concav = float(-coeffs[0])
    else:
        quad_r2, concav = 0.0, 0.0

    lag1 = _safe_corrcoef_1d(rel[:-1], rel[1:]) if len(rel) >= 4 else 0.0
    asym = float(abs(t_peak - 7.0) / 7.0)
    nbrs = []
    if amx_rel > 0:
        nbrs.append(v[amx_rel - 1])
    if amx_rel < len(v) - 1:
        nbrs.append(v[amx_rel + 1])
    sharp = float(vmax - (sum(nbrs) / len(nbrs))) if nbrs else 0.0

    return {
        "w6": w6,
        "w3": w3,
        "interior": interior,
        "prom": prom,
        "range": rng,
        "quad_r2": quad_r2,
        "concav": concav,
        "lag1": lag1,
        "asym": asym,
        "sharp": sharp,
        "argmax": float(t_peak),
    }


def extract_features_v3(X_db: np.ndarray) -> pd.DataFrame:
    """Extract shift-invariant, glitch-robust, excess-spectrum tabular features from (N, 15, 25) dB."""
    X = np.asarray(X_db, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (15, 25):
        raise ValueError(f"Expected shape (N, 15, 25), got {X.shape}")

    n = X.shape[0]
    Xc, obs, min_obs, clean_min, E_lin, min_tot_db = _prepare_clean_masks(X)
    eps = 1e-30

    # Precompute subband dB time series
    sb_db: dict[str, np.ndarray] = {}
    for sb_name, sl in SUB_BANDS.items():
        se = np.where(min_obs, np.nansum(E_lin[:, :, sl], axis=2), np.nan)
        sdb = 10.0 * np.log10(np.maximum(se, eps))
        sdb[~min_obs] = np.nan
        sb_db[sb_name] = sdb

    # Preallocate dictionaries of arrays
    col_names_fixed = (
        ["glitch_n_drop_min", "glitch_med_minus_min_db", "glitch_max_adj_drop_db", "phys_share_raw_q15"]
        + [f"phys_share_q{q}" for q in QUANTILES]
        + [f"phys_logit_q{q}" for q in QUANTILES]
        + ["phys_share_q10_minus_q50", "phys_share_q15_minus_q35", "phys_share_q20_ratio_q35"]
        + [
            "bb_clean_range_max_min", "bb_clean_range_95_20", "bb_clean_upper_tail_95_50",
            "bb_clean_lower_tail_50_20", "bb_clean_asymmetry", "bb_clean_std_db",
            "bb_clean_iqr_db", "bb_clean_frac_gt_p20_3db", "bb_clean_frac_gt_p20_6db",
            "bb_clean_frac_gt_p20_10db", "bb_clean_lin_max_over_p20",
        ]
        + [f"ex_tot_db_b{b:02d}" for b in range(25)]
        + [f"ex_hfp_db_b{b:02d}" for b in (0, 5, 9, 13, 15, 19, 21, 23)]
        + [
            "ex_frac_vlf", "ex_frac_lf", "ex_frac_mf", "ex_frac_hf",
            "ex_slope_db_per_dec", "ex_curv_mid_vs_ends", "ex_entropy",
            "ex_flatness_db", "ex_crest", "ex_peak_band_idx",
            "ex_631_fish_contrast", "ex_shrimp_6309_contrast",
            "ex_2512_minus_31", "ex_1000_minus_63", "ex_3981_minus_100",
            "ex_bg_cos_tot", "ex_bg_l1_tot", "ex_bg_jsd_tot",
            "ex_bg_cos_hfp", "ex_bg_l1_hfp", "ex_bg_jsd_hfp",
            "svd_rank1_ratio", "svd_rank2_ratio", "svd_eff_rank", "svd_lin_rank1_ratio",
            "top3_ex_spec_cos_mean", "top3_ex_spec_cos_min",
        ]
    )
    # Allocate arrays
    data: dict[str, np.ndarray] = {c: np.zeros(n, dtype=np.float64) for c in col_names_fixed}

    morph_cols: list[str] = []
    for ser in ("bb", "vlf", "lf", "mf", "hf"):
        for k in ("w6", "w3", "interior", "prom", "range", "quad_r2", "concav", "lag1", "asym", "sharp"):
            morph_cols.append(f"m_{ser}_{k}")
    morph_cols += [
        "w6_lf_minus_hf", "w6_bb_minus_hf",
        "argmax_absdiff_lf_hf", "argmax_absdiff_mf_hf", "argmax_absdiff_bb_hf", "argmax_absdiff_vlf_hf",
        "tcorr_vlf_lf", "tcorr_lf_mf", "tcorr_mf_hf", "tcorr_lf_hf", "tcorr_vlf_hf",
        "morph_mean_comod", "morph_frac_comod_065", "morph_low_high_comod_diff",
        "quiet_d63_shrimp", "quiet_d79_nyquist", "quiet_hf_minus_vlf", "quiet_mf_minus_vlf",
        "quiet_tilt_slope", "loud_hf_minus_vlf", "delta_hf_minus_vlf",
        "active_quiet_tilt_change", "active_quiet_shape_cosine",
    ]
    for sb in ("vlf", "lf", "mf", "hf"):
        morph_cols += [f"{sb}_share_q20", f"{sb}_contrib_q20", f"{sb}_range_95_20", f"{sb}_std_db"]
    morph_cols.append("mask_obs_frac")
    for c in morph_cols:
        data[c] = np.zeros(n, dtype=np.float64)

    for i in range(n):
        v_idx = np.flatnonzero(min_obs[i])
        c_idx = np.flatnonzero(clean_min[i])

        # Glitch stats on raw valid minutes
        v_db = min_tot_db[i, v_idx]
        n_drop = int(len(v_idx) - len(c_idx))
        data["glitch_n_drop_min"][i] = float(n_drop)
        data["glitch_med_minus_min_db"][i] = float(np.median(v_db) - v_db.min())
        data["glitch_max_adj_drop_db"][i] = float(np.abs(np.diff(v_db)).max()) if len(v_db) >= 2 else 0.0

        # Raw physics share q15 (non-clean)
        b15_raw_lin = np.power(10.0, np.clip(np.percentile(Xc[i, v_idx], 15.0, axis=0), -60.0, 60.0) / 10.0)
        E_v = E_lin[i, v_idx]
        tot_v = max(float(E_v.sum()), eps)
        data["phys_share_raw_q15"][i] = float(np.clip(np.maximum(E_v - b15_raw_lin[None, :], 0.0).sum() / tot_v, 0.0, 1.0))

        # Glitch-clean baseline shares
        X_c = Xc[i, c_idx]
        sh_map: dict[int, float] = {}
        for q in QUANTILES:
            b_db_q = np.percentile(X_c, q, axis=0)
            b_lin_q = np.power(10.0, np.clip(b_db_q, -60.0, 60.0) / 10.0)
            ex_all = np.maximum(E_v - b_lin_q[None, :], 0.0)
            sh = float(np.clip(ex_all.sum() / tot_v, 0.0, 1.0))
            sh_map[q] = sh
            data[f"phys_share_q{q}"][i] = sh
            sh_clip = float(np.clip(sh, 1e-4, 1.0 - 1e-4))
            data[f"phys_logit_q{q}"][i] = float(np.log(sh_clip / (1.0 - sh_clip)))

        data["phys_share_q10_minus_q50"][i] = sh_map[10] - sh_map[50]
        data["phys_share_q15_minus_q35"][i] = sh_map[15] - sh_map[35]
        data["phys_share_q20_ratio_q35"][i] = sh_map[20] / (sh_map[35] + 1e-4)

        # Clean broadband trajectory stats
        c_db = min_tot_db[i, c_idx]
        c_e = np.sum(E_lin[i, c_idx], axis=1)
        p05, p20, p50, p75, p25, p95 = np.percentile(c_db, [5, 20, 50, 75, 25, 95])
        data["bb_clean_range_max_min"][i] = float(c_db.max() - c_db.min())
        data["bb_clean_range_95_20"][i] = float(p95 - p20)
        data["bb_clean_upper_tail_95_50"][i] = float(p95 - p50)
        data["bb_clean_lower_tail_50_20"][i] = float(p50 - p20)
        data["bb_clean_asymmetry"][i] = float((p95 - p50) - (p50 - p20))
        data["bb_clean_std_db"][i] = float(np.std(c_db))
        data["bb_clean_iqr_db"][i] = float(p75 - p25)
        data["bb_clean_frac_gt_p20_3db"][i] = float((c_db - p20 > 3.0).mean())
        data["bb_clean_frac_gt_p20_6db"][i] = float((c_db - p20 > 6.0).mean())
        data["bb_clean_frac_gt_p20_10db"][i] = float((c_db - p20 > 10.0).mean())
        data["bb_clean_lin_max_over_p20"][i] = float(np.clip(c_e.max() / max(float(np.percentile(c_e, 20)), eps), 1.0, 1e6))

        # --- Linear-subtracted excess spectrogram E_ex using q20 clean baseline ---
        b20_db = np.percentile(X_c, 20.0, axis=0)
        b20_lin = np.power(10.0, np.clip(b20_db, -60.0, 60.0) / 10.0)
        E_c = E_lin[i, c_idx]  # (T_clean, 25)
        E_ex = np.maximum(E_c - b20_lin[None, :], 0.0)  # (T_clean, 25)

        t_bb_peak = int(c_idx[int(np.argmax(c_db))])
        t_hf_peak = int(c_idx[int(np.argmax(sb_db["hf"][i, c_idx]))])
        S_ex_tot = E_ex.sum(axis=0)
        S_ex_hfp = np.maximum(E_lin[i, t_hf_peak] - b20_lin, 0.0)

        # Fallback if S_ex_tot or S_ex_hfp is identically 0
        s_tot_sum = float(S_ex_tot.sum())
        if s_tot_sum <= 1e-24:
            S_ex_tot = np.maximum(E_c.max(axis=0) - E_c.min(axis=0), 1e-18)
            s_tot_sum = float(S_ex_tot.sum())
        if float(S_ex_hfp.sum()) <= 1e-24:
            S_ex_hfp = S_ex_tot.copy()

        ex_tot_db = 10.0 * np.log10(np.maximum(S_ex_tot / max(float(S_ex_tot.max()), 1e-30), 1e-45))
        ex_tot_db = np.clip(ex_tot_db, -45.0, 0.0)
        ex_hfp_db = 10.0 * np.log10(np.maximum(S_ex_hfp / max(float(S_ex_hfp.max()), 1e-30), 1e-45))
        ex_hfp_db = np.clip(ex_hfp_db, -45.0, 0.0)

        for b in range(25):
            data[f"ex_tot_db_b{b:02d}"][i] = float(ex_tot_db[b])
        for b in (0, 5, 9, 13, 15, 19, 21, 23):
            data[f"ex_hfp_db_b{b:02d}"][i] = float(ex_hfp_db[b])

        ex_frac = S_ex_tot / s_tot_sum
        data["ex_frac_vlf"][i] = float(ex_frac[0:6].sum())
        data["ex_frac_lf"][i] = float(ex_frac[6:13].sum())
        data["ex_frac_mf"][i] = float(ex_frac[13:19].sum())
        data["ex_frac_hf"][i] = float(ex_frac[19:25].sum())
        data["ex_slope_db_per_dec"][i] = float((ex_tot_db @ LOG10_F_C) / LOG10_F_VAR)
        data["ex_curv_mid_vs_ends"][i] = float(ex_tot_db[10:19].mean() - 0.5 * (ex_tot_db[0:5].mean() + ex_tot_db[21:25].mean()))
        data["ex_entropy"][i] = float(-np.sum(ex_frac * np.log(np.maximum(ex_frac, 1e-12))) / np.log(25.0))
        data["ex_flatness_db"][i] = float(ex_tot_db.mean() - 10.0 * np.log10(max(float(np.power(10.0, ex_tot_db / 10.0).mean()), 1e-30)))
        data["ex_crest"][i] = float(ex_frac.max() * 25.0)
        data["ex_peak_band_idx"][i] = float(np.argmax(S_ex_tot) / 24.0)

        # Targeted acoustic contrast ratios
        data["ex_631_fish_contrast"][i] = float(ex_tot_db[13] - 0.5 * (ex_tot_db[11] + ex_tot_db[15]))
        data["ex_shrimp_6309_contrast"][i] = float(ex_tot_db[23] - ex_tot_db[21])
        data["ex_2512_minus_31"][i] = float(ex_tot_db[19] - ex_tot_db[0])
        data["ex_1000_minus_63"][i] = float(ex_tot_db[15] - ex_tot_db[3])
        data["ex_3981_minus_100"][i] = float(ex_tot_db[21] - ex_tot_db[5])

        # Excess vs Background divergence
        b20_frac = b20_lin / max(float(b20_lin.sum()), eps)
        hfp_frac = S_ex_hfp / max(float(S_ex_hfp.sum()), eps)
        b20_db_c = b20_db - b20_db.mean()
        ext_db_c = ex_tot_db - ex_tot_db.mean()
        hfp_db_c = ex_hfp_db - ex_hfp_db.mean()

        data["ex_bg_cos_tot"][i] = _safe_cosine(b20_db_c, ext_db_c)
        data["ex_bg_l1_tot"][i] = float(np.sum(np.abs(ex_frac - b20_frac)))
        data["ex_bg_jsd_tot"][i] = _js_divergence(ex_frac, b20_frac)
        data["ex_bg_cos_hfp"][i] = _safe_cosine(b20_db_c, hfp_db_c)
        data["ex_bg_l1_hfp"][i] = float(np.sum(np.abs(hfp_frac - b20_frac)))
        data["ex_bg_jsd_hfp"][i] = _js_divergence(hfp_frac, b20_frac)

        # --- Rank-1 Spatiotemporal SVD & Top-3 active-minute excess consistency ---
        med_b20 = max(float(np.median(b20_lin)), 1e-12)
        M_ex = np.log1p(E_ex / med_b20)
        if M_ex.shape[0] >= 2 and float(M_ex.sum()) > 1e-12:
            s = np.linalg.svd(M_ex, compute_uv=False)
            s2 = s ** 2
            s2_sum = max(float(s2.sum()), 1e-12)
            p_s = np.maximum(s2 / s2_sum, 1e-12)
            data["svd_rank1_ratio"][i] = float(s2[0] / s2_sum)
            data["svd_rank2_ratio"][i] = float((s2[0] + (s2[1] if len(s2) > 1 else 0.0)) / s2_sum)
            data["svd_eff_rank"][i] = float(-np.sum(p_s * np.log(p_s)) / max(np.log(len(s2)), 1e-9))

            s_lin = np.linalg.svd(E_ex / max(float(E_ex.max()), 1e-12), compute_uv=False)
            s_lin2 = s_lin ** 2
            data["svd_lin_rank1_ratio"][i] = float(s_lin2[0] / max(float(s_lin2.sum()), 1e-12))
        else:
            data["svd_rank1_ratio"][i] = 1.0
            data["svd_rank2_ratio"][i] = 1.0
            data["svd_eff_rank"][i] = 0.0
            data["svd_lin_rank1_ratio"][i] = 1.0

        row_e = E_ex.sum(axis=1)
        top_k = np.argsort(row_e)[-min(3, len(row_e)) :]
        if len(top_k) >= 2:
            cos_pairs = []
            for a_i in range(len(top_k)):
                for b_i in range(a_i + 1, len(top_k)):
                    cos_pairs.append(_safe_cosine(M_ex[top_k[a_i]], M_ex[top_k[b_i]]))
            data["top3_ex_spec_cos_mean"][i] = float(np.mean(cos_pairs))
            data["top3_ex_spec_cos_min"][i] = float(np.min(cos_pairs))
        else:
            data["top3_ex_spec_cos_mean"][i] = 1.0
            data["top3_ex_spec_cos_min"][i] = 1.0

        # --- Sub-band & broadband morphology on clean minutes ---
        ser_map = {
            "bb": min_tot_db[i],
            "vlf": sb_db["vlf"][i],
            "lf": sb_db["lf"][i],
            "mf": sb_db["mf"][i],
            "hf": sb_db["hf"][i],
        }
        m_stats = {sname: _series_morphology(sarr, c_idx) for sname, sarr in ser_map.items()}
        for sname, st in m_stats.items():
            for k in ("w6", "w3", "interior", "prom", "range", "quad_r2", "concav", "lag1", "asym", "sharp"):
                data[f"m_{sname}_{k}"][i] = st[k]

        data["w6_lf_minus_hf"][i] = m_stats["lf"]["w6"] - m_stats["hf"]["w6"]
        data["w6_bb_minus_hf"][i] = m_stats["bb"]["w6"] - m_stats["hf"]["w6"]
        data["argmax_absdiff_lf_hf"][i] = abs(m_stats["lf"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_mf_hf"][i] = abs(m_stats["mf"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_bb_hf"][i] = abs(m_stats["bb"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_vlf_hf"][i] = abs(m_stats["vlf"]["argmax"] - m_stats["hf"]["argmax"])

        c_vlf = ser_map["vlf"][c_idx]
        c_lf = ser_map["lf"][c_idx]
        c_mf = ser_map["mf"][c_idx]
        c_hf = ser_map["hf"][c_idx]
        data["tcorr_vlf_lf"][i] = _safe_corrcoef_1d(c_vlf, c_lf)
        data["tcorr_lf_mf"][i] = _safe_corrcoef_1d(c_lf, c_mf)
        data["tcorr_mf_hf"][i] = _safe_corrcoef_1d(c_mf, c_hf)
        data["tcorr_lf_hf"][i] = _safe_corrcoef_1d(c_lf, c_hf)
        data["tcorr_vlf_hf"][i] = _safe_corrcoef_1d(c_vlf, c_hf)

        band_corrs = np.array([_safe_corrcoef_1d(X_c[:, b], c_db) for b in range(25)], dtype=np.float64)
        data["morph_mean_comod"][i] = float(band_corrs.mean())
        data["morph_frac_comod_065"][i] = float((band_corrs > 0.65).mean())
        data["morph_low_high_comod_diff"][i] = float(band_corrs[:10].mean() - band_corrs[15:].mean())

        # Quiet & loud context (shift invariant)
        k_sub = max(1, len(c_idx) // 5)
        ord_c = c_idx[np.argsort(c_db)]
        q_prof = Xc[i, ord_c[:k_sub]].mean(axis=0)
        l_prof = Xc[i, ord_c[-k_sub:]].mean(axis=0)
        sdiff = l_prof - q_prof

        q_hv = float(q_prof[17:25].mean() - q_prof[0:5].mean())
        l_hv = float(l_prof[17:25].mean() - l_prof[0:5].mean())
        data["quiet_d63_shrimp"][i] = float(q_prof[23] - q_prof[21])
        data["quiet_d79_nyquist"][i] = float(q_prof[24] - q_prof[22])
        data["quiet_hf_minus_vlf"][i] = q_hv
        data["quiet_mf_minus_vlf"][i] = float(q_prof[11:17].mean() - q_prof[0:5].mean())
        data["quiet_tilt_slope"][i] = float((q_prof @ LOG10_F_C) / LOG10_F_VAR)
        data["loud_hf_minus_vlf"][i] = l_hv
        data["delta_hf_minus_vlf"][i] = l_hv - q_hv
        data["active_quiet_tilt_change"][i] = float((sdiff @ LOG10_F_C) / LOG10_F_VAR)
        data["active_quiet_shape_cosine"][i] = _safe_cosine(q_prof - q_prof.mean(), l_prof - l_prof.mean())

        # Sub-band shares & dynamics (q20 clean)
        for sb_name, sl in SUB_BANDS.items():
            b_sb_lin = b20_lin[sl]
            e_sb_v = E_lin[i, v_idx][:, sl]
            ex_sb = np.maximum(e_sb_v - b_sb_lin[None, :], 0.0)
            data[f"{sb_name}_share_q20"][i] = float(ex_sb.sum() / max(float(e_sb_v.sum()), eps))
            data[f"{sb_name}_contrib_q20"][i] = float(ex_sb.sum() / tot_v)
            sdb_c = sb_db[sb_name][i, c_idx]
            data[f"{sb_name}_range_95_20"][i] = float(np.percentile(sdb_c, 95) - np.percentile(sdb_c, 20))
            data[f"{sb_name}_std_db"][i] = float(np.std(sdb_c))

        # Deliberately ONLY overall obs fraction — omit mask_m01_missing to prevent station shortcuts!
        data["mask_obs_frac"][i] = float(min_obs[i].mean())

    df = pd.DataFrame(data)
    assert np.isfinite(df.to_numpy()).all(), "Non-finite tabular features detected"
    return df


def build_cnn_tensors_v3(X_db: np.ndarray) -> np.ndarray:
    """Build shift-invariant 7-channel tensor (N, 7, 15, 25) float32 for AcousticCNN2D_v3."""
    Xc, obs, min_obs, clean_min, E_lin, _ = _prepare_clean_masks(X_db)
    n = Xc.shape[0]

    # Glitch-clean per-band q20 baseline
    base_q20_db = np.empty((n, 1, 25), dtype=np.float64)
    for i in range(n):
        c_idx = np.flatnonzero(clean_min[i])
        base_q20_db[i, 0] = np.percentile(Xc[i, c_idx], 20.0, axis=0)
    b20_lin = np.power(10.0, np.clip(base_q20_db, -60.0, 60.0) / 10.0)

    obs_sum = np.where(obs, Xc, 0.0).sum(axis=2, keepdims=True)
    obs_cnt = np.maximum(obs.sum(axis=2, keepdims=True), 1)
    min_mean = obs_sum / obs_cnt

    # ch0: centered relative level
    ch0 = np.where(obs, np.clip(Xc / 15.0, -4.0, 4.0), 0.0)
    # ch1: glitch-robust dB excess above q20 band baseline
    ch1 = np.where(obs, np.clip((Xc - base_q20_db) / 15.0, -4.0, 4.0), 0.0)
    # ch2: within-minute spectral shape
    ch2 = np.where(obs, np.clip((Xc - min_mean) / 15.0, -4.0, 4.0), 0.0)
    # ch3: linear-subtracted excess energy map normalized by baseline
    E_ex = np.where(obs, np.maximum(E_lin - b20_lin, 0.0), 0.0)
    ch3 = np.where(obs, np.clip(np.log10(1.0 + E_ex / np.maximum(b20_lin, 1e-12)), 0.0, 4.0) / 2.0, 0.0)

    # ch4: temporal derivative across adjacent observed minutes (0 at NaN boundaries)
    Xc_clean = np.where(obs, Xc, 0.0)
    d_fwd = np.zeros_like(Xc)
    d_bwd = np.zeros_like(Xc)
    obs_fwd = np.zeros_like(obs)
    obs_bwd = np.zeros_like(obs)
    d_fwd[:, :-1, :] = Xc_clean[:, 1:, :] - Xc_clean[:, :-1, :]
    obs_fwd[:, :-1, :] = obs[:, 1:, :] & obs[:, :-1, :]
    d_bwd[:, 1:, :] = Xc_clean[:, 1:, :] - Xc_clean[:, :-1, :]
    obs_bwd[:, 1:, :] = obs[:, 1:, :] & obs[:, :-1, :]
    cnt_diff = obs_fwd.astype(np.float64) + obs_bwd.astype(np.float64)
    deriv = np.where(cnt_diff > 0, (np.where(obs_fwd, d_fwd, 0.0) + np.where(obs_bwd, d_bwd, 0.0)) / np.maximum(cnt_diff, 1.0), 0.0)
    ch4 = np.where(obs, np.clip(deriv / 15.0, -4.0, 4.0), 0.0)

    # ch5: normalized log-frequency coordinate broadcast to (N, 15, 25)
    ch5 = np.where(obs, np.broadcast_to(FREQ_COORD[None, None, :], Xc.shape), 0.0)
    # ch6: observation mask (binary)
    ch6 = obs.astype(np.float64)

    tensors = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6], axis=1).astype(np.float32)
    assert np.isfinite(tensors).all(), "Non-finite CNN tensors detected"
    return tensors


def run_self_test() -> None:
    """Verify <1e-10 tabular invariance and <1e-5 7-channel CNN invariance under NaNs & -40 dB glitches."""
    rng = np.random.default_rng(123)
    X = rng.normal(90.0, 6.0, size=(24, 15, 25))
    # Add missing minutes & synthetic -40 dB downward glitches
    X[:8, 0, :] = np.nan
    X[8:14, 14, :] = np.nan
    X[14:16, 7, :] = np.nan
    X[:10, 5, :] -= 40.0
    X_shifted = X + 37.5

    f1 = extract_features_v3(X)
    f2 = extract_features_v3(X_shifted)
    max_diff = float(np.max(np.abs(f1.to_numpy() - f2.to_numpy())))
    assert max_diff < 1e-10, f"Tabular invariance violated: max_diff={max_diff}"
    assert "mask_m01_missing" not in f1.columns

    t1 = build_cnn_tensors_v3(X)
    t2 = build_cnn_tensors_v3(X_shifted)
    assert t1.shape == (24, 7, 15, 25)
    t_diff = float(np.max(np.abs(t1 - t2)))
    assert t_diff < 1e-5, f"CNN tensor invariance violated: max_diff={t_diff}"

    s1, p1 = compute_robust_physics_share(X, q=20.0)
    s2, p2 = compute_robust_physics_share(X_shifted, q=20.0)
    assert float(np.max(np.abs(s1 - s2))) < 1e-12
    assert float(np.max(np.abs(p1 - p2))) < 1e-12
    print(f"modeling_v3_features self-test: PASS ({f1.shape[1]} features, 7 CNN channels, max_tab_diff={max_diff:.2e}, max_cnn_diff={t_diff:.2e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_only:
        run_self_test()


if __name__ == "__main__":
    main()
