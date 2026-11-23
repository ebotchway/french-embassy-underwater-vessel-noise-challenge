#!/usr/bin/env python3
"""Hydrophone-calibration-invariant R(t,f), sharp-step, and 7-channel CNN tensor extractors (v4)."""
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
from modeling_v3_features import (  # noqa: E402
    FREQ_COORD,
    LOG10_F_C,
    LOG10_F_VAR,
    QUANTILES,
    SUB_BANDS,
    _js_divergence,
    _prepare_clean_masks,
    _safe_corrcoef_1d,
    _safe_cosine,
    _series_morphology,
    compute_robust_physics_share,
)


def _sharp_step_metrics(series: np.ndarray, idx_clean: np.ndarray) -> dict[str, float]:
    """Compute step-jump, boxcar, second-difference, and local-peak features on a clean 1D series."""
    v = series[idx_clean]
    if len(v) < 3:
        return {
            "max_abs_diff": 0.0,
            "step_ratio": 0.0,
            "n_up_8db": 0.0,
            "n_down_8db": 0.0,
            "max_2nd_diff": 0.0,
            "n_peaks": 0.0,
            "boxcar_ratio": 0.0,
            "plateau3": float(len(v)),
        }
    rng = max(float(v.max() - v.min()), 1.0)
    d1 = np.diff(v)
    max_abs = float(np.abs(d1).max())
    d2 = np.diff(v, n=2)
    max_d2 = float(np.abs(d2).max()) if len(d2) > 0 else 0.0
    n_up = float((d1 >= 8.0).sum())
    n_dn = float((d1 <= -8.0).sum())
    n_peaks = float(((v[1:-1] > v[:-2] + 3.0) & (v[1:-1] > v[2:] + 3.0)).sum())
    vmax = float(v.max())
    med = float(np.median(v))
    boxcar = float((vmax - med) / rng)
    plateau3 = float((v >= vmax - 3.0).sum())
    return {
        "max_abs_diff": max_abs,
        "step_ratio": max_abs / rng,
        "n_up_8db": n_up,
        "n_down_8db": n_dn,
        "max_2nd_diff": max_d2,
        "n_peaks": n_peaks,
        "boxcar_ratio": boxcar,
        "plateau3": plateau3,
    }


def extract_features_v4(X_db: np.ndarray) -> pd.DataFrame:
    """Extract h(f)-invariant R(t,f) features, sharp-step stats, and robust energy shares."""
    X = np.asarray(X_db, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (15, 25):
        raise ValueError(f"Expected shape (N, 15, 25), got {X.shape}")

    n = X.shape[0]
    Xc, obs, min_obs, clean_min, E_lin, min_tot_db = _prepare_clean_masks(X)
    eps = 1e-30

    sb_db: dict[str, np.ndarray] = {}
    for sb_name, sl in SUB_BANDS.items():
        se = np.where(min_obs, np.nansum(E_lin[:, :, sl], axis=2), np.nan)
        sdb = 10.0 * np.log10(np.maximum(se, eps))
        sdb[~min_obs] = np.nan
        sb_db[sb_name] = sdb

    # Pre-declare column names cleanly
    cols: list[str] = (
        # Group A: 100% h(f)-invariant R_* block (cancels any additive h(f) identically!)
        [f"R_max_b{b:02d}" for b in range(25)]
        + ["R_p90_vlf", "R_p90_lf", "R_p90_mf", "R_p90_hf"]
        + [
            "R_slope_db_per_dec", "R_curv_mid_vs_ends", "R_hf_minus_vlf",
            "R_mf_minus_vlf", "R_hf_minus_mf", "R_std_bands", "R_mean_bands",
            "R_max_val", "R_max_band_idx", "R_frac_gt_6db", "R_frac_gt_12db",
            "R_631_fish_contrast", "R_shrimp_6309_contrast",
            "R_svd_rank1", "R_svd_rank2", "R_svd_eff_rank",
            "R_top3_cos_mean", "R_top3_cos_min",
        ]
    )
    for ser in ("R_bar", "R_hf", "R_mf"):
        for k in ("max_abs_diff", "step_ratio", "n_up_8db", "n_down_8db", "max_2nd_diff", "n_peaks", "boxcar_ratio", "plateau3"):
            cols.append(f"R_{ser.split('_')[1]}_{k}")

    # Group B: Glitch-robust energy shares & excess divergence (for share magnitude & balance)
    cols += (
        ["glitch_n_drop_min", "glitch_med_minus_min_db", "glitch_max_adj_drop_db", "phys_share_raw_q15"]
        + [f"phys_share_q{q}" for q in QUANTILES]
        + [f"phys_logit_q{q}" for q in QUANTILES]
        + ["phys_share_q10_minus_q50", "phys_share_q15_minus_q35", "phys_share_q20_ratio_q35"]
        + [
            "bb_clean_range_max_min", "bb_clean_range_95_20", "bb_clean_upper_tail_95_50",
            "bb_clean_lower_tail_50_20", "bb_clean_asymmetry", "bb_clean_std_db",
            "bb_clean_iqr_db", "bb_clean_frac_gt_p20_3db", "bb_clean_frac_gt_p20_6db",
            "bb_clean_frac_gt_p20_10db", "bb_clean_lin_max_over_p20",
            "ex_frac_vlf", "ex_frac_lf", "ex_frac_mf", "ex_frac_hf",
            "ex_curv_mid_vs_ends", "ex_entropy", "ex_crest",
            "ex_bg_cos_tot", "ex_bg_l1_tot", "ex_bg_jsd_tot",
            "ex_bg_cos_hfp", "ex_bg_jsd_hfp",
            "svd_rank1_ratio", "svd_lin_rank1_ratio",
        ]
    )
    for sb in ("vlf", "lf", "mf", "hf"):
        cols += [f"{sb}_share_q20", f"{sb}_contrib_q20", f"{sb}_range_95_20", f"{sb}_std_db"]

    # Group C: Frequency-dependent CPA morphology & cross-band alignment
    for ser in ("bb", "vlf", "lf", "mf", "hf"):
        for k in ("w6", "w3", "interior", "prom", "range", "quad_r2", "concav", "lag1", "asym", "sharp"):
            cols.append(f"m_{ser}_{k}")
    cols += [
        "m_hf_max_abs_diff", "m_hf_step_ratio", "m_bb_max_abs_diff", "m_bb_step_ratio",
        "w6_lf_minus_hf", "w6_bb_minus_hf",
        "argmax_absdiff_lf_hf", "argmax_absdiff_mf_hf", "argmax_absdiff_bb_hf", "argmax_absdiff_vlf_hf",
        "tcorr_vlf_lf", "tcorr_lf_mf", "tcorr_mf_hf", "tcorr_lf_hf", "tcorr_vlf_hf",
        "morph_mean_comod", "morph_frac_comod_065", "morph_low_high_comod_diff",
        "quiet_d63_shrimp", "quiet_d79_nyquist", "delta_hf_minus_vlf",
        "active_quiet_tilt_change", "active_quiet_shape_cosine", "mask_obs_frac",
    ]

    data = {c: np.zeros(n, dtype=np.float64) for c in cols}

    for i in range(n):
        v_idx = np.flatnonzero(min_obs[i])
        c_idx = np.flatnonzero(clean_min[i])
        X_c = Xc[i, c_idx]
        E_v = E_lin[i, v_idx]
        E_c = E_lin[i, c_idx]
        tot_v = max(float(E_v.sum()), eps)

        # Clean q20 baseline & h(f)-invariant relative rise R(t, f) = Xc(t, f) - b20_db(f)
        b20_db = np.percentile(X_c, 20.0, axis=0)
        b20_lin = np.power(10.0, np.clip(b20_db, -60.0, 60.0) / 10.0)
        R_c = X_c - b20_db[None, :]  # (T_clean, 25), exact h(f)-invariant!
        R_pos = np.maximum(R_c, 0.0)

        R_max = np.clip(R_pos.max(axis=0), 0.0, 50.0)
        R_p90 = np.clip(np.percentile(R_pos, 90.0, axis=0), 0.0, 50.0)
        for b in range(25):
            data[f"R_max_b{b:02d}"][i] = float(R_max[b])

        data["R_p90_vlf"][i] = float(R_p90[0:6].mean())
        data["R_p90_lf"][i] = float(R_p90[6:13].mean())
        data["R_p90_mf"][i] = float(R_p90[13:19].mean())
        data["R_p90_hf"][i] = float(R_p90[19:25].mean())

        data["R_slope_db_per_dec"][i] = float((R_max @ LOG10_F_C) / LOG10_F_VAR)
        data["R_curv_mid_vs_ends"][i] = float(R_max[10:19].mean() - 0.5 * (R_max[0:5].mean() + R_max[21:25].mean()))
        data["R_hf_minus_vlf"][i] = float(R_max[19:25].mean() - R_max[0:6].mean())
        data["R_mf_minus_vlf"][i] = float(R_max[13:19].mean() - R_max[0:6].mean())
        data["R_hf_minus_mf"][i] = float(R_max[19:25].mean() - R_max[13:19].mean())
        data["R_std_bands"][i] = float(R_max.std())
        data["R_mean_bands"][i] = float(R_max.mean())
        data["R_max_val"][i] = float(R_max.max())
        data["R_max_band_idx"][i] = float(np.argmax(R_max) / 24.0)
        data["R_frac_gt_6db"][i] = float((R_max > 6.0).mean())
        data["R_frac_gt_12db"][i] = float((R_max > 12.0).mean())
        data["R_631_fish_contrast"][i] = float(R_max[13] - 0.5 * (R_max[11] + R_max[15]))
        data["R_shrimp_6309_contrast"][i] = float(R_max[23] - R_max[21])

        # SVD on h(f)-invariant R_pos
        if R_pos.shape[0] >= 2 and float(R_pos.sum()) > 1e-12:
            s_r = np.linalg.svd(R_pos, compute_uv=False)
            s2_r = s_r ** 2
            s2_r_sum = max(float(s2_r.sum()), 1e-12)
            pr = np.maximum(s2_r / s2_r_sum, 1e-12)
            data["R_svd_rank1"][i] = float(s2_r[0] / s2_r_sum)
            data["R_svd_rank2"][i] = float((s2_r[0] + (s2_r[1] if len(s2_r) > 1 else 0.0)) / s2_r_sum)
            data["R_svd_eff_rank"][i] = float(-np.sum(pr * np.log(pr)) / max(np.log(len(s2_r)), 1e-9))
        else:
            data["R_svd_rank1"][i] = 1.0
            data["R_svd_rank2"][i] = 1.0
            data["R_svd_eff_rank"][i] = 0.0

        row_r = R_pos.sum(axis=1)
        top_r = np.argsort(row_r)[-min(3, len(row_r)) :]
        if len(top_r) >= 2:
            c_pairs = [_safe_cosine(R_pos[top_r[a]], R_pos[top_r[b]]) for a in range(len(top_r)) for b in range(a + 1, len(top_r))]
            data["R_top3_cos_mean"][i] = float(np.mean(c_pairs))
            data["R_top3_cos_min"][i] = float(np.min(c_pairs))
        else:
            data["R_top3_cos_mean"][i] = 1.0
            data["R_top3_cos_min"][i] = 1.0

        # h(f)-invariant sharp-step/boxcar stats on R_bar, R_hf, R_mf
        R_full = Xc[i] - b20_db[None, :]
        r_series_map = {
            "bar": R_full.mean(axis=1),
            "hf": R_full[:, 19:25].mean(axis=1),
            "mf": R_full[:, 13:19].mean(axis=1),
        }
        for s_key, s_arr in r_series_map.items():
            st = _sharp_step_metrics(s_arr, c_idx)
            for k, val in st.items():
                data[f"R_{s_key}_{k}"][i] = val

        # --- Group B: Glitch telemetry & physics energy shares ---
        v_db = min_tot_db[i, v_idx]
        data["glitch_n_drop_min"][i] = float(len(v_idx) - len(c_idx))
        data["glitch_med_minus_min_db"][i] = float(np.median(v_db) - v_db.min())
        data["glitch_max_adj_drop_db"][i] = float(np.abs(np.diff(v_db)).max()) if len(v_db) >= 2 else 0.0

        b15_raw_lin = np.power(10.0, np.clip(np.percentile(Xc[i, v_idx], 15.0, axis=0), -60.0, 60.0) / 10.0)
        data["phys_share_raw_q15"][i] = float(np.clip(np.maximum(E_v - b15_raw_lin[None, :], 0.0).sum() / tot_v, 0.0, 1.0))

        sh_map: dict[int, float] = {}
        for q in QUANTILES:
            b_lin_q = np.power(10.0, np.clip(np.percentile(X_c, q, axis=0), -60.0, 60.0) / 10.0)
            sh = float(np.clip(np.maximum(E_v - b_lin_q[None, :], 0.0).sum() / tot_v, 0.0, 1.0))
            sh_map[q] = sh
            data[f"phys_share_q{q}"][i] = sh
            sh_c = float(np.clip(sh, 1e-4, 1.0 - 1e-4))
            data[f"phys_logit_q{q}"][i] = float(np.log(sh_c / (1.0 - sh_c)))

        data["phys_share_q10_minus_q50"][i] = sh_map[10] - sh_map[50]
        data["phys_share_q15_minus_q35"][i] = sh_map[15] - sh_map[35]
        data["phys_share_q20_ratio_q35"][i] = sh_map[20] / (sh_map[35] + 1e-4)

        c_db = min_tot_db[i, c_idx]
        c_e = np.sum(E_c, axis=1)
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

        E_ex = np.maximum(E_c - b20_lin[None, :], 0.0)
        S_ex_tot = E_ex.sum(axis=0)
        if float(S_ex_tot.sum()) <= 1e-24:
            S_ex_tot = np.maximum(E_c.max(axis=0) - E_c.min(axis=0), 1e-18)
        s_sum = float(S_ex_tot.sum())
        ex_frac = S_ex_tot / s_sum
        ex_tot_db = np.clip(10.0 * np.log10(np.maximum(S_ex_tot / max(float(S_ex_tot.max()), eps), 1e-45)), -45.0, 0.0)

        t_hfp = int(c_idx[int(np.argmax(sb_db["hf"][i, c_idx]))])
        S_ex_hfp = np.maximum(E_lin[i, t_hfp] - b20_lin, 0.0)
        if float(S_ex_hfp.sum()) <= 1e-24:
            S_ex_hfp = S_ex_tot.copy()
        hfp_frac = S_ex_hfp / max(float(S_ex_hfp.sum()), eps)
        ex_hfp_db = np.clip(10.0 * np.log10(np.maximum(S_ex_hfp / max(float(S_ex_hfp.max()), eps), 1e-45)), -45.0, 0.0)

        data["ex_frac_vlf"][i] = float(ex_frac[0:6].sum())
        data["ex_frac_lf"][i] = float(ex_frac[6:13].sum())
        data["ex_frac_mf"][i] = float(ex_frac[13:19].sum())
        data["ex_frac_hf"][i] = float(ex_frac[19:25].sum())
        data["ex_curv_mid_vs_ends"][i] = float(ex_tot_db[10:19].mean() - 0.5 * (ex_tot_db[0:5].mean() + ex_tot_db[21:25].mean()))
        data["ex_entropy"][i] = float(-np.sum(ex_frac * np.log(np.maximum(ex_frac, 1e-12))) / np.log(25.0))
        data["ex_crest"][i] = float(ex_frac.max() * 25.0)

        b20_frac = b20_lin / max(float(b20_lin.sum()), eps)
        data["ex_bg_cos_tot"][i] = _safe_cosine(b20_db - b20_db.mean(), ex_tot_db - ex_tot_db.mean())
        data["ex_bg_l1_tot"][i] = float(np.sum(np.abs(ex_frac - b20_frac)))
        data["ex_bg_jsd_tot"][i] = _js_divergence(ex_frac, b20_frac)
        data["ex_bg_cos_hfp"][i] = _safe_cosine(b20_db - b20_db.mean(), ex_hfp_db - ex_hfp_db.mean())
        data["ex_bg_jsd_hfp"][i] = _js_divergence(hfp_frac, b20_frac)

        med_b20 = max(float(np.median(b20_lin)), 1e-12)
        M_ex = np.log1p(E_ex / med_b20)
        if M_ex.shape[0] >= 2 and float(M_ex.sum()) > 1e-12:
            s = np.linalg.svd(M_ex, compute_uv=False)
            s2 = s ** 2
            data["svd_rank1_ratio"][i] = float(s2[0] / max(float(s2.sum()), 1e-12))
            s_lin = np.linalg.svd(E_ex / max(float(E_ex.max()), 1e-12), compute_uv=False)
            s2_l = s_lin ** 2
            data["svd_lin_rank1_ratio"][i] = float(s2_l[0] / max(float(s2_l.sum()), 1e-12))
        else:
            data["svd_rank1_ratio"][i] = 1.0
            data["svd_lin_rank1_ratio"][i] = 1.0

        for sb_name, sl in SUB_BANDS.items():
            b_sb_lin = b20_lin[sl]
            e_sb_v = E_v[:, sl]
            ex_sb = np.maximum(e_sb_v - b_sb_lin[None, :], 0.0)
            data[f"{sb_name}_share_q20"][i] = float(ex_sb.sum() / max(float(e_sb_v.sum()), eps))
            data[f"{sb_name}_contrib_q20"][i] = float(ex_sb.sum() / tot_v)
            sdb_c = sb_db[sb_name][i, c_idx]
            data[f"{sb_name}_range_95_20"][i] = float(np.percentile(sdb_c, 95) - np.percentile(sdb_c, 20))
            data[f"{sb_name}_std_db"][i] = float(np.std(sdb_c))

        # --- Group C: Subband morphology, sharp steps, & alignment ---
        ser_map = {"bb": min_tot_db[i], "vlf": sb_db["vlf"][i], "lf": sb_db["lf"][i], "mf": sb_db["mf"][i], "hf": sb_db["hf"][i]}
        m_stats = {sname: _series_morphology(sarr, c_idx) for sname, sarr in ser_map.items()}
        for sname, st in m_stats.items():
            for k in ("w6", "w3", "interior", "prom", "range", "quad_r2", "concav", "lag1", "asym", "sharp"):
                data[f"m_{sname}_{k}"][i] = st[k]

        hf_step = _sharp_step_metrics(sb_db["hf"][i], c_idx)
        bb_step = _sharp_step_metrics(min_tot_db[i], c_idx)
        data["m_hf_max_abs_diff"][i] = hf_step["max_abs_diff"]
        data["m_hf_step_ratio"][i] = hf_step["step_ratio"]
        data["m_bb_max_abs_diff"][i] = bb_step["max_abs_diff"]
        data["m_bb_step_ratio"][i] = bb_step["step_ratio"]

        data["w6_lf_minus_hf"][i] = m_stats["lf"]["w6"] - m_stats["hf"]["w6"]
        data["w6_bb_minus_hf"][i] = m_stats["bb"]["w6"] - m_stats["hf"]["w6"]
        data["argmax_absdiff_lf_hf"][i] = abs(m_stats["lf"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_mf_hf"][i] = abs(m_stats["mf"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_bb_hf"][i] = abs(m_stats["bb"]["argmax"] - m_stats["hf"]["argmax"])
        data["argmax_absdiff_vlf_hf"][i] = abs(m_stats["vlf"]["argmax"] - m_stats["hf"]["argmax"])

        c_vlf, c_lf, c_mf, c_hf = ser_map["vlf"][c_idx], ser_map["lf"][c_idx], ser_map["mf"][c_idx], ser_map["hf"][c_idx]
        data["tcorr_vlf_lf"][i] = _safe_corrcoef_1d(c_vlf, c_lf)
        data["tcorr_lf_mf"][i] = _safe_corrcoef_1d(c_lf, c_mf)
        data["tcorr_mf_hf"][i] = _safe_corrcoef_1d(c_mf, c_hf)
        data["tcorr_lf_hf"][i] = _safe_corrcoef_1d(c_lf, c_hf)
        data["tcorr_vlf_hf"][i] = _safe_corrcoef_1d(c_vlf, c_hf)

        band_corrs = np.array([_safe_corrcoef_1d(X_c[:, b], c_db) for b in range(25)], dtype=np.float64)
        data["morph_mean_comod"][i] = float(band_corrs.mean())
        data["morph_frac_comod_065"][i] = float((band_corrs > 0.65).mean())
        data["morph_low_high_comod_diff"][i] = float(band_corrs[:10].mean() - band_corrs[15:].mean())

        k_sub = max(1, len(c_idx) // 5)
        ord_c = c_idx[np.argsort(c_db)]
        q_prof = Xc[i, ord_c[:k_sub]].mean(axis=0)
        l_prof = Xc[i, ord_c[-k_sub:]].mean(axis=0)
        sdiff = l_prof - q_prof
        q_hv = float(q_prof[17:25].mean() - q_prof[0:5].mean())
        l_hv = float(l_prof[17:25].mean() - l_prof[0:5].mean())
        data["quiet_d63_shrimp"][i] = float(q_prof[23] - q_prof[21])
        data["quiet_d79_nyquist"][i] = float(q_prof[24] - q_prof[22])
        data["delta_hf_minus_vlf"][i] = l_hv - q_hv
        data["active_quiet_tilt_change"][i] = float((sdiff @ LOG10_F_C) / LOG10_F_VAR)
        data["active_quiet_shape_cosine"][i] = _safe_cosine(q_prof - q_prof.mean(), l_prof - l_prof.mean())
        data["mask_obs_frac"][i] = float(min_obs[i].mean())

    df = pd.DataFrame(data)
    assert np.isfinite(df.to_numpy()).all(), "Non-finite tabular features detected"
    return df


def build_cnn_tensors_v4(X_db: np.ndarray) -> np.ndarray:
    """Build 7-channel h(f)-invariant-first tensor (N, 7, 15, 25) float32 for CNN2D_v4."""
    Xc, obs, _, clean_min, _, _ = _prepare_clean_masks(X_db)
    n = Xc.shape[0]

    base_q20_db = np.empty((n, 1, 25), dtype=np.float64)
    for i in range(n):
        c_idx = np.flatnonzero(clean_min[i])
        base_q20_db[i, 0] = np.percentile(Xc[i, c_idx], 20.0, axis=0)

    # R(t, f) = Xc(t, f) - base_q20_db(f) -> strictly h(f)-invariant!
    R = np.where(obs, Xc - base_q20_db, 0.0)
    R_cnt = np.maximum(obs.sum(axis=2, keepdims=True), 1)
    R_min_mean = R.sum(axis=2, keepdims=True) / R_cnt

    # ch0: relative rise dB R/15
    ch0 = np.where(obs, np.clip(R / 15.0, -4.0, 4.0), 0.0)
    # ch1: relative excess SNR ratio log10(1 + max(10^(R/10)-1, 0))/2
    snr_ex = np.maximum(np.power(10.0, np.clip(R, -40.0, 40.0) / 10.0) - 1.0, 0.0)
    ch1 = np.where(obs, np.clip(np.log10(1.0 + snr_ex), 0.0, 4.0) / 2.0, 0.0)
    # ch2: spectral shape of relative rise within minute
    ch2 = np.where(obs, np.clip((R - R_min_mean) / 15.0, -4.0, 4.0), 0.0)

    # ch3: centered temporal derivative of R across adjacent observed minutes
    d_fwd = np.zeros_like(R)
    d_bwd = np.zeros_like(R)
    obs_fwd = np.zeros_like(obs)
    obs_bwd = np.zeros_like(obs)
    d_fwd[:, :-1, :] = R[:, 1:, :] - R[:, :-1, :]
    obs_fwd[:, :-1, :] = obs[:, 1:, :] & obs[:, :-1, :]
    d_bwd[:, 1:, :] = R[:, 1:, :] - R[:, :-1, :]
    obs_bwd[:, 1:, :] = obs[:, 1:, :] & obs[:, :-1, :]
    cnt_diff = obs_fwd.astype(np.float64) + obs_bwd.astype(np.float64)
    deriv = np.where(cnt_diff > 0, (np.where(obs_fwd, d_fwd, 0.0) + np.where(obs_bwd, d_bwd, 0.0)) / np.maximum(cnt_diff, 1.0), 0.0)
    ch3 = np.where(obs, np.clip(deriv / 15.0, -4.0, 4.0), 0.0)

    # ch4: Xc within-minute spectral shape (carries total band weighting for share magnitude)
    Xc_clean = np.where(obs, Xc, 0.0)
    xc_min_mean = Xc_clean.sum(axis=2, keepdims=True) / R_cnt
    ch4 = np.where(obs, np.clip((Xc_clean - xc_min_mean) / 15.0, -4.0, 4.0), 0.0)

    # ch5: normalized log-frequency coordinate
    ch5 = np.where(obs, np.broadcast_to(FREQ_COORD[None, None, :], Xc.shape), 0.0)
    # ch6: observation mask
    ch6 = obs.astype(np.float64)

    tensors = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6], axis=1).astype(np.float32)
    assert np.isfinite(tensors).all(), "Non-finite CNN tensors detected"
    return tensors


def run_self_test() -> None:
    """Verify <1e-10 global shift AND exact h(f) calibration invariance across R_* & CNN channels."""
    rng = np.random.default_rng(123)
    X = rng.normal(90.0, 6.0, size=(24, 15, 25))
    X[:8, 0, :] = np.nan
    X[8:14, 14, :] = np.nan
    X[14:16, 7, :] = np.nan
    X[:10, 5, :] -= 40.0

    # 1. Global shift invariance (+37.5 dB)
    X_shift = X + 37.5
    f1 = extract_features_v4(X)
    f2 = extract_features_v4(X_shift)
    global_tab_diff = float(np.max(np.abs(f1.to_numpy() - f2.to_numpy())))
    assert global_tab_diff < 1e-10, f"Global tabular shift invariance violated: {global_tab_diff}"
    assert "mask_m01_missing" not in f1.columns

    t1 = build_cnn_tensors_v4(X)
    t2 = build_cnn_tensors_v4(X_shift)
    global_cnn_diff = float(np.max(np.abs(t1 - t2)))
    assert global_cnn_diff < 1e-5, f"Global CNN shift invariance violated: {global_cnn_diff}"

    # 2. Exact h(f) frequency-response invariance (+6 dB/dec tilt + cosine ripple)
    norm_b = np.linspace(0.0, 1.0, 25)
    h_db = 6.0 * (np.log10(EXACT_CENTRES / 500.0)) + 2.5 * np.cos(2.0 * np.pi * norm_b + 0.7)
    X_cal = X + h_db[None, None, :]
    f_cal = extract_features_v4(X_cal)
    r_cols = [c for c in f1.columns if c.startswith("R_")]
    assert len(r_cols) >= 65
    hf_r_diff = float(np.max(np.abs(f1[r_cols].to_numpy() - f_cal[r_cols].to_numpy())))
    assert hf_r_diff < 1e-10, f"h(f) R_* invariance violated: {hf_r_diff}"

    t_cal = build_cnn_tensors_v4(X_cal)
    inv_ch = [0, 1, 2, 3, 5, 6]
    hf_cnn_diff = float(np.max(np.abs(t1[:, inv_ch] - t_cal[:, inv_ch])))
    assert hf_cnn_diff < 1e-5, f"h(f) CNN channel invariance violated: {hf_cnn_diff}"

    print(
        f"modeling_v4_features self-test: PASS ({f1.shape[1]} features [{len(r_cols)} R_* h(f)-inv], "
        f"global_diff={global_tab_diff:.2e}, h_f_R_diff={hf_r_diff:.2e}, h_f_cnn_diff={hf_cnn_diff:.2e})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_only:
        run_self_test()


if __name__ == "__main__":
    main()
