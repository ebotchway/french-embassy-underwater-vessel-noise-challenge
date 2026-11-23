#!/usr/bin/env python3
"""Mask-aware, shift-invariant feature and CNN-tensor extractor for 15x25 dB matrices."""
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

QUANTILES = (5, 10, 15, 20, 25, 35, 50)
SUB_BANDS = {
    "vlf": slice(0, 6),    # 31.6 - 100 Hz
    "lf": slice(6, 13),    # 125.9 - 501.2 Hz
    "mf": slice(13, 19),   # 631 - 1995.3 Hz
    "hf": slice(19, 25),   # 2511.9 - 7943.3 Hz
}
LOG10_F_C = np.log10(EXACT_CENTRES) - np.log10(EXACT_CENTRES).mean()
LOG10_F_VAR = float(np.sum(LOG10_F_C ** 2))


def compute_physics_baseline_share(X_db: np.ndarray, q: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute shift-invariant multi-quantile excess energy share and peak-excess score."""
    X = np.asarray(X_db, dtype=np.float64)
    obs = np.isfinite(X)
    win_med = np.nanmedian(X, axis=(1, 2), keepdims=True)
    Xc = X - win_med
    E_lin = np.where(obs, np.power(10.0, np.clip(Xc, -60.0, 60.0) / 10.0), np.nan)

    base_db = np.nanpercentile(Xc, q, axis=1)
    base_lin = np.power(10.0, np.clip(base_db, -60.0, 60.0) / 10.0)
    excess = np.where(obs, np.maximum(E_lin - base_lin[:, None, :], 0.0), 0.0)
    total_e = np.nansum(E_lin, axis=(1, 2))
    share = np.clip(excess.sum(axis=(1, 2)) / np.maximum(total_e, 1e-30), 0.0, 1.0)

    # Broadband minute peak excess over baseline quantile as uncalibrated detection score
    min_obs = obs.all(axis=2)
    min_tot_e = np.where(min_obs, np.nansum(E_lin, axis=2), np.nan)
    min_tot_db = 10.0 * np.log10(np.maximum(min_tot_e, 1e-30))
    min_tot_db[~min_obs] = np.nan
    delta_max = np.nanpercentile(min_tot_db, 95, axis=1) - np.nanpercentile(min_tot_db, q, axis=1)
    score = 1.0 / (1.0 + np.exp(-0.55 * (delta_max - 2.2)))
    return share.astype(np.float64), np.clip(score, 0.0, 1.0).astype(np.float64)


def _safe_corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a_c * b_c) / denom)


def _extract_single_window_morphology(Xc_win: np.ndarray, E_win: np.ndarray, min_obs: np.ndarray) -> dict[str, float]:
    """Temporal CPA bell, lag-1 autocorrelation, and cross-band co-modulation scalars."""
    idx_valid = np.flatnonzero(min_obs)
    L_all = 10.0 * np.log10(np.maximum(np.nansum(E_win[idx_valid], axis=1), 1e-30))
    L_c = L_all - np.median(L_all)

    # 3-point smoothed envelope to reject single-minute transient glitches
    if len(L_c) >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        padded = np.pad(L_c, 1, mode="edge")
        L_sm = np.convolve(padded, kernel, mode="valid")
        smooth_range = float(L_sm.max() - L_sm.min())
    else:
        smooth_range = float(L_c.max() - L_c.min())

    # Concave quadratic fit L(t) ~ a*t^2 + b*t + c over centered normalized time
    t_norm = (idx_valid.astype(np.float64) - 7.0) / 7.0
    if len(t_norm) >= 4:
        coeffs = np.polyfit(t_norm, L_c, deg=2)
        pred = np.polyval(coeffs, t_norm)
        ss_res = float(np.sum((L_c - pred) ** 2))
        ss_tot = float(np.sum((L_c - L_c.mean()) ** 2))
        quad_r2 = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 1e-9 else 0.0
        concavity = float(-coeffs[0])
    else:
        quad_r2, concavity = 0.0, 0.0

    # CPA peak prominence relative to window edge minutes
    edge_mean = 0.5 * (L_c[0] + L_c[-1])
    cpa_prom = float(L_c.max() - edge_mean)

    # Adjacent observed minute differences and lag-1/lag-2 autocorrelation
    diffs = np.abs(np.diff(L_c)) if len(L_c) >= 2 else np.array([0.0])
    adj_max_diff = float(diffs.max())
    adj_med_diff = float(np.median(diffs))
    smoothness_ratio = float(adj_med_diff / max(L_c.max() - L_c.min(), 1e-6))

    lag1_corr = _safe_corrcoef_1d(L_c[:-1], L_c[1:]) if len(L_c) >= 4 else 0.0
    lag2_corr = _safe_corrcoef_1d(L_c[:-2], L_c[2:]) if len(L_c) >= 5 else 0.0

    # Cross-band co-modulation: fraction of 25 bands correlating (r > 0.45) with broadband L_c
    band_corrs = np.empty(25, dtype=np.float64)
    X_valid = Xc_win[idx_valid]
    for b in range(25):
        band_corrs[b] = _safe_corrcoef_1d(X_valid[:, b], L_c)
    frac_comod_45 = float((band_corrs > 0.45).mean())
    frac_comod_65 = float((band_corrs > 0.65).mean())
    mean_comod = float(band_corrs.mean())
    low_high_comod = float(band_corrs[:10].mean() - band_corrs[15:].mean())

    return {
        "morph_smooth_range_db": smooth_range,
        "morph_raw_range_db": float(L_c.max() - L_c.min()),
        "morph_quad_concavity": concavity,
        "morph_quad_r2": float(np.clip(quad_r2, -1.0, 1.0)),
        "morph_cpa_prominence": cpa_prom,
        "morph_adj_max_diff": adj_max_diff,
        "morph_adj_med_diff": adj_med_diff,
        "morph_smoothness_ratio": smoothness_ratio,
        "morph_lag1_corr": lag1_corr,
        "morph_lag2_corr": lag2_corr,
        "morph_frac_comod_045": frac_comod_45,
        "morph_frac_comod_065": frac_comod_65,
        "morph_mean_comod": mean_comod,
        "morph_low_high_comod_diff": low_high_comod,
    }


def extract_features(X_db: np.ndarray) -> pd.DataFrame:
    """Extract shift-invariant, mask-aware tabular features from (N, 15, 25) dB matrices."""
    X = np.asarray(X_db, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (15, 25):
        raise ValueError(f"Expected shape (N, 15, 25), got {X.shape}")

    n = X.shape[0]
    obs = np.isfinite(X)
    min_obs = obs.all(axis=2)  # (N, 15)
    assert min_obs.any(axis=1).all(), "Window with zero observed minutes"

    # Global offset-invariant centering
    win_med = np.nanmedian(X, axis=(1, 2), keepdims=True)
    Xc = X - win_med
    E_lin = np.where(obs, np.power(10.0, np.clip(Xc, -60.0, 60.0) / 10.0), np.nan)
    total_e = np.nansum(E_lin, axis=(1, 2))
    eps = 1e-30

    feats: dict[str, np.ndarray] = {}

    # --- Group A: Multi-quantile baseline physics excess shares & broadband distribution ---
    shares_by_q: dict[int, np.ndarray] = {}
    for q in QUANTILES:
        b_db = np.nanpercentile(Xc, q, axis=1)  # (N, 25)
        b_lin = np.power(10.0, np.clip(b_db, -60.0, 60.0) / 10.0)
        ex_lin = np.where(obs, np.maximum(E_lin - b_lin[:, None, :], 0.0), 0.0)
        sh = np.clip(ex_lin.sum(axis=(1, 2)) / np.maximum(total_e, eps), 0.0, 1.0)
        shares_by_q[q] = sh
        feats[f"phys_share_q{q}"] = sh
        sh_clip = np.clip(sh, 1e-4, 1.0 - 1e-4)
        feats[f"phys_logit_q{q}"] = np.log(sh_clip / (1.0 - sh_clip))

    feats["phys_share_q10_minus_q50"] = shares_by_q[10] - shares_by_q[50]
    feats["phys_share_q5_minus_q25"] = shares_by_q[5] - shares_by_q[25]
    feats["phys_share_q15_ratio_q35"] = shares_by_q[15] / (shares_by_q[35] + 1e-4)

    # Broadband minute series metrics
    min_tot_e = np.where(min_obs, np.nansum(E_lin, axis=2), np.nan)
    min_tot_db = 10.0 * np.log10(np.maximum(min_tot_e, eps))
    min_tot_db[~min_obs] = np.nan

    min_db_p05 = np.nanpercentile(min_tot_db, 5, axis=1)
    min_db_p15 = np.nanpercentile(min_tot_db, 15, axis=1)
    min_db_p50 = np.nanpercentile(min_tot_db, 50, axis=1)
    min_db_p85 = np.nanpercentile(min_tot_db, 85, axis=1)
    min_db_p95 = np.nanpercentile(min_tot_db, 95, axis=1)

    feats["bb_range_95_05"] = min_db_p95 - min_db_p05
    feats["bb_range_95_15"] = min_db_p95 - min_db_p15
    feats["bb_upper_tail_95_50"] = min_db_p95 - min_db_p50
    feats["bb_lower_tail_50_05"] = min_db_p50 - min_db_p05
    feats["bb_asymmetry"] = (min_db_p95 - min_db_p50) - (min_db_p50 - min_db_p05)
    feats["bb_std_db"] = np.nanstd(min_tot_db, axis=1)
    feats["bb_iqr_db"] = np.nanpercentile(min_tot_db, 75, axis=1) - np.nanpercentile(min_tot_db, 25, axis=1)
    feats["bb_frac_above_med_3db"] = np.nanmean((min_tot_db - min_db_p50[:, None]) > 3.0, axis=1)
    feats["bb_frac_above_p15_4db"] = np.nanmean((min_tot_db - min_db_p15[:, None]) > 4.0, axis=1)
    feats["bb_frac_above_p15_8db"] = np.nanmean((min_tot_db - min_db_p15[:, None]) > 8.0, axis=1)
    feats["bb_linear_max_over_p15"] = np.nanmax(min_tot_e, axis=1) / np.maximum(np.nanpercentile(min_tot_e, 15, axis=1), eps)

    # --- Group B: Sub-band dynamics, excess shares, spectral shape & tilt ---
    band_mean_db = np.nanmean(Xc, axis=1)         # (N, 25)
    band_std_db = np.nanstd(Xc, axis=1)           # (N, 25)
    band_p15_db = np.nanpercentile(Xc, 15, axis=1)
    band_p90_db = np.nanpercentile(Xc, 90, axis=1)
    band_mean_lin = np.nanmean(E_lin, axis=1)     # (N, 25)
    band_frac_energy = band_mean_lin / np.maximum(band_mean_lin.sum(axis=1, keepdims=True), eps)

    for sb_name, sl in SUB_BANDS.items():
        sb_e = E_lin[:, :, sl]
        sb_obs = obs[:, :, sl]
        sb_tot_lin = np.nansum(sb_e, axis=2)  # (N, 15)
        sb_tot_lin[~min_obs] = np.nan
        sb_db = 10.0 * np.log10(np.maximum(sb_tot_lin, eps))
        sb_db[~min_obs] = np.nan

        for q in (10, 25):
            b_sb = np.nanpercentile(Xc[:, :, sl], q, axis=1)
            b_sb_lin = np.power(10.0, np.clip(b_sb, -60.0, 60.0) / 10.0)
            ex_sb = np.where(sb_obs, np.maximum(sb_e - b_sb_lin[:, None, :], 0.0), 0.0)
            feats[f"{sb_name}_share_q{q}"] = ex_sb.sum(axis=(1, 2)) / np.maximum(np.nansum(sb_e, axis=(1, 2)), eps)
            feats[f"{sb_name}_contrib_q{q}"] = ex_sb.sum(axis=(1, 2)) / np.maximum(total_e, eps)

        feats[f"{sb_name}_mean_rel_db"] = np.nanmean(band_mean_db[:, sl], axis=1)
        feats[f"{sb_name}_energy_frac"] = band_frac_energy[:, sl].sum(axis=1)
        feats[f"{sb_name}_range_95_15"] = np.nanpercentile(sb_db, 95, axis=1) - np.nanpercentile(sb_db, 15, axis=1)
        feats[f"{sb_name}_std_db"] = np.nanstd(sb_db, axis=1)
        feats[f"{sb_name}_upper_tail"] = np.nanpercentile(sb_db, 95, axis=1) - np.nanpercentile(sb_db, 50, axis=1)
        feats[f"{sb_name}_frac_gt_p15_3db"] = np.nanmean((sb_db - np.nanpercentile(sb_db, 15, axis=1)[:, None]) > 3.0, axis=1)

    feats["spec_ratio_vlf_hf_db"] = feats["vlf_mean_rel_db"] - feats["hf_mean_rel_db"]
    feats["spec_ratio_lf_hf_db"] = feats["lf_mean_rel_db"] - feats["hf_mean_rel_db"]
    feats["spec_ratio_mf_lf_db"] = feats["mf_mean_rel_db"] - feats["lf_mean_rel_db"]
    feats["spec_ratio_hf_midlow_db"] = feats["hf_mean_rel_db"] - 0.5 * (feats["lf_mean_rel_db"] + feats["mf_mean_rel_db"])

    # Spectral tilt slope, flatness, crest, entropy, peak concentration
    feats["spec_tilt_slope_db_per_dec"] = (band_mean_db @ LOG10_F_C) / LOG10_F_VAR
    feats["spec_peak_band_idx"] = np.argmax(band_mean_lin, axis=1).astype(np.float64) / 24.0
    feats["spec_crest_factor"] = band_frac_energy.max(axis=1) * 25.0
    top3_sorted = np.sort(band_frac_energy, axis=1)[:, -3:].sum(axis=1)
    feats["spec_top3_energy_frac"] = top3_sorted
    feats["spec_entropy"] = -np.sum(band_frac_energy * np.log(np.maximum(band_frac_energy, 1e-12)), axis=1) / np.log(25.0)
    feats["spec_flatness_db"] = band_mean_db.mean(axis=1) - 10.0 * np.log10(np.maximum(band_mean_lin.mean(axis=1), eps))

    # Per-band emergence summaries + selected peak emergence bands
    band_emerg = band_p90_db - band_p15_db
    feats["band_emerg_mean"] = band_emerg.mean(axis=1)
    feats["band_emerg_std"] = band_emerg.std(axis=1)
    feats["band_emerg_max"] = band_emerg.max(axis=1)
    feats["band_emerg_p75"] = np.percentile(band_emerg, 75, axis=1)
    feats["band_emerg_argmax"] = np.argmax(band_emerg, axis=1).astype(np.float64) / 24.0
    feats["band_std_mean"] = band_std_db.mean(axis=1)
    feats["band_std_max"] = band_std_db.max(axis=1)
    for idx in (0, 2, 5, 8, 10, 13, 15, 18, 21, 24):
        feats[f"band_rel_db_b{idx:02d}"] = band_mean_db[:, idx]
        feats[f"band_emerg_b{idx:02d}"] = band_emerg[:, idx]
        feats[f"band_std_b{idx:02d}"] = band_std_db[:, idx]

    # Compare active high-minute spectrum vs quiet low-minute spectrum (pure shipping spectral shift!)
    order_min = np.argsort(np.where(min_obs, min_tot_db, np.inf), axis=1)
    n_valid_arr = min_obs.sum(axis=1)
    quiet_prof = np.empty((n, 25), dtype=np.float64)
    loud_prof = np.empty((n, 25), dtype=np.float64)
    morph_rows = []
    for i in range(n):
        nv = int(n_valid_arr[i])
        k = max(1, nv // 5)
        valid_sorted = order_min[i, :nv]
        quiet_prof[i] = Xc[i, valid_sorted[:k]].mean(axis=0)
        loud_prof[i] = Xc[i, valid_sorted[-k:]].mean(axis=0)
        morph_rows.append(_extract_single_window_morphology(Xc[i], E_lin[i], min_obs[i]))

    spec_diff = loud_prof - quiet_prof
    feats["active_quiet_diff_mean_db"] = spec_diff.mean(axis=1)
    feats["active_quiet_diff_max_db"] = spec_diff.max(axis=1)
    feats["active_quiet_diff_min_db"] = spec_diff.min(axis=1)
    feats["active_quiet_diff_std_db"] = spec_diff.std(axis=1)
    feats["active_quiet_tilt_change"] = (spec_diff @ LOG10_F_C) / LOG10_F_VAR
    # Cosine similarity of shape difference vs normalized quiet shape
    q_norm = np.linalg.norm(quiet_prof - quiet_prof.mean(axis=1, keepdims=True), axis=1)
    l_norm = np.linalg.norm(loud_prof - loud_prof.mean(axis=1, keepdims=True), axis=1)
    dot = np.sum(
        (quiet_prof - quiet_prof.mean(axis=1, keepdims=True))
        * (loud_prof - loud_prof.mean(axis=1, keepdims=True)),
        axis=1,
    )
    feats["active_quiet_shape_cosine"] = dot / np.maximum(q_norm * l_norm, 1e-9)

    morph_df = pd.DataFrame(morph_rows)
    for col in morph_df.columns:
        feats[col] = morph_df[col].to_numpy(dtype=np.float64)

    # --- Group D: Mask telemetry ---
    feats["mask_obs_frac"] = min_obs.mean(axis=1)
    feats["mask_m01_missing"] = (~min_obs[:, 0]).astype(np.float64)
    feats["mask_m15_missing"] = (~min_obs[:, 14]).astype(np.float64)

    df = pd.DataFrame(feats)
    assert np.isfinite(df.to_numpy()).all(), "Non-finite tabular features detected"
    return df


def build_cnn_tensors(X_db: np.ndarray) -> np.ndarray:
    """Build shift-invariant 4-channel tensor (N, 4, 15, 25) float32 for PyTorch 2D CNN."""
    X = np.asarray(X_db, dtype=np.float64)
    obs = np.isfinite(X)
    win_med = np.nanmedian(X, axis=(1, 2), keepdims=True)
    Xc = X - win_med

    band_p15 = np.nanpercentile(Xc, 15, axis=1, keepdims=True)
    obs_sum = np.where(obs, Xc, 0.0).sum(axis=2, keepdims=True)
    obs_cnt = np.maximum(obs.sum(axis=2, keepdims=True), 1)
    min_mean = obs_sum / obs_cnt

    ch0 = np.where(obs, np.clip(Xc / 15.0, -4.0, 4.0), 0.0)
    ch1 = np.where(obs, np.clip((Xc - band_p15) / 15.0, -4.0, 4.0), 0.0)
    ch2 = np.where(obs, np.clip((Xc - min_mean) / 15.0, -4.0, 4.0), 0.0)
    ch3 = obs.astype(np.float64)

    tensors = np.stack([ch0, ch1, ch2, ch3], axis=1).astype(np.float32)
    assert np.isfinite(tensors).all()
    return tensors


def extract_v1_combined96_features(X_db: np.ndarray) -> np.ndarray:
    """Reproduce Notebook 01 Experiment 002 combined_96 + per-window max shift baseline."""
    X = np.asarray(X_db, dtype=np.float64).copy()
    if X.ndim != 3 or X.shape[1:] != (15, 25):
        raise ValueError(f"Expected shape (N, 15, 25), got {X.shape}")

    t_all = np.arange(15, dtype=np.float64)
    n = X.shape[0]
    for i in range(n):
        win = X[i]
        if not np.isfinite(win).all():
            obs_t = np.isfinite(win).all(axis=1)
            assert obs_t.any(), "Window with zero observed minutes"
            valid_t = t_all[obs_t]
            for b in range(25):
                win[:, b] = np.interp(t_all, valid_t, win[obs_t, b])
        win -= win.max()

    global_ = np.column_stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2)), X.min(axis=(1, 2)), X.max(axis=(1, 2))])
    spectral = np.column_stack([X.mean(axis=1), X.std(axis=1)])
    diff = np.diff(X, axis=1)
    temporal = np.column_stack([diff.mean(axis=2), diff.std(axis=2), np.abs(diff).mean(axis=2)])
    combined = np.column_stack([global_, spectral, temporal])
    assert combined.shape == (n, 96) and np.isfinite(combined).all()
    return combined


def run_self_test() -> None:
    """Verify invariance of V2 features/tensors and finite V1 legacy baseline features under NaNs."""
    rng = np.random.default_rng(123)
    X = rng.normal(90.0, 6.0, size=(24, 15, 25))
    # Whole-minute NaN masks including missing m01, m15, interior
    X[:8, 0, :] = np.nan
    X[8:14, 14, :] = np.nan
    X[14:16, 7, :] = np.nan
    X_shifted = X + 37.5

    f1 = extract_features(X)
    f2 = extract_features(X_shifted)
    max_diff = float(np.max(np.abs(f1.to_numpy() - f2.to_numpy())))
    assert max_diff < 1e-10, f"Tabular invariance violated: max_diff={max_diff}"

    t1 = build_cnn_tensors(X)
    t2 = build_cnn_tensors(X_shifted)
    t_diff = float(np.max(np.abs(t1 - t2)))
    assert t_diff < 1e-5, f"CNN tensor invariance violated: max_diff={t_diff}"

    s1, p1 = compute_physics_baseline_share(X)
    s2, p2 = compute_physics_baseline_share(X_shifted)
    assert float(np.max(np.abs(s1 - s2))) < 1e-12
    assert float(np.max(np.abs(p1 - p2))) < 1e-12

    v1 = extract_v1_combined96_features(X)
    v1_s = extract_v1_combined96_features(X_shifted)
    assert v1.shape == (24, 96) and np.allclose(v1, v1_s, atol=1e-10)
    print(f"modeling_v2_features self-test: PASS ({f1.shape[1]} V2 features, {v1.shape[1]} V1 features, max_shift_diff={max_diff:.2e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_only:
        run_self_test()


if __name__ == "__main__":
    main()
