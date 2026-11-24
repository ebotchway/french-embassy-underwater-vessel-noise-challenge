#!/usr/bin/env python3
"""Site-level held-out relative vessel-like acoustic activity, observed-signal support & uncertainty audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_mixtures_v1 import BANDS  # noqa: E402
from extract_external_decidecade import EXACT_CENTRES  # noqa: E402
from modeling_v2_train import predict_from_bundle as predict_v2  # noqa: E402
from modeling_v3_features import compute_robust_physics_share  # noqa: E402
from modeling_v4_simulator import LOG10_F_NORM, STRATA_ORDER, profile_backgrounds_v4  # noqa: E402
from modeling_v4_train import predict_from_bundle_v4 as predict_v4  # noqa: E402

MANIFEST_DEFAULT = ROOT / "reports/eda/background_split_manifest_v2.csv"
STATIONS_CSV_DEFAULT = ROOT / "data/zindi/extracted/stations.csv"
V2_BUNDLE_DEFAULT = ROOT / "models/v2/bundle.joblib"
V4_BUNDLE_DEFAULT = ROOT / "models/v4/bundle_v4.joblib"
OUT_DIR_DEFAULT = ROOT / "reports/standards_benchmark"

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#666666"]

SITE_TO_CANONICAL = {
    "1 Richards Bay": "01_Richards_Bay",
    "2 Durban": "02_Durban",
    "3 Cape_Recife": "03_Cape_Recife",
    "4 Bird Island": "04_Bird_Island",
    "5 St Croix Island": "05_St_Croix_Island",
    "6 Riy_Banks": "06_Riy_Banks",
    "7 St_Francis": "07_St_Francis",
    "9 Fish Hoek": "09_Fish_Hoek",
    "10 Sea Point": "10_Sea_Point",
    "13 Saldhana Bay": "13_Saldhana_Bay",
}

DISPLAY_NAME = {
    "01_Richards_Bay": "Richards Bay",
    "02_Durban": "Durban",
    "03_Cape_Recife": "Cape Recife",
    "04_Bird_Island": "Bird Island",
    "05_St_Croix_Island": "St Croix",
    "06_Riy_Banks": "Riy Banks",
    "07_St_Francis": "St Francis",
    "09_Fish_Hoek": "Fish Hoek",
    "10_Sea_Point": "Sea Point",
    "13_Saldhana_Bay": "Saldanha Bay",
}


def compute_temporal_excess_residuals(X_db: np.ndarray) -> np.ndarray:
    """Compute exact h(f)-invariant per-band excess residual R(t, f) = X(t, f) - Q_{0.20}(X(:, f))."""
    arr = np.asarray(X_db, dtype=np.float64)
    q20 = np.nanpercentile(arr, 20.0, axis=1, keepdims=True)
    return arr - q20


def classify_r_hf_ship_proxy(X_db: np.ndarray) -> np.ndarray:
    """Deterministic h(f)-invariant localized MF+HF vessel-like transient indicator over clean matrices."""
    R = compute_temporal_excess_residuals(X_db)
    hf_p = np.power(10.0, np.clip(R[:, :, 17:25], -40.0, 40.0) / 10.0).mean(axis=2)
    mf_p = np.power(10.0, np.clip(R[:, :, 11:17], -40.0, 40.0) / 10.0).mean(axis=2)
    bb_p = np.power(10.0, np.clip(R, -40.0, 40.0) / 10.0).mean(axis=2)

    hf_db = 10.0 * np.log10(np.maximum(hf_p, 1e-30))
    mf_db = 10.0 * np.log10(np.maximum(mf_p, 1e-30))
    bb_db = 10.0 * np.log10(np.maximum(bb_p, 1e-30))

    hf_rng = hf_db.max(axis=1) - hf_db.min(axis=1)
    mf_rng = mf_db.max(axis=1) - mf_db.min(axis=1)
    amx_hf = np.argmax(hf_db, axis=1)
    w6_hf = (hf_db >= hf_db.max(axis=1, keepdims=True) - 6.0).sum(axis=1)
    drop_db = np.median(bb_db, axis=1) - bb_db.min(axis=1)
    return (
        (drop_db <= 14.0)
        & (hf_rng >= 10.0)
        & (mf_rng >= 8.0)
        & (amx_hf >= 1)
        & (amx_hf <= 13)
        & (w6_hf <= 8)
    )


def day_block_bootstrap_ci(
    days: np.ndarray,
    values: np.ndarray,
    stat_fn: str = "mean",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and 1,000-draw calendar-day block bootstrap 95% CI."""
    uniq_days, inv = np.unique(days, return_inverse=True)
    n_days = len(uniq_days)
    if stat_fn == "mean":
        point = float(np.mean(values))
        day_sums = np.bincount(inv, weights=values, minlength=n_days).astype(np.float64)
        day_cnts = np.bincount(inv, minlength=n_days).astype(np.float64)
        if n_days < 2:
            return point, point, point
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            pick = rng.integers(0, n_days, size=n_days)
            draws[i] = float(day_sums[pick].sum() / max(day_cnts[pick].sum(), 1.0))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return point, float(lo), float(hi)

    if stat_fn == "median":
        point = float(np.median(values))
        if n_days < 2:
            return point, point, point
        buckets = [values[inv == d] for d in range(n_days)]
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            pick = rng.integers(0, n_days, size=n_days)
            samp = np.concatenate([buckets[p] for p in pick])
            draws[i] = float(np.median(samp))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return point, float(lo), float(hi)

    raise ValueError(f"Unsupported stat_fn={stat_fn}")


def run_self_test() -> None:
    """Verify exact h(f) calibration invariance and calendar-day block bootstrap in memory."""
    rng = np.random.default_rng(42)
    t_idx = np.arange(15, dtype=np.float64)
    X = np.full((12, 15, 25), 72.0, dtype=np.float64) + rng.normal(0.0, 0.35, size=(12, 15, 25))
    # Inject 4 clean interior broadband transient passages
    bell = 16.0 * np.exp(-0.5 * ((t_idx - 7.0) / 1.4) ** 2)[:, None]
    X[:4, :, 6:25] += bell
    # Inject 2 VLF swell windows (no MF/HF transient)
    X[4:6, :, :6] += 13.0 * np.exp(-0.5 * ((t_idx - 7.0) / 3.0) ** 2)[:, None]
    # Inject 2 telemetry dropout glitches (>14 dB drop)
    X[6:8, 9, :] -= 22.0

    tilt = 8.0 * LOG10_F_NORM[None, None, :] + 15.0
    X_warp = X + tilt

    R0 = compute_temporal_excess_residuals(X)
    R1 = compute_temporal_excess_residuals(X_warp)
    max_r_diff = float(np.max(np.abs(R0 - R1)))
    assert max_r_diff <= 1e-12, f"R(t,f) invariance violated under +8 dB/dec +15 dB: {max_r_diff}"

    flips = int((classify_r_hf_ship_proxy(X) != classify_r_hf_ship_proxy(X_warp)).sum())
    assert flips == 0, f"Relative transient proxy flipped under +8 dB/dec warp: {flips}"
    assert classify_r_hf_ship_proxy(X)[:4].all() and not classify_r_hf_ship_proxy(X)[4:].any()

    days = np.repeat(np.array(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]), 3)
    vals = classify_r_hf_ship_proxy(X).astype(np.float64)
    pt, lo, hi = day_block_bootstrap_ci(days, vals, stat_fn="mean", n_boot=200, seed=42)
    assert 0.0 <= lo <= pt <= hi <= 1.0, f"Invalid bootstrap CI: {(pt, lo, hi)}"
    print(f"PASS SA relative activity self-test (max|dR|={max_r_diff:.2e}, tilt_flips={flips}, ci=({lo:.3f},{hi:.3f}))")


def _extract_heldout_metrics(heldout_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact per-window acoustic descriptors and strata strictly for held-out real SA windows."""
    work = profile_backgrounds_v4(heldout_df).copy()
    work["window_mean_db"] = np.nan
    work["obs_vlf_peak_r_db"] = np.nan
    work["obs_mf_peak_r_db"] = np.nan
    work["obs_hf_peak_r_db"] = np.nan
    work["obs_bb_rng_r_db"] = np.nan
    work["calendar_day"] = pd.to_datetime(work["start_time"]).dt.strftime("%Y-%m-%d")
    work["is_rel_vessel_proxy"] = (work["stratum"] == "strict_hf_ship").astype(np.float64)

    for pq_path, grp in work.groupby("source_parquet", sort=True):
        arr = pd.read_parquet(str(pq_path), columns=BANDS).to_numpy(dtype=np.float64)
        pos_mat = np.asarray([json.loads(s) for s in grp["row_positions_json"]], dtype=np.int64)
        W = arr[pos_mat]
        R = compute_temporal_excess_residuals(W)
        idx_arr = grp.index.to_numpy()
        work.loc[idx_arr, "window_mean_db"] = W.mean(axis=(1, 2))
        work.loc[idx_arr, "obs_vlf_peak_r_db"] = R[:, :, 0:6].max(axis=(1, 2))
        work.loc[idx_arr, "obs_mf_peak_r_db"] = R[:, :, 11:17].max(axis=(1, 2))
        work.loc[idx_arr, "obs_hf_peak_r_db"] = R[:, :, 17:25].max(axis=(1, 2))
        work.loc[idx_arr, "obs_bb_rng_r_db"] = R.max(axis=(1, 2))
    return work


def _sample_heldout_windows(work: pd.DataFrame, max_per_site: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """Build deterministic stratified held-out window sample across all 8 usable sites."""
    picked_frames = []
    for st in sorted(work["station"].unique()):
        sub = work[work["station"] == st].sort_values(["start_time", "run_id"]).reset_index(drop=True)
        if len(sub) <= max_per_site:
            picked_frames.append(sub)
        else:
            pick_idx = np.linspace(0, len(sub) - 1, max_per_site, dtype=int)
            picked_frames.append(sub.iloc[pick_idx].copy())
    samp = pd.concat(picked_frames, ignore_index=True)

    X_list = []
    samp["orig_order"] = np.arange(len(samp))
    rows_out = np.zeros((len(samp), 15, 25), dtype=np.float64)
    for pq_path, grp in samp.groupby("source_parquet", sort=True):
        arr = pd.read_parquet(str(pq_path), columns=BANDS).to_numpy(dtype=np.float64)
        pos_mat = np.asarray([json.loads(s) for s in grp["row_positions_json"]], dtype=np.int64)
        rows_out[grp["orig_order"].to_numpy()] = arr[pos_mat]
    return samp.drop(columns=["orig_order"]), rows_out


def plot_site_relative_activity_figure(
    site_df: pd.DataFrame,
    support_df: pd.DataFrame,
    out_png: Path,
    n_samp: int,
) -> None:
    """Render 300-DPI 3-panel Zindi-compliant site-level relative acoustic activity & held-out signal support figure."""
    plt.rcParams.update({
        "font.size": 8.6,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "grid.linestyle": "--",
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.55), gridspec_kw={"width_ratios": [1.18, 1.15, 1.05]})

    # --- Panel (a): Site-Level Relative Acoustic Activity Map (All 10 SA Candidate Coordinates) ---
    ax0 = axes[0]
    usable = site_df[site_df["usable_15min"]].copy()
    excluded = site_df[~site_df["usable_15min"]].copy()

    vmin, vmax = 0.0, max(20.0, float(usable["rel_vessel_activity_pct"].max()) * 1.12)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Highlight Saldanha Bay candidate uncertainty span first
    sal = usable[usable["station"] == "13_Saldhana_Bay"].iloc[0]
    if pd.notna(sal["lon_alt"]) and pd.notna(sal["lat_alt"]):
        ax0.plot([sal["lon"], sal["lon_alt"]], [sal["lat"], sal["lat_alt"]], color="#C0392B", ls="--", lw=1.2, zorder=3)
        ax0.scatter([sal["lon_alt"]], [sal["lat_alt"]], marker="x", s=46, color="#C0392B", lw=1.6, zorder=4, label="Unresolved Saldanha alt. coord.")

    # Plot excluded <15-min sites cleanly
    ax0.scatter(
        excluded["lon"],
        excluded["lat"],
        marker="s",
        s=54,
        facecolors="none",
        edgecolors="#555555",
        linewidths=1.4,
        zorder=4,
        label="Excluded (<15 continuous min: n=0)",
    )

    # Plot held-out validation periods (circles) vs completely unseen holdout sites (stars)
    val_sites = usable[usable["eval_scope"] == "heldout_period_val"]
    ho_sites = usable[usable["eval_scope"] == "unseen_site_holdout"]

    sc_val = ax0.scatter(
        val_sites["lon"],
        val_sites["lat"],
        c=val_sites["rel_vessel_activity_pct"],
        cmap=cmap,
        norm=norm,
        s=32.0 + 18.0 * np.log1p(val_sites["windows_heldout"]),
        marker="o",
        edgecolors="#1A1A1A",
        linewidths=0.9,
        zorder=5,
        label="Held-out period (6 dev sites)",
    )
    ax0.scatter(
        ho_sites["lon"],
        ho_sites["lat"],
        c=ho_sites["rel_vessel_activity_pct"],
        cmap=cmap,
        norm=norm,
        s=155.0 + 14.0 * np.log1p(ho_sites["windows_heldout"]),
        marker="*",
        edgecolors="#1A1A1A",
        linewidths=0.9,
        zorder=6,
        label="Unseen site holdout (Fish Hoek, Sea Point)",
    )

    # Explicit non-overlapping label offsets per station
    offsets = {
        "Richards Bay": (0, 20, "center"),
        "Durban": (-18, -8, "right"),
        "Bird Island": (22, 16, "left"),
        "St Croix": (-16, 26, "right"),
        "Riy Banks": (26, -24, "left"),
        "Cape Recife": (2, -34, "center"),
        "St Francis": (-16, -24, "right"),
        "Fish Hoek": (-6, -28, "center"),
        "Sea Point": (14, 12, "left"),
        "Saldanha Bay": (18, 18, "left"),
    }
    for r in site_df.itertuples(index=False):
        dx, dy, ha = offsets[r.display_name]
        txt = f"{r.display_name}\n({r.rel_vessel_activity_pct:.1f}%)" if r.usable_15min else f"{r.display_name}\n(0 win)"
        col = "#444444" if not r.usable_15min else "#111111"
        ax0.annotate(
            txt,
            xy=(r.lon, r.lat),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.9,
            fontweight="bold" if r.eval_scope == "unseen_site_holdout" else "normal",
            color=col,
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.65, "shrinkA": 2, "shrinkB": 3},
        )

    cbar = fig.colorbar(sc_val, ax=ax0, fraction=0.044, pad=0.02)
    cbar.set_label("Relative Acoustic Activity (%, h(f)-inv.)", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.2)

    ax0.set_xlim(16.2, 34.4)
    ax0.set_ylim(-35.8, -26.2)
    ax0.set_xlabel("Longitude (°E, working candidate coords)")
    ax0.set_ylabel("Latitude (°S shown as negative)")
    ax0.set_title("(a) Site-Level Relative Acoustic Activity (Non-Interpolated)")
    ax0.legend(loc="upper left", fontsize=6.1, frameon=True, markerscale=0.72)

    # --- Panel (b): Relative Vessel Proxy vs Total-Sound Baseline & Day-Block 95% CI ---
    ax1 = axes[1]
    ordered = usable.sort_values("rel_vessel_activity_pct", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    bar_colors = [PALETTE[3] if s == "unseen_site_holdout" else PALETTE[0] for s in ordered["eval_scope"]]
    ax1.barh(y, ordered["rel_vessel_activity_pct"], color=bar_colors, height=0.60, alpha=0.90)

    for yi, r in enumerate(ordered.itertuples(index=False)):
        err_lo = max(0.0, r.rel_vessel_activity_pct - r.rel_ci95_low_pct)
        err_hi = max(0.0, r.rel_ci95_high_pct - r.rel_vessel_activity_pct)
        ls_ci = "--" if r.provisional_low_days else "-"
        eb = ax1.errorbar(
            r.rel_vessel_activity_pct,
            yi,
            xerr=[[err_lo], [err_hi]],
            fmt="none",
            ecolor="#1A1A1A",
            elinewidth=1.25,
            capsize=3.2,
            zorder=5,
        )
        eb[2][0].set_linestyle(ls_ci)
        prov_tag = "*" if r.provisional_low_days else ""
        ann_x = max(r.rel_ci95_high_pct, r.rel_vessel_activity_pct) + 0.55
        ax1.text(
            ann_x,
            yi,
            f"{r.rel_vessel_activity_pct:.1f}%{prov_tag} | B0: {r.total_sound_median_db:.1f} dB ({int(r.calendar_days)}d)",
            va="center",
            fontsize=6.8,
            color="#222222",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{r.display_name}{' [Unseen]' if r.eval_scope=='unseen_site_holdout' else ''}" for r in ordered.itertuples(index=False)],
        fontsize=7.6,
    )
    ax1.set_xlim(0, 42.0)
    ax1.set_xlabel("Relative Vessel-Like Activity (%, h(f)-inv. + 95% Day CI)")
    ax1.set_title("(b) Vessel Proxy vs. Total-Sound Baseline (B0)")
    legend_b = [
        Line2D([0], [0], color=PALETTE[0], lw=6, label="Held-out period (6 sites)"),
        Line2D([0], [0], color=PALETTE[3], lw=6, label="Unseen site holdout (2 sites)"),
        Line2D([0], [0], color="#1A1A1A", lw=1.2, ls="--", label="*Provisional CI (<5 days)"),
    ]
    ax1.legend(handles=legend_b, loc="lower right", fontsize=6.6, frameon=True)

    # --- Panel (c): Held-Out Observed Signal Support & Baseline Specificity ---
    ax2 = axes[2]
    all_sp = support_df[support_df["scope"] == "all_heldout_8_sites"].copy()
    strata_order = ["strict_hf_ship", "ambient_normal", "vlf_surf", "reef_active", "glitch_drop"]
    nice_str = ["Strict\nShip-Like", "Ambient\nNormal", "VLF Surf\nSwell", "Reef Biol.\nChorus", "Telemetry\nGlitch"]
    all_sp = all_sp.set_index("stratum").loc[strata_order].reset_index()

    xs = np.arange(len(all_sp))
    w_bar = 0.25
    ax2.bar(xs - w_bar, all_sp["obs_hf_peak_r_db"] / 30.0, width=w_bar, color=PALETTE[1], label="Obs. HF Excess R(t,f) / 30 dB")
    ax2.bar(xs, all_sp["physics_q20_share"], width=w_bar, color=PALETTE[6], label="Baseline: Physics_q20 (False Share)")
    ax2.bar(xs + w_bar, all_sp["ensemble_v4_share"], width=w_bar, color=PALETTE[2], label="Ensemble_v4 (Real Bg. Share)")

    for i, r in enumerate(all_sp.itertuples(index=False)):
        ax2.text(i - w_bar, r.obs_hf_peak_r_db / 30.0 + 0.015, f"{r.obs_hf_peak_r_db:.1f}dB", ha="center", va="bottom", fontsize=5.9, rotation=90)
        ax2.text(i, r.physics_q20_share + 0.015, f"{r.physics_q20_share:.2f}", ha="center", va="bottom", fontsize=6.1)
        ax2.text(i + w_bar, r.ensemble_v4_share + 0.015, f"{r.ensemble_v4_share:.3f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(nice_str, fontsize=7.3)
    ax2.set_ylim(0, 1.18)
    ax2.set_ylabel("Normalized Magnitude / Predicted Share")
    ax2.set_title(f"(c) Held-Out Real Signal Support & Specificity (n={n_samp:,})")
    ax2.legend(loc="upper left", fontsize=6.5, frameon=True)

    fig.suptitle(
        "South African Held-Out Acoustic Telemetry (17,487 Windows): Relative Acoustic Activity (Uncalibrated Proxy) — Zero Spatial Interpolation",
        fontsize=9.6,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(w_pad=1.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--stations-csv", type=Path, default=STATIONS_CSV_DEFAULT)
    parser.add_argument("--v2-bundle", type=Path, default=V2_BUNDLE_DEFAULT)
    parser.add_argument("--v4-bundle", type=Path, default=V4_BUNDLE_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(args.manifest)
    heldout_df = df_all[df_all["split"].isin(["validation", "station_holdout"])].copy().reset_index(drop=True)
    assert len(heldout_df) == 17487, f"Expected 17487 heldout real SA windows, got {len(heldout_df)}"

    work = _extract_heldout_metrics(heldout_df)
    samp_df, X_samp = _sample_heldout_windows(work, max_per_site=300)

    b2 = joblib.load(args.v2_bundle)
    b4 = joblib.load(args.v4_bundle)
    sh_phys, sc_phys = compute_robust_physics_share(X_samp)
    sh_v2, sc_v2 = predict_v2(b2, X_samp)
    sh_v4, sc_v4 = predict_v4(b4, X_samp)

    samp_df["physics_q20_share"] = sh_phys
    samp_df["physics_q20_score"] = sc_phys
    samp_df["ensemble_v2_share"] = sh_v2
    samp_df["ensemble_v2_score"] = sc_v2
    samp_df["ensemble_v4_share"] = sh_v4
    samp_df["ensemble_v4_score"] = sc_v4

    stations_raw = pd.read_csv(args.stations_csv)
    site_rows = []
    for st_row in stations_raw.itertuples(index=False):
        canon = SITE_TO_CANONICAL[st_row.site]
        dname = DISPLAY_NAME[canon]
        sub_full = work[work["station"] == canon]
        sub_samp = samp_df[samp_df["station"] == canon]

        if len(sub_full) == 0:
            site_rows.append({
                "station": canon,
                "display_name": dname,
                "usable_15min": False,
                "eval_scope": "excluded_lt15min",
                "lon": float(st_row.longitude_candidate),
                "lat": float(st_row.latitude_candidate),
                "lon_alt": float(st_row.longitude_alternative) if pd.notna(st_row.longitude_alternative) else np.nan,
                "lat_alt": float(st_row.latitude_alternative) if pd.notna(st_row.latitude_alternative) else np.nan,
                "coordinate_status": str(st_row.coordinate_status),
                "windows_heldout": 0,
                "runs_heldout": 0,
                "calendar_days": 0,
                "recorders": 0,
                "start_date": str(st_row.first_export_timestamp)[:10],
                "end_date": str(st_row.last_export_timestamp)[:10],
                "provisional_low_days": True,
                "rel_vessel_activity_pct": np.nan,
                "rel_ci95_low_pct": np.nan,
                "rel_ci95_high_pct": np.nan,
                "total_sound_median_db": np.nan,
                "total_ci95_low_db": np.nan,
                "total_ci95_high_db": np.nan,
                "obs_hf_peak_r_db_median": np.nan,
                "obs_mf_peak_r_db_median": np.nan,
                "obs_vlf_peak_r_db_median": np.nan,
                "sample_n": 0,
                "physics_q20_mean_share": np.nan,
                "ensemble_v2_mean_share": np.nan,
                "ensemble_v4_mean_share": np.nan,
                "ensemble_v4_mean_score": np.nan,
            })
            continue

        split_val = str(sub_full["split"].iloc[0])
        eval_scope = "unseen_site_holdout" if split_val == "station_holdout" else "heldout_period_val"
        days_arr = sub_full["calendar_day"].to_numpy()
        n_days = int(sub_full["calendar_day"].nunique())
        n_runs = int(sub_full["run_id"].nunique())

        rel_pt, rel_lo, rel_hi = day_block_bootstrap_ci(
            days_arr,
            100.0 * sub_full["is_rel_vessel_proxy"].to_numpy(),
            stat_fn="mean",
            n_boot=1000,
            seed=42,
        )
        tot_pt, tot_lo, tot_hi = day_block_bootstrap_ci(
            days_arr,
            sub_full["window_mean_db"].to_numpy(),
            stat_fn="median",
            n_boot=1000,
            seed=42,
        )

        site_rows.append({
            "station": canon,
            "display_name": dname,
            "usable_15min": True,
            "eval_scope": eval_scope,
            "lon": float(st_row.longitude_candidate),
            "lat": float(st_row.latitude_candidate),
            "lon_alt": float(st_row.longitude_alternative) if pd.notna(st_row.longitude_alternative) else np.nan,
            "lat_alt": float(st_row.latitude_alternative) if pd.notna(st_row.latitude_alternative) else np.nan,
            "coordinate_status": str(st_row.coordinate_status),
            "windows_heldout": int(len(sub_full)),
            "runs_heldout": n_runs,
            "calendar_days": n_days,
            "recorders": int(sub_full["recorder"].nunique()),
            "start_date": str(sub_full["calendar_day"].min()),
            "end_date": str(sub_full["calendar_day"].max()),
            "provisional_low_days": bool(n_days < 5),
            "rel_vessel_activity_pct": rel_pt,
            "rel_ci95_low_pct": rel_lo,
            "rel_ci95_high_pct": rel_hi,
            "total_sound_median_db": tot_pt,
            "total_ci95_low_db": tot_lo,
            "total_ci95_high_db": tot_hi,
            "obs_hf_peak_r_db_median": float(sub_full["obs_hf_peak_r_db"].median()),
            "obs_mf_peak_r_db_median": float(sub_full["obs_mf_peak_r_db"].median()),
            "obs_vlf_peak_r_db_median": float(sub_full["obs_vlf_peak_r_db"].median()),
            "sample_n": int(len(sub_samp)),
            "physics_q20_mean_share": float(sub_samp["physics_q20_share"].mean()),
            "ensemble_v2_mean_share": float(sub_samp["ensemble_v2_share"].mean()),
            "ensemble_v4_mean_share": float(sub_samp["ensemble_v4_share"].mean()),
            "ensemble_v4_mean_score": float(sub_samp["ensemble_v4_score"].mean()),
        })

    site_df = pd.DataFrame(site_rows).sort_values(["usable_15min", "station"], ascending=[False, True]).reset_index(drop=True)
    site_csv = args.out_dir / "sa_site_relative_vessel_activity.csv"
    site_df.to_csv(site_csv, index=False)

    # Observed-signal support by physical stratum & evaluation scope
    support_rows = []
    for sc_name, mask in [
        ("all_heldout_8_sites", np.ones(len(samp_df), dtype=bool)),
        ("unseen_sites_fish_hoek_sea_point", (samp_df["split"] == "station_holdout").to_numpy()),
        ("heldout_periods_6_dev_sites", (samp_df["split"] == "validation").to_numpy()),
    ]:
        part = samp_df[mask]
        for st_name in STRATA_ORDER:
            g = part[part["stratum"] == st_name]
            if len(g) == 0:
                continue
            support_rows.append({
                "scope": sc_name,
                "stratum": st_name,
                "n_windows": int(len(g)),
                "obs_vlf_peak_r_db": float(g["obs_vlf_peak_r_db"].median()),
                "obs_mf_peak_r_db": float(g["obs_mf_peak_r_db"].median()),
                "obs_hf_peak_r_db": float(g["obs_hf_peak_r_db"].median()),
                "physics_q20_share": float(g["physics_q20_share"].mean()),
                "ensemble_v2_share": float(g["ensemble_v2_share"].mean()),
                "ensemble_v4_share": float(g["ensemble_v4_share"].mean()),
                "ensemble_v4_score": float(g["ensemble_v4_score"].mean()),
            })
    support_df = pd.DataFrame(support_rows)
    support_csv = args.out_dir / "sa_heldout_signal_support.csv"
    support_df.to_csv(support_csv, index=False)

    fig_png = fig_dir / "fig2_site_relative_acoustic_activity_map.png"
    plot_site_relative_activity_figure(site_df, support_df, fig_png, n_samp=int(len(samp_df)))

    usable_df = site_df[site_df["usable_15min"]]
    summary = {
        "total_heldout_real_windows": int(len(work)),
        "sampled_model_windows": int(len(samp_df)),
        "usable_sites": int(len(usable_df)),
        "unseen_holdout_sites": ["09_Fish_Hoek", "10_Sea_Point"],
        "excluded_lt15min_sites": ["03_Cape_Recife", "06_Riy_Banks"],
        "bootstrap_draws": 1000,
        "bootstrap_seed": 42,
        "overall_rel_vessel_rate_pct": float(100.0 * work["is_rel_vessel_proxy"].mean()),
        "overall_physics_q20_false_share": float(samp_df["physics_q20_share"].mean()),
        "overall_ensemble_v2_false_share": float(samp_df["ensemble_v2_share"].mean()),
        "overall_ensemble_v4_false_share": float(samp_df["ensemble_v4_share"].mean()),
        "fig_sha256": hashlib.sha256(fig_png.read_bytes()).hexdigest(),
    }
    (args.out_dir / "sa_site_activity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nSite summary (relative acoustic activity):")
    print(site_df[["display_name", "eval_scope", "windows_heldout", "calendar_days", "rel_vessel_activity_pct", "rel_ci95_low_pct", "rel_ci95_high_pct", "total_sound_median_db", "physics_q20_mean_share", "ensemble_v4_mean_share"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
