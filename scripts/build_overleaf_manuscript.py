#!/usr/bin/env python3
"""Build, validate, and package an Overleaf-ready APA 7th LaTeX research manuscript."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OVERLEAF_DIR = ROOT / "deliverables/overleaf_manuscript"
FIG_DIR = OVERLEAF_DIR / "figures"
MAIN_TEX = OVERLEAF_DIR / "main.tex"
BIB_FILE = OVERLEAF_DIR / "references.bib"
README_FILE = OVERLEAF_DIR / "README_OVERLEAF.md"
MANIFEST_JSON = OVERLEAF_DIR / "OVERLEAF_MANIFEST.json"
OVERLEAF_ZIP = OVERLEAF_DIR / "overleaf_research_manuscript.zip"
PDF_FILE = OVERLEAF_DIR / "research_manuscript_latex.pdf"

EXACT_CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25, dtype=np.float64))
BAND_COLS = [f"TOL_{f:g}" for f in [
    31.6, 39.8, 50.1, 63.1, 79.4,
    100.0, 125.9, 158.5, 199.5, 251.2,
    316.2, 398.1, 501.2, 631.0, 794.3,
    1000.0, 1258.9, 1584.9, 1995.3, 2511.9,
    3162.3, 3981.1, 5011.9, 6309.6, 7943.3,
]]

STATION_LABELS = {
    "01_Richards_Bay": "Richards Bay",
    "02_Durban": "Durban (n=1)",
    "04_Bird_Island": "Bird Island",
    "05_St_Croix_Island": "St Croix Island",
    "07_St_Francis": "St Francis",
    "09_Fish_Hoek": "Fish Hoek (Holdout)",
    "10_Sea_Point": "Sea Point (Holdout)",
    "13_Saldhana_Bay": "Saldanha Bay",
}

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#332288"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tex_esc(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(fig_dir: Path) -> dict[str, dict]:
    """Generate all 7 publication-grade figures directly from verified repository outputs."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Six-Station Real Acoustic Variability (3 panels) ---
    st_sum = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/validated_station_summary.csv")
    boot_ci = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/run_bootstrap_intervals.csv")
    lvl = boot_ci[boot_ci["metric"] == "window_mean_level"].merge(
        st_sum[["station", "sampled_runs", "recorders", "median_temporal_band_variability"]],
        on="station",
    )
    lvl["label"] = lvl["station"].map(STATION_LABELS)
    lvl = lvl.sort_values("median")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
    y_pos = np.arange(len(lvl))
    axes[0].barh(y_pos, lvl["sampled_runs"], color=PALETTE[0], edgecolor="white")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(lvl["label"])
    axes[0].set_xlabel("Sampled Indexed Runs")
    axes[0].set_title("(a) Training Run Coverage (n = 460)")
    for yi, v in zip(y_pos, lvl["sampled_runs"]):
        axes[0].text(v + 2, yi, str(int(v)), va="center", fontsize=8)
    axes[0].set_xlim(0, 142)

    has_ci = lvl["run_bootstrap_low"].notna()
    axes[1].errorbar(
        lvl.loc[has_ci, "median"],
        y_pos[has_ci],
        xerr=[
            lvl.loc[has_ci, "median"] - lvl.loc[has_ci, "run_bootstrap_low"],
            lvl.loc[has_ci, "run_bootstrap_high"] - lvl.loc[has_ci, "median"],
        ],
        fmt="o",
        color=PALETTE[3],
        ecolor=PALETTE[5],
        elinewidth=1.8,
        capsize=4,
        label="Median +/- 95% Run CI",
    )
    axes[1].scatter(
        lvl.loc[~has_ci, "median"],
        y_pos[~has_ci],
        marker="D",
        s=48,
        color="#666666",
        zorder=5,
        label="Single run (no CI)",
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Source-Export Level (uncalibrated dB)")
    axes[1].set_title("(b) Median Level & 95% Bootstrap CI")
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].set_xlim(55, 98)

    axes[2].barh(y_pos, lvl["median_temporal_band_variability"], color=PALETTE[2], edgecolor="white")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("Median Within-Window SD (dB)")
    axes[2].set_title("(c) Within-Window Variability")
    for yi, v in zip(y_pos, lvl["median_temporal_band_variability"]):
        axes[2].text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=8)
    axes[2].set_xlim(0, 4.35)
    plt.tight_layout(w_pad=1.6)
    _save_fig(fig, fig_dir / "fig1_sa_real_acoustics.png")

    # --- Figure 2: Canonical Non-Interpolated Candidate Site Context Map ---
    src_map = ROOT / "deliverables/research_submission/figures/site_context_candidate_locations.png"
    assert src_map.is_file(), f"Missing canonical site map: {src_map}"
    shutil.copy2(src_map, fig_dir / "fig2_site_context_map.png")

    # --- Figure 3: Cross-Corpus Spectral Shapes, Shape Mismatch & Anomaly Spread ---
    sa_spec = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv")
    vt_2s = pd.read_parquet(ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet")
    hms_2s = pd.read_parquet(ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet")
    shape_dist = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/shape_distance_descriptive.csv")
    anom = pd.read_csv(ROOT / "reports/research_manuscript_v1/station_relative_anomaly_proxy.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for idx, st in enumerate(sorted(sa_spec["station"].unique())):
        sub = sa_spec[sa_spec["station"] == st].sort_values("band_hz")
        axes[0].semilogx(
            sub["band_hz"],
            sub["median_relative_shape_db"],
            color=PALETTE[idx % len(PALETTE)],
            lw=1.7,
            label=STATION_LABELS[st],
        )

    def _peak_norm_db(arr: np.ndarray) -> np.ndarray:
        db = 10.0 * np.log10(np.maximum(arr, 1e-30))
        med = np.median(db, axis=0)
        return med - np.max(med)

    vt_act = vt_2s[vt_2s["label"] != "background"][BAND_COLS].to_numpy()
    hms_mat = hms_2s[BAND_COLS].to_numpy()
    axes[0].semilogx(EXACT_CENTRES, _peak_norm_db(vt_act), color="black", ls="--", lw=2.0, label="VTUAD 2s Median (525 clips)")
    axes[0].semilogx(EXACT_CENTRES, _peak_norm_db(hms_mat), color="#882255", ls="-.", lw=2.0, label="HearMyShip 2s Median (96 clips)")
    axes[0].set_xlabel("Third-Octave Band Centre Frequency (Hz)")
    axes[0].set_ylabel("Peak-Normalized Level (dB)")
    axes[0].set_title("(a) Peak-Normalized Site vs External Vessel Spectra")
    axes[0].legend(ncol=2, loc="lower left", fontsize=7.6, frameon=True)

    med_dist = shape_dist.groupby("station")["median_shape_difference_rms_db"].median().sort_values()
    st_order = list(med_dist.index)
    y2 = np.arange(len(st_order))
    iqr_anom = []
    for st in st_order:
        vals = anom.loc[anom["station"] == st, "station_relative_level_deviation"].dropna()
        iqr_anom.append(float(np.percentile(vals, 90) - np.percentile(vals, 10)) if len(vals) > 1 else 0.0)

    bars = axes[1].barh(y2 + 0.18, med_dist.values, height=0.35, color=PALETTE[3], label="Median VTUAD RMS Distance (dB)")
    axes[1].barh(y2 - 0.18, iqr_anom, height=0.35, color=PALETTE[0], label="Station P90-P10 Run Spread (dB)")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([STATION_LABELS[s] for s in st_order])
    axes[1].set_xlabel("Magnitude (dB)")
    axes[1].set_title("(b) Spectral Shape Mismatch vs Run Spread")
    for rect, val in zip(bars, med_dist.values):
        axes[1].text(val + 0.4, rect.get_y() + rect.get_height() / 2, f"{val:.1f}", va="center", fontsize=7.8)
    axes[1].set_xlim(0, 33.5)
    axes[1].legend(loc="lower right", frameon=True, fontsize=7.8)
    plt.tight_layout(w_pad=1.6)
    _save_fig(fig, fig_dir / "fig3_spectral_structure_mismatch.png")

    # --- Figure 4: Test Missingness Profile & Experiment 007 Decision Gate Trade-off ---
    miss = pd.read_csv(ROOT / "reports/audits/phase3_representation_detection_v1/test_missingness_mask_by_minute.csv")
    pol7 = pd.read_csv(ROOT / "reports/audits/experiment007_decision_gate_v1/policy_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.9))
    pct_miss = 100.0 * miss["rows_full_minute_missing"] / 1500.0
    bar_cols = [PALETTE[3] if p > 5 else PALETTE[0] for p in pct_miss]
    axes[0].bar(miss["minute"], pct_miss, color=bar_cols, edgecolor="white")
    axes[0].set_xticks(miss["minute"])
    axes[0].set_xlabel("Window Minute Index (m01 to m15)")
    axes[0].set_ylabel("Missing Test Windows (%)")
    axes[0].set_title("(a) Read-Only Test.csv Whole-Minute Absence")
    axes[0].text(1.4, pct_miss.iloc[0] - 3.5, f"m01: {pct_miss.iloc[0]:.1f}%\n(624/1,500)", fontsize=8, fontweight="bold", color=PALETTE[3])
    axes[0].text(12.2, pct_miss.iloc[-1] + 2.0, f"m15: {pct_miss.iloc[-1]:.1f}%\n(100/1,500)", fontsize=8, color=PALETTE[3])
    axes[0].set_ylim(0, 48)

    pol_show = pol7[pol7["policy"].isin(["original_only", "contrast_varied", "hybrid"])].copy()
    pol_show["label"] = [
        f"{r.policy.replace('_', ' ').title()}\n(thr={r.threshold:.2f})"
        for r in pol_show.itertuples(index=False)
    ]
    # Keep concise non-redundant 4 prespecified rows reported in Table 2
    keep_mask = [0, 2, 3, 4]
    pol_sub = pol7.iloc[keep_mask].copy()
    short_labels = ["Orig\n(thr=0.50)", "Contrast\n(thr=0.50)", "Contrast\n(thr=0.55)", "Hybrid\n(thr=0.50)"]
    xp = np.arange(len(pol_sub))
    w = 0.27
    axes[1].bar(xp - w, pol_sub["rmse"], width=w, color=PALETTE[0], label="Share RMSE (lower better)")
    axes[1].bar(xp, pol_sub["fpr_no_addition"], width=w, color=PALETTE[3], label="Control FPR (lower better)")
    axes[1].bar(xp + w, pol_sub["tpr_observed_positive"], width=w, color=PALETTE[2], label="Synthetic TPR (higher better)")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(short_labels, fontsize=8.0)
    axes[1].set_ylabel("Metric Value")
    axes[1].set_title("(b) Exp 007 Trade-off (n = 840 Source-Disjoint Cases)")
    axes[1].legend(loc="upper left", ncol=2, fontsize=7.6, frameon=True)
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout(w_pad=1.6)
    _save_fig(fig, fig_dir / "fig4_domain_gap_exp007.png")

    # --- Figure 5: Deterministic 11-Model Benchmark (Val vs Unseen-Station Holdout) ---
    bm = pd.read_csv(ROOT / "reports/modeling_v2/benchmark_metrics.csv")
    val_bm = bm[bm["split"] == "val"].set_index("model")
    ho_bm = bm[bm["split"] == "holdout"].set_index("model")
    order = ho_bm.sort_values("zindi_score_1_minus_rmse").index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
    xm = np.arange(len(order))
    wb = 0.38
    axes[0].barh(xm - wb / 2, val_bm.loc[order, "zindi_score_1_minus_rmse"], height=wb, color=PALETTE[2], label="Validation (6 stations, n=1,500)")
    axes[0].barh(xm + wb / 2, ho_bm.loc[order, "zindi_score_1_minus_rmse"], height=wb, color=PALETTE[3], label="Unseen Holdout (2 stations, n=541)")
    axes[0].set_yticks(xm)
    axes[0].set_yticklabels(order, fontsize=8.5)
    axes[0].set_xlim(0.45, 0.94)
    axes[0].set_xlabel("Competition Metric (1 - RMSE)")
    axes[0].set_title("(a) Added-Share Accuracy Across All 11 Models")
    axes[0].axvline(float(ho_bm.loc["V1_RF_MaxShift", "zindi_score_1_minus_rmse"]), color="#666666", ls=":", lw=1.2)
    axes[0].legend(loc="lower right", fontsize=7.6, frameon=True)

    axes[1].barh(xm - wb / 2, val_bm.loc[order, "roc_auc"], height=wb, color=PALETTE[0], label="Validation")
    axes[1].barh(xm + wb / 2, ho_bm.loc[order, "roc_auc"], height=wb, color=PALETTE[1], label="Unseen Holdout")
    axes[1].set_yticks(xm)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0.45, 1.02)
    axes[1].set_xlabel("Detection ROC AUC")
    axes[1].set_title("(b) Vessel Presence Discrimination")
    axes[1].axvline(0.50, color="#999999", ls="--", lw=1.0)
    axes[1].legend(loc="lower right", fontsize=7.8, frameon=True)
    plt.tight_layout(w_pad=1.5)
    _save_fig(fig, fig_dir / "fig5_benchmark_11_models.png")

    # --- Figure 6: Interpretability (Top-15 Features) & Station Breakdown ---
    imp_df = pd.read_csv(ROOT / "reports/modeling_v2/feature_importance.csv").head(15).iloc[::-1]
    st_bd = pd.read_csv(ROOT / "reports/modeling_v2/station_breakdown.csv")
    piv_st = st_bd.pivot(index=["split", "station"], columns="model", values="rmse")[["Physics_q15", "V1_RF_MaxShift", "Ensemble_v2"]]
    piv_st = piv_st.sort_values("Ensemble_v2", ascending=False)

    def _feat_color(name: str) -> str:
        if name.startswith("morph_") or name.startswith("active_"):
            return PALETTE[3]
        if name.startswith(("hf_", "mf_", "lf_", "vlf_", "spec_")):
            return PALETTE[0]
        return PALETTE[2]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    f_cols = [_feat_color(f) for f in imp_df["feature"]]
    axes[0].barh(imp_df["feature"], imp_df["mean_norm_importance"], color=f_cols)
    axes[0].set_xlabel("Mean Normalized GBDT Gain")
    axes[0].set_title("(a) Top-15 Acoustic Predictors")
    axes[0].tick_params(axis="y", labelsize=8.0)
    axes[0].legend(
        handles=[
            Patch(facecolor=PALETTE[0], label="Band Contrib."),
            Patch(facecolor=PALETTE[3], label="Morphology"),
            Patch(facecolor=PALETTE[2], label="Emergence"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=7.6,
    )

    ys = np.arange(len(piv_st))
    labels_st = [f"{STATION_LABELS.get(st, st)}" for _, st in piv_st.index]
    axes[1].barh(ys - 0.25, piv_st["Physics_q15"], height=0.24, color="#999999", label="Physics_q15")
    axes[1].barh(ys, piv_st["V1_RF_MaxShift"], height=0.24, color=PALETTE[1], label="V1_RF_MaxShift")
    axes[1].barh(ys + 0.25, piv_st["Ensemble_v2"], height=0.24, color=PALETTE[2], label="Ensemble_v2")
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels(labels_st, fontsize=8.2)
    axes[1].set_xlim(0.0, 0.72)
    axes[1].set_xlabel("Added-Share RMSE (Lower is Better)")
    axes[1].set_title("(b) Error Reduction Across All 8 Stations")
    axes[1].legend(loc="upper right", frameon=True, fontsize=7.8)
    plt.tight_layout(w_pad=1.6)
    _save_fig(fig, fig_dir / "fig6_interpretability_and_stations.png")

    # --- Figure 7: Holdout Calibration, ROC & Trajectory/Mask Robustness (3 panels) ---
    ho_pred = pd.read_parquet(ROOT / "reports/modeling_v2/holdout_predictions.parquet")
    ho_pred["v1_sq"] = (ho_pred["pred_share_V1_RF_MaxShift"] - ho_pred["y_true"]) ** 2
    ho_pred["ens_sq"] = (ho_pred["pred_share_Ensemble_v2"] - ho_pred["y_true"]) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    axes[0].scatter(ho_pred["y_true"], ho_pred["pred_share_Ensemble_v2"], s=11, alpha=0.45, color=PALETTE[5])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.0)
    axes[0].set_xlabel("True Observed Share")
    axes[0].set_ylabel("Ensemble_v2 Predicted Share")
    axes[0].set_title(r"(a) Unseen Holdout Alignment ($R^2 = 0.848$)")

    y_bin_ho = (ho_pred["y_true"] > 0.0).astype(int)
    for m, col, label in [
        ("Physics_q15", "#888888", "Physics_q15"),
        ("V1_RF_MaxShift", PALETTE[1], "V1_RF_MaxShift"),
        ("CatBoost", PALETTE[2], "CatBoost"),
        ("CNN2D", PALETTE[0], "CNN2D"),
        ("Ensemble_v2", PALETTE[3], "Ensemble_v2"),
    ]:
        fpr, tpr, _ = roc_curve(y_bin_ho, ho_pred[f"pred_score_{m}"])
        auc = roc_auc_score(y_bin_ho, ho_pred[f"pred_score_{m}"])
        axes[1].plot(fpr, tpr, color=col, lw=2.0 if m in ("CNN2D", "Ensemble_v2") else 1.2, label=f"{label} ({auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("(b) Unseen Holdout ROC Curves")
    axes[1].legend(loc="lower right", fontsize=7.4, frameon=True)

    traj_df = ho_pred.groupby("trajectory_mode")[["v1_sq", "ens_sq"]].mean().pow(0.5).sort_values("ens_sq")
    yt = np.arange(len(traj_df))
    axes[2].barh(yt - 0.18, traj_df["v1_sq"], height=0.34, color=PALETTE[1], label="V1_RF_MaxShift")
    axes[2].barh(yt + 0.18, traj_df["ens_sq"], height=0.34, color=PALETTE[2], label="Ensemble_v2")
    axes[2].set_yticks(yt)
    axes[2].set_yticklabels(traj_df.index, fontsize=8.2)
    axes[2].set_xlim(0.0, 0.225)
    axes[2].set_ylim(-1.15, 4.55)
    axes[2].set_xlabel("Holdout RMSE")
    axes[2].set_title("(c) Holdout Error by Trajectory")
    axes[2].legend(loc="lower right", ncol=2, fontsize=7.5, frameon=True)
    plt.tight_layout(w_pad=1.4)
    _save_fig(fig, fig_dir / "fig7_holdout_diagnostics_and_test.png")

    fig_info = {}
    for p in sorted(fig_dir.glob("*.png")):
        with Image.open(p) as im:
            w_px, h_px = im.size
        fig_info[p.name] = {"bytes": p.stat().st_size, "width": w_px, "height": h_px, "sha256": _sha256(p)}
    return fig_info


def write_references_bib(path: Path) -> None:
    """Write verified APA 7th-edition BibTeX database."""
    bib = r"""@misc{zindi2026challenge,
  author       = {{Zindi}},
  year         = {2026},
  title        = {{French Embassy Underwater Vessel Noise Challenge}: Competition overview},
  howpublished = {Zindi Africa},
  url          = {https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge},
  urldate      = {2026-11-23}
}

@misc{zindi2026rules,
  author       = {{Zindi}},
  year         = {2026},
  title        = {{French Embassy Underwater Vessel Noise Challenge}: Data dictionary, starter notebook, and write-up instructions},
  howpublished = {Zindi Africa},
  url          = {https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35821},
  urldate      = {2026-11-23}
}

@dataset{yuan2026vtuad,
  author    = {Yuan, Hao and Wang, Wenbo and Li, Tian and Chen, Guici and Zeng, Lingjiang and Liu, Sixian},
  year      = {2026},
  title     = {{VTUAD (rebuilt): A 12-class vessel-type underwater acoustic corpus independently regenerated from the ONC archive}},
  version   = {1.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22274549},
  url       = {https://doi.org/10.5281/zenodo.22274549}
}

@misc{yuan2026vtuad132,
  author    = {Yuan, Hao and Wang, Wenbo and Li, Tian and Chen, Guici and Zeng, Lingjiang and Liu, Sixian},
  year      = {2026},
  title     = {{VTUAD (rebuilt)} version 1.3.2 {Zenodo} release},
  version   = {1.3.2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23037895},
  url       = {https://doi.org/10.5281/zenodo.23037895}
}

@misc{onc2026oceans,
  author       = {{Ocean Networks Canada}},
  year         = {2026},
  title        = {{Oceans 3.0 Data Portal} ({ICLISTENAF2523} {Strait of Georgia} hydrophone archive)},
  howpublished = {University of Victoria},
  url          = {https://data.oceannetworks.ca/},
  urldate      = {2026-11-23}
}

@article{shipton2025hearmyship,
  author  = {Shipton, Maya and Obradovi{\'c}, Josip and Mi{\v{s}}kovi{\'c}, Nikola and Diamant, Roee},
  year    = {2025},
  title   = {Underwater radiated noise characteristics of small vessels -- An analysis of the {HearMyShip} database},
  journal = {Marine Pollution Bulletin},
  volume  = {213},
  pages   = {117903},
  doi     = {10.1016/j.marpolbul.2025.117903}
}

@misc{hearmyship2025platform,
  author       = {{Laboratory for Underwater Systems and Technologies (LABUST, University of Zagreb FER)} and {Underwater Acoustic and Navigation Laboratory (ANL, University of Haifa)}},
  year         = {2025},
  title        = {{HearMyShip}: Acoustic and Optical Observatory Platform for Small-Vessel Underwater Radiated Noise ({EU Horizon Europe Grant 101086340})},
  url          = {https://hearmyship.fer.hr/},
  urldate      = {2026-11-23}
}

@standard{iso2017underwater,
  author       = {{International Organization for Standardization}},
  year         = {2017},
  title        = {Underwater acoustics -- Terminology ({ISO Standard No. 18405:2017})},
  organization = {International Organization for Standardization}
}

@article{wenz1962ambient,
  author  = {Wenz, Gordon M.},
  year    = {1962},
  title   = {Acoustic ambient noise in the ocean: Spectra and sources},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {34},
  number  = {12},
  pages   = {1936--1956},
  doi     = {10.1121/1.1909155}
}

@article{welch1967fft,
  author  = {Welch, Peter D.},
  year    = {1967},
  title   = {The use of fast {Fourier} transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms},
  journal = {IEEE Transactions on Audio and Electroacoustics},
  volume  = {15},
  number  = {2},
  pages   = {70--73},
  doi     = {10.1109/TAU.1967.1161901}
}

@article{merchant2015measuring,
  author  = {Merchant, Nathan D. and Fristrup, Kurt M. and Johnson, Mark P. and Tyack, Peter L. and Witt, Matthew J. and Blondel, Philippe and Parks, Susan E.},
  year    = {2015},
  title   = {Measuring acoustic habitats},
  journal = {Methods in Ecology and Evolution},
  volume  = {6},
  number  = {3},
  pages   = {257--265},
  doi     = {10.1111/2041-210X.12330}
}

@article{duarte2021soundscape,
  author  = {Duarte, Carlos M. and Chapuis, Lucille and Collin, Shaun P. and Costa, Daniel P. and Devassy, Reny P. and Eguiluz, Victor M. and Erbe, Christine and Gordon, Timothy A. C. and Halpern, Benjamin S. and Harding, Harry R. and Havlik, Michelle N. and Meekan, Mark and Merchant, Nathan D. and Miksis-Olds, Jennifer L. and Parsons, Miles and Predragovic, Milica and Radford, Andrew N. and Radford, Craig A. and Simpson, Stephen D. and Slabbekoorn, Hans and Staaterman, Erica and Van Opzeeland, Ilse C. and Winderen, Jana and Zhang, Xiangliang and Juanes, Francis},
  year    = {2021},
  title   = {The soundscape of the {Anthropocene} ocean},
  journal = {Science},
  volume  = {371},
  number  = {6529},
  pages   = {eaba4658},
  doi     = {10.1126/science.aba4658}
}

@article{breiman2001rf,
  author  = {Breiman, Leo},
  year    = {2001},
  title   = {Random forests},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  doi     = {10.1023/A:1010933404324}
}

@inproceedings{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  year      = {2016},
  title     = {{XGBoost}: A scalable tree boosting system},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  publisher = {ACM},
  doi       = {10.1145/2939672.2939785}
}

@inproceedings{ke2017lightgbm,
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  year      = {2017},
  title     = {{LightGBM}: A highly efficient gradient boosting decision tree},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {3146--3154}
}

@inproceedings{prokhorenkova2018catboost,
  author    = {Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  year      = {2018},
  title     = {{CatBoost}: Unbiased boosting with categorical features},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  pages     = {6639--6649}
}

@inproceedings{he2016resnet,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year      = {2016},
  title     = {Deep residual learning for image recognition},
  booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  publisher = {IEEE},
  doi       = {10.1109/CVPR.2016.90}
}
"""
    path.write_text(bib, encoding="utf-8")


def build_main_tex(out_path: Path) -> None:
    """Populate LaTeX manuscript dynamically with exact verified numbers from workspace artifacts."""
    st_sum = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/validated_station_summary.csv")
    boot_ci = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/run_bootstrap_intervals.csv")
    pol7 = pd.read_csv(ROOT / "reports/audits/experiment007_decision_gate_v1/policy_comparison.csv")
    bm = pd.read_csv(ROOT / "reports/modeling_v2/benchmark_metrics.csv")
    imp = pd.read_csv(ROOT / "reports/modeling_v2/feature_importance.csv")
    st_bd = pd.read_csv(ROOT / "reports/modeling_v2/station_breakdown.csv")
    shape_dist = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/shape_distance_descriptive.csv")
    weights = json.loads((ROOT / "reports/modeling_v2/ensemble_weights.json").read_text())
    sanity = json.loads((ROOT / "reports/modeling_v2/sanity_summary.json").read_text())
    v6_verif = json.loads((ROOT / "deliverables/research_submission/SUBMISSION_VERIFICATION.json").read_text())

    # Format Table 2 rows (Real Six-Station Descriptive Summary)
    lvl = boot_ci[boot_ci["metric"] == "window_mean_level"].merge(st_sum, on="station")
    med_shape_by_st = shape_dist.groupby("station")["median_shape_difference_rms_db"].median().to_dict()
    table2_rows = []
    for r in lvl.itertuples(index=False):
        st_name = _tex_esc(STATION_LABELS[r.station].replace(" (n=1)", ""))
        ci_str = (
            "Not estimable ($n=1$)"
            if pd.isna(r.run_bootstrap_low)
            else f"{r.run_bootstrap_low:.3f}--{r.run_bootstrap_high:.3f}"
        )
        rms_s = f"{med_shape_by_st[r.station]:.3f}"
        table2_rows.append(
            f"    {st_name} & {int(r.sampled_runs)} & {int(r.recorders)} & {r.median_observed_level:.3f} & {ci_str} & {r.median_temporal_band_variability:.3f} & {rms_s} \\\\"
        )

    # Format Table 3 rows (Exp 007 decision gate)
    exp7_labels = {
        ("original_only", ".5_fixed"): "Original-only (cutoff 0.50)",
        ("contrast_varied", ".5_fixed"): "Contrast-varied (cutoff 0.50)",
        ("contrast_varied", "calibration_selected"): "Contrast-varied (cal. cutoff 0.55)",
        ("hybrid", ".5_fixed"): "Hybrid (orig. det. 0.50 + varied reg.)",
    }
    table3_rows = []
    for r in pol7.itertuples(index=False):
        key = (r.policy, r.threshold_policy)
        if key not in exp7_labels:
            continue
        table3_rows.append(
            f"    {exp7_labels[key]} & {r.rmse:.6f} & {r.mae:.6f} & {r.auc:.6f} & {r.tpr_observed_positive:.5f} & {r.fpr_no_addition:.5f} \\\\"
        )

    # Format Table 4 rows (Deterministic 11-Model Benchmark)
    val_bm = bm[bm["split"] == "val"].set_index("model")
    ho_bm = bm[bm["split"] == "holdout"].set_index("model")
    model_order = [
        ("Constant_Mean", "Train fold mean", "Constant baseline"),
        ("Physics_q15", "$q=15$ quantile floor", "Analytical physics"),
        ("V1_RF_MaxShift", "96 V1 (max-shift)", "Notebook 01 RF"),
        ("Linear", "138 V2 (shift-inv.)", "Ridge + LogReg"),
        ("RF_v2", "138 V2 (shift-inv.)", "Random Forest"),
        ("CatBoost", "138 V2 (shift-inv.)", "CatBoost"),
        ("HistGBM", "138 V2 (shift-inv.)", "HistGBM"),
        ("XGBoost", "138 V2 (shift-inv.)", "XGBoost hist"),
        ("LightGBM", "138 V2 (shift-inv.)", "LightGBM (det.)"),
        ("CNN2D", "$(4,15,25)$ masked grid", "PyTorch CNN2D"),
        ("Ensemble_v2", "V2 tabular + grid", "OOF Simplex Blend"),
    ]
    table4_rows = []
    for m_key, feat_desc, fam_label in model_order:
        v = val_bm.loc[m_key]
        h = ho_bm.loc[m_key]
        bold_open = "\\textbf{" if m_key == "Ensemble_v2" else ""
        bold_close = "}" if m_key == "Ensemble_v2" else ""
        table4_rows.append(
            f"    {bold_open}{_tex_esc(m_key)}{bold_close} & {feat_desc} & "
            f"{v['zindi_score_1_minus_rmse']:.4f} & {v['rmse']:.4f} & {v['roc_auc']:.4f} & "
            f"{h['zindi_score_1_minus_rmse']:.4f} & {h['rmse']:.4f} & {h['roc_auc']:.4f} \\\\"
        )

    # Format Table 5 rows (Station Breakdown Val vs Unseen Holdout)
    piv_st = st_bd.pivot(index=["split", "station"], columns="model", values=["zindi_score_1_minus_rmse", "rmse", "roc_auc"])
    table5_rows = []
    for split_key, st_key in piv_st.sort_index(level=[0, 1]).index:
        split_label = "Holdout" if split_key == "holdout" else "Val"
        st_nice = _tex_esc(STATION_LABELS[st_key].replace(" (Holdout)", "").replace(" (n=1)", ""))
        v1_sc = piv_st.loc[(split_key, st_key), ("zindi_score_1_minus_rmse", "V1_RF_MaxShift")]
        en_sc = piv_st.loc[(split_key, st_key), ("zindi_score_1_minus_rmse", "Ensemble_v2")]
        en_rm = piv_st.loc[(split_key, st_key), ("rmse", "Ensemble_v2")]
        v1_au = piv_st.loc[(split_key, st_key), ("roc_auc", "V1_RF_MaxShift")]
        en_au = piv_st.loc[(split_key, st_key), ("roc_auc", "Ensemble_v2")]
        table5_rows.append(
            f"    {split_label} & {st_nice} & {v1_sc:.4f} & {en_sc:.4f} & {en_rm:.4f} & {v1_au:.4f} & {en_au:.4f} \\\\"
        )

    top_feats = [f"\\texttt{{{_tex_esc(f)}}} ({val:.3f})" for f, val in zip(imp["feature"].head(5), imp["mean_norm_importance"].head(5))]
    sh_w = weights["share_weights"]
    sc_w = weights["score_weights"]

    tex = rf"""\documentclass[10pt,a4paper]{{article}}

\usepackage[margin=0.68in]{{geometry}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[american]{{babel}}
\usepackage{{csquotes}}
\usepackage[style=apa, backend=biber, natbib=true]{{biblatex}}
\addbibresource{{references.bib}}

\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs,tabularx,array}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{microtype}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage{{titlesec}}
\usepackage{{hyperref}}

\definecolor{{AccentBlue}}{{HTML}}{{0072B2}}
\definecolor{{AccentOrange}}{{HTML}}{{D55E00}}
\hypersetup{{colorlinks=true,linkcolor=AccentBlue,citecolor=AccentBlue,urlcolor=AccentBlue}}
\captionsetup{{font=small,labelfont=bf,skip=4pt}}
\titlespacing*{{\section}}{{0pt}}{{10pt}}{{4pt}}
\titlespacing*{{\subsection}}{{0pt}}{{7pt}}{{3pt}}
\setlist[itemize]{{leftmargin=1.4em,itemsep=2pt,topsep=2pt}}
\setlist[enumerate]{{leftmargin=1.5em,itemsep=2pt,topsep=2pt}}

\title{{\vspace{{-1.1em}}\Large\textbf{{Acoustic Variability, Spectral Domain Shift, and Deterministic Vessel-Energy Modeling in South African Coastal Soundscapes}}}}

% TODO: Replace author name and affiliation before final compilation in Overleaf.
\author{{Research Write-Up Submission --- French Embassy Underwater Vessel Noise Challenge \\ \small Prepared for Zindi Research Evaluation Track}}
\date{{23 November 2026}}

\begin{{document}}
\maketitle
\vspace{{-1.0em}}

\begin{{abstract}}
Monitoring anthropogenic shipping pressure in biodiverse coastal waters requires distinguishing vessel-radiated acoustic energy from uncalibrated ambient background variability. Using competition telemetry from the French Embassy Underwater Vessel Noise Challenge on Zindi, I audit 1,715,779 one-minute third-octave summaries across ten South African sites (25 bands from 31.6\,Hz to 7.94\,kHz) and resolve the severe generalization gap observed when my initial baseline (0.761536 public $1 - \mathrm{{RMSE}}$, 0.517533 diagnostic AUC) encountered field mixtures. This investigation proceeds in two linked stages. First, I establish a reproducible descriptive baseline across 460 training-only recording runs from six validated sites, documenting wide inter-station source-export offsets (medians 61.520 to 90.604 uncalibrated dB), within-window variability (1.123 to 3.623\,dB), peak-referenced spectral shape distances to Ocean Networks Canada VTUAD v1.3.0 recordings (7.477 to 26.202\,dB RMS), and Experiment~007 false-alarm trade-offs (120 paired scenarios, 840 cases). Second, resolving the phase-3 band-edge mismatch and max-level distortion audit, I align external extraction from all 525 VTUAD clips and 96 HearMyShip small-craft recordings strictly to two-second Hann base-10 deci-decade periodograms, inject propagation warps, closest-point-of-approach (CPA) trajectories, and observed \texttt{{Test.csv}} missing-minute masks, and benchmark 11 deterministic model families. A leak-free, bit-exact 5-fold simplex ensemble (\texttt{{CNN2D}} weight 0.516 plus four GBDT families) lifts validation $1 - \mathrm{{RMSE}}$ from 0.8270 (legacy random forest) to {val_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} (ROC AUC {val_bm.loc['Ensemble_v2','roc_auc']:.4f}) and generalizes without collapse to completely unseen holdout stations (Fish Hoek and Sea Point, $n=541$, $1 - \mathrm{{RMSE}} = {ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f}$, ROC AUC {ho_bm.loc['Ensemble_v2','roc_auc']:.4f}). Crucially, I demonstrate why uncalibrated, asynchronously sampled total-sound summaries cannot validate a vessel-pressure map until calibration constants, verified coordinates, and clock-aligned vessel labels are provided.

\medskip
\noindent\textbf{{Keywords:}} underwater acoustics, anthropogenic ocean noise, domain shift, third-octave spectra, deterministic deep ensembles, South Africa.
\end{{abstract}}

\section{{Introduction and Research Objectives}}

Underwater radiated noise from commercial shipping and coastal small craft elevates low- and mid-frequency ambient sound levels across global marine habitats, altering acoustic communication space and foraging behavior across marine fauna \parencite{{duarte2021soundscape,wenz1962ambient}}. Passive acoustic monitoring provides continuous empirical baselines \parencite{{merchant2015measuring}}, yet automated vessel quantification remains challenging in unlabelled coastal deployments where hydrophone calibration metadata, bathymetric propagation paths, and synchronized Automatic Identification System (AIS) tracks are unavailable \parencite{{iso2017underwater}}.

The Zindi French Embassy Underwater Vessel Noise Challenge provides multi-year third-octave acoustic telemetry from South African coastal monitoring stations \parencite{{zindi2026challenge,zindi2026rules}}. On the predictive track, models estimate two bounded quantities over 15-minute $\times$ 25-band telemetry matrices: the added-vessel energy fraction \texttt{{share\_added\_vessel}} $\in [0,1]$ (ranked strictly by $1 - \mathrm{{RMSE}}$) and a diagnostic presence score \texttt{{score\_added\_vessel}} $\in [0,1]$. On the research track, evaluators assess real South African empirical grounding, uncertainty quantification, feature interpretability, transparent limitation analysis, and spatial visualization \parencite{{zindi2026rules}}.

Rigorous scientific reporting requires separating three quantities that are frequently conflated in machine-learning benchmarks: (1) \textbf{{total recorded acoustic telemetry}} exported in arbitrary recorder units, (2) \textbf{{synthetically imposed energy fractions}} defined relative to an additive external clip over observed bins, and (3) \textbf{{independently verified natural vessel traffic}}. My first leaderboard probe (Submission~001) scored 0.761536 public $1 - \mathrm{{RMSE}}$ and near-random 0.517533 diagnostic AUC despite achieving 0.112061 RMSE on clean synthetic splits, exposing structural flaws in spectral band edges, per-window max normalization, missing-minute interpolation, and unwarped training contrast.

This paper documents the end-to-end resolution of those failures and provides five concrete contributions:
\begin{{enumerate}}
    \item A training-only, run-bootstrap characterization of acoustic levels, temporal variability, and robust station anomalies across 460 South African monitoring runs.
    \item A non-interpolated spatial site-context figure anchored strictly at working candidate coordinates, paired with explicit evidentiary boundaries against mislabelling total sound as vessel pressure.
    \item Empirical diagnosis of spectral-shape mismatch across VTUAD and HearMyShip external corpora, whole-minute \texttt{{Test.csv}} dropout profiles, and false-alarm costs of naive contrast augmentation (Experiment~007).
    \item Exact alignment to starter-notebook two-second base-10 deci-decade extraction, global-offset-invariant physical features (138 dimensions invariant to $+37.5$\,dB shifts), and a bit-exact deterministic benchmark across 11 models including a masked 2D residual CNN.
    \item Generalization verification on completely unseen South African stations (Fish Hoek and Sea Point) alongside reproducible code and verification archives.
\end{{enumerate}}

\section{{Data Provenance, Integrity, and Leak-Free Partitioning}}

\subsection{{South African Monitoring Archive}}
The organizer telemetry archive contains 1,715,779 one-minute rows spanning ten coastal South African sites \parencite{{zindi2026challenge}}. Raw station tables contain 62 nominal \texttt{{TOL\_*}} columns and metadata tokens (\texttt{{site}}, \texttt{{recorder}}, \texttt{{t}}, \texttt{{dur}}, \texttt{{sample\_rate\_hz}}). Auditing across all stations confirms zero duplicate timestamps, 100.00\% non-null completeness across the 25 competition bands (31.6--7943.3\,Hz), and 17 recorder configurations with recording cadence spanning 1--360\,s. Two sites (Cape Recife with 456 runs and Riy Banks) never reach 15 continuous minutes and are excluded cleanly, leaving 88,143 non-overlapping complete 15-minute windows across 18,169 continuous runs and eight usable stations. Crucially, \texttt{{stations.csv}} designates locations as working site-level candidates (including two unresolved candidate positions at Saldanha Bay), deployment dates are omitted, and acoustic levels represent uncalibrated source exports rather than physical sound pressure levels in dB\,re\,1\,$\mu\text{{Pa}}$ \parencite{{iso2017underwater}}.

\subsection{{External Vessel Corpora and Integrity Verification}}
To construct controlled additive vessel mixtures without accessing \texttt{{Test.csv}} values, I incorporate two publicly accessible external underwater acoustic datasets:
\begin{{itemize}}
    \item \textbf{{VTUAD (rebuilt) v1.3.0:}} Reconstructed from Ocean Networks Canada hydrophone \texttt{{ICLISTENAF2523}} (Strait of Georgia, 2016--2017, mono 32\,kHz WAV) with AIS vessel associations under CC BY 4.0 \parencite{{yuan2026vtuad,onc2026oceans}}. All 525 strict-interval WAV files passed SHA-256 verification against publisher checksums. Retaining metadata labels avoids contamination: 22 WAVs inside \texttt{{vessel/}} carry background labels and are isolated alongside 16 \texttt{{background/}} files (38 total), leaving 487 active vessel clips across 11 classes (tug 48.9\% of complete minutes). My local v1.3.0 snapshot is cited distinctly from the later v1.3.2 Zenodo record \parencite{{yuan2026vtuad132}}.
    \item \textbf{{HearMyShip coastal-craft corpus:}} Coastal optical-acoustic recordings gathered by LABUST (University of Zagreb FER) and ANL (University of Haifa) under EU Horizon Europe Grant 101086340 \parencite{{shipton2025hearmyship,hearmyship2025platform}}. Because VTUAD over-represents large commercial tugs and offshore transits, I fetched a stratified sample of 96 valid mono 48\,kHz 24-bit WAV clips from the 1,149-entry catalog across eight small-craft and coastal vessel classes (Yacht: 23, Ferry: 23, Motorboat: 23, Tour boat: 21, Cargo Ship: 2, Fishing Vessel: 2, Auxiliary ship: 1, Tour Boat - Diving boat: 1; 24 Sailboat and Tour boat URLs returned HTTP 404).
\end{{itemize}}

\subsection{{Leak-Free Partitioning Design}}
Evaluating generalization to unseen acoustic environments requires preventing background run memorization, vessel hull leakage, and cross-site overfitting. Table~\ref{{tab:inventory}} summarizes both the descriptive V6 partition (\texttt{{exp004\_train}}) and the deterministic Modeling~v2 benchmark splits. Every training CV fold ($k \in \{{0,\dots,4\}}$), validation window (\texttt{{val}}), and unseen-station holdout (Fish Hoek + Sea Point) maintains strictly zero shared continuous background \texttt{{run\_id}}, zero shared VTUAD \texttt{{hull\_id}}, and zero shared HearMyShip vessel/passage group identifiers. \texttt{{Test.csv}} (1,500 rows) is touched strictly for read-only structural missingness profiling (48.3\% of rows omit whole minutes) and final frozen one-way inference.

\begin{{table}}[tbp]
\centering
\caption{{Data inventory and source-isolation audit across descriptive and predictive partitions.}}
\label{{tab:inventory}}
\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}llllr@{{}}}}
\toprule
Dataset / Partition & Temporal / Vessel Scope & Isolation Boundary & Extracted Units & Count \\
\midrule
SA Archive (Full) & 10 sites, 17 recorders & Excluded Cape Recife / Riy Banks & 1-min rows / 15-min win. & 1,715,779 / 88,143 \\
SA Descriptive V6 & 6 dev. stations (\texttt{{exp004\_train}}) & 1 window/run, cap 120/site & Indexed runs & 460 \\
VTUAD v1.3.0 & 172 active hulls + 38 bg. refs & Zero shared \texttt{{hull\_id}} & 2s min-rows / WAVs & 2,511 / 525 \\
HearMyShip & 8 coastal/small-craft categories & Grouped by passage folder & 48\,kHz 2s clips (tr/va) & 96 (75 / 21) \\
Model~v2 Train & 6 known SA stations & Disjoint 5-fold \texttt{{run\_id}} \& hulls & Synthetic windows (22.6\% zero) & 6,000 \\
Model~v2 Val & 6 known SA stations & 9--49\,d run gap; unseen hulls & Synthetic windows (23.2\% zero) & 1,500 \\
Model~v2 Holdout & Unseen Fish Hoek \& Sea Point & Unseen stations \& unseen hulls & Synthetic windows (20.0\% zero) & 541 \\
Zindi \texttt{{Test.csv}} & Unknown SA sites/periods & Strictly inference-only & Evaluation windows & 1,500 \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{Methodology: Signal Processing, Physics Simulation, and Ensembles}}

\subsection{{Reconciling Spectral Extraction: Two-Second Base-10 Deci-Decade Integration}}
Early exploratory runs (Notebook~01) applied Welch's method (\texttt{{nperseg=32768}}, 50\% overlap) with base-2 octave edges $f_c 2^{{\pm 1/6}}$ \parencite{{welch1967fft}}, creating minor bin overlap and bandwidth mismatches up to 2.6\,dB against the challenge telemetry specification. Resolving this discrepancy in Notebook~02, I match the organizer starter implementation across all 525 VTUAD and 96 HearMyShip WAVs: waveforms are segmented into non-overlapping $T_{{\mathrm{{seg}}}}=2$\,s Hann-windowed frames (\texttt{{detrend='constant'}}), converted to one-sided power spectral density $S(f)$ with bin spacing $\Delta f = 0.5$\,Hz, and integrated strictly across base-10 deci-decade bands \parencite{{zindi2026rules}}:
\begin{{equation}}
\label{{eq:decidecade}}
f_k = 10^{{1.5 + 0.1 k}}, \quad k \in \{{0,\dots,24\}}, \qquad P_k = \Delta f \sum_{{f \in [f_k 10^{{-0.05}},\, f_k 10^{{0.05}})}} S(f), \quad \text{{when }} f_k 10^{{0.05}} \le \frac{{f_s}}{{2}}.
\end{{equation}}
VTUAD segments are averaged across 30 consecutive frames per complete minute (2,511 rows), while 20\,s HearMyShip closest-point-of-approach (CPA) recordings yield exact 10-segment averages per clip. Absolute digital gain is discarded prior to mixing.

\subsection{{Additive Linear-Energy Targets Under Observed Telemetry Masks}}
Let $B_{{\mathrm{{dB}}}}(t,b)$ be an observed $15 \times 25$ South African background matrix. Rather than shifting by $\max(B_{{\mathrm{{dB}}}})$, which injects spurious level dependencies whenever loud transients appear, linear power is computed relative to a fixed constant $R_0 = 100$\,dB: $B(t,b) = 10^{{(B_{{\mathrm{{dB}}}}(t,b)-R_0)/10}}$. Notice that any constant calibration offset $C$ cancels identically in linear energy ratios and centered log features. Given a nonnegative external vessel matrix $V_0(t,b)$ scaled by $\alpha = \frac{{q \sum B}}{{(1-q) \sum V_0}}$ for target $q \in [0,1]$, the mixture is $M(t,b) = B(t,b) + \alpha V_0(t,b)$. Under whole-minute telemetry dropout mask $\Omega \subseteq \{{1,\dots,15\}} \times \{{1,\dots,25\}}$, true observed target share is recomputed strictly over observed cells:
\begin{{equation}}
\label{{eq:share_target}}
y_{{\mathrm{{obs}}}} = \frac{{\sum_{{(t,b)\in \Omega}} \alpha V_0(t,b)}}{{\sum_{{(t,b)\in \Omega}} \big(B(t,b) + \alpha V_0(t,b)\big)}} \in [0,1], \qquad X(t,b) =
\begin{{cases}}
R_0 + 10\log_{{10}} \max(M(t,b), 10^{{-30}}), & (t,b) \in \Omega, \\
\mathrm{{NaN}}, & (t,b) \notin \Omega.
\end{{cases}}
\end{{equation}}
Descriptive anomalies in Section~4.1 summarize sampled run means $x_{{is}}$ relative to each site's run median using normalized median absolute deviation: $z_{{is}} = (x_{{is}} - \operatorname{{median}}_s(x)) / (1.4826 \cdot \operatorname{{MAD}}_s(x))$, leaving Durban ($n=1$) cleanly undefined rather than zero-filled.

\subsection{{Propagation-Warped Trajectory Simulation}}
Audits in Experiments~005--007 established that naive flat insertion of Strait-of-Georgia spectra creates artificial spectral discontinuities relative to South African hydrophones. My Modeling~v2 simulator samples 22\% pure background ($y=0$) alongside positive mixtures spanning weak ($[0.003, 0.05]$), moderate ($[0.05, 0.35]$), strong ($[0.35, 0.90]$), and dominant ($[0.90, 0.992]$) regimes across four trajectory modes:
\begin{{enumerate}}
    \item \textbf{{CPA passing bell (45\%):}} Geometric spreading $g(t) \propto \big(1 + ((t-t_{{\mathrm{{cpa}}}})/w)^2\big)^{{-p}}$ ($w \in [1.2,4.2]$\,min, $p \in [0.85,1.35]$) combined with distance-dependent high-frequency absorption above 300\,Hz.
    \item \textbf{{Harbour entry/exit (27\%):}} Tapered cosine transits occupying 2--12 contiguous minutes.
    \item \textbf{{Stationary continuous shipping (18\%):}} All-15-minute presence with bounded random-walk drift ($\pm 4$\,dB), specifically targeting regimes where simple min-to-max contrast detectors fail.
    \item \textbf{{Native VTUAD progressions (10\%):}} Unmodified multi-minute field sequences when $\ge 4$ minutes are available.
\end{{enumerate}}
Every positive trajectory undergoes random propagation distortion ($\pm 6$\,dB/decade spectral tilt, cosine waveguide modal ripple up to 2.5\,dB, $\mathcal{{N}}(0,0.75)$\,dB minute flutter, and 35\% probability of $\beta \in [0.15,0.45]$ log-shape blending toward local station background color), plus structural whole-minute masking matching \texttt{{Test.csv}} probabilities (41.6\% \texttt{{m01}}, 6.7\% \texttt{{m15}}, 2.0\% interior).

\subsection{{Shift-Invariant Features, Dual-Head Architectures, and Bit-Exact Determinism}}
Because recorder calibrations across stations differ by nearly 30\,dB, every Modeling~v2 predictor depends solely on window-centered coordinates $\tilde{{X}} = X - \operatorname{{nanmedian}}_\Omega(X)$ and relative linear power $\tilde{{E}} = 10^{{\tilde{{X}}/10}}$ over observed entries $\Omega$. Adding any arbitrary scalar $+37.5$\,dB produces identical outputs to floating-point precision ($\max|\Delta f| = 2.95 \times 10^{{-14}}$). From $\tilde{{X}}$ and $\tilde{{E}}$, I engineer 138 mask-aware tabular features spanning four physical families:
\begin{{itemize}}
    \item \textbf{{Analytical multi-quantile floor baselines:}} Excess energy share $s_q = \sum_\Omega \max(\tilde{{E}} - \hat{{B}}_q, 0) / \sum_\Omega \tilde{{E}}$ and logit transforms across $q \in \{{5,10,15,20,25,35,50\}}$, where $\hat{{B}}_q(b)$ is the $q$-th temporal percentile per band (\texttt{{Physics\_q15}} uses $q=15$ directly without training), plus broadband range, tail asymmetry, and excursion fractions.
    \item \textbf{{Sub-band cavitation and machinery emergence:}} Separate shares and temporal ranges across Very Low Frequency (\texttt{{vlf}}: 31.6--100\,Hz), Low Frequency (\texttt{{lf}}: 125.9--501.2\,Hz), Mid Frequency (\texttt{{mf}}: 631--1995.3\,Hz), and High Frequency (\texttt{{hf}}: 2511.9--7943.3\,Hz), plus spectral tilt slope, crest factor, entropy, flatness, and loud-minus-quiet cosine shifts.
    \item \textbf{{Temporal morphology and co-modulation:}} 3-minute smoothed range (suppressing single-minute biologic pings), quadratic parabolic concavity ($R^2$ and $-a$ from degree-2 fits over observed time), lag-1 and lag-2 autocorrelation, and cross-band co-modulation fractions quantifying how many of the 25 bands track broadband energy ($r > 0.45$ and $r > 0.65$).
    \item \textbf{{Mask telemetry indicators:}} Observed minute fraction and \texttt{{m01}}/\texttt{{m15}} missing flags so models adjust prediction uncertainty when edge minutes are absent.
\end{{itemize}}
In parallel, a PyTorch dual-head residual network (\texttt{{AcousticCNN2D}}, adapted from residual blocks of \citealp{{he2016resnet}}) operates directly on shift-invariant masked tensors $(N, 4, 15, 25)$ whose channels encode $\tilde{{X}}/15$, band-floor anomaly $(\tilde{{X}} - \hat{{B}}_{{15}})/15$, per-minute spectral shape, and binary observation mask $M \in \{{0,1\}}$, using masked global mean/max pooling over $M$ and optimizing joint loss $\mathcal{{L}} = \mathrm{{MSE}}(\hat{{y}}, y) + 0.12\,\mathrm{{BCE}}(\hat{{z}}, \mathbb{{I}}[y>0])$.

I benchmark 11 deterministic models on identical 5-fold splits: \texttt{{Constant\_Mean}}, \texttt{{Physics\_q15}}, legacy \texttt{{V1\_RF\_MaxShift}} (96 Notebook~01 features with linear edge interpolation and per-window max subtraction), \texttt{{RF\_v2}} \parencite{{breiman2001rf}}, \texttt{{Linear}} (\texttt{{Ridge(solver='cholesky')}} + \texttt{{LogisticRegression}}), \texttt{{HistGBM}}, \texttt{{LightGBM}} \parencite{{ke2017lightgbm}}, \texttt{{XGBoost}} \parencite{{chen2016xgboost}}, \texttt{{CatBoost}} \parencite{{prokhorenkova2018catboost}}, \texttt{{CNN2D}}, and \texttt{{Ensemble\_v2}}. GBDTs and \texttt{{CNN2D}} use internal disjoint $(k+1)\bmod 5$ early stopping inside \texttt{{train}}, ensuring 100\% uncontaminated out-of-fold (OOF) predictions for fitting convex SLSQP simplex weights ($\sum_j w_j = 1, w_j \ge 0$) separately for share RMSE and log-loss logit score blending. Bit-exact determinism is locked across runs via \texttt{{PYTHONHASHSEED=42}}, single-thread BLAS/OMP settings, \texttt{{deterministic=True}} in LightGBM, and \texttt{{torch.use\_deterministic\_algorithms(True)}} on 4-thread CPU.

\section{{Empirical Results}}

\subsection{{Real Six-Station South African Acoustic Variability}}
Across the 460 training-only runs (Table~\ref{{tab:sa_summary}} and Figure~\ref{{fig:sa_acoustics}}), median run-window levels span nearly 30 uncalibrated dB between Saldanha Bay (61.520) and Bird Island (90.604), while conditional 95\% run-bootstrap intervals quantify sampling precision everywhere except single-run Durban ($n=1$). Within-window temporal variability also differs threefold (St Croix Island: 1.123\,dB vs St Francis: 3.623\,dB), confirming that single static thresholds cannot transfer uniformly across monitoring stations.

\begin{{table}}[tbp]
\centering
\caption{{Training-only six-station South African descriptive summary (source-export uncalibrated dB; 1,000-draw conditional run bootstrap, seed 42) and VTUAD median shape RMS distance.}}
\label{{tab:sa_summary}}
\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lrrrlrr@{{}}}}
\toprule
Monitoring Site & Runs & Recorders & Median Level & Conditional 95\% CI & Within-Win. Var. & VTUAD RMS Dist. (dB) \\
\midrule
{chr(10).join(table2_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig1_sa_real_acoustics.png}}
\caption{{Six-station training-only South African real-acoustic summary: (a) sampled indexed recording runs ($n=460$ total, capped at 120/site); (b) median run-window level with 95\% conditional run-bootstrap intervals (omitting Durban $n=1$); and (c) median within-window temporal-band variability.}}
\label{{fig:sa_acoustics}}
\end{{figure}}

\subsection{{Geographic Site Context and Spatial Boundaries}}
Figure~\ref{{fig:site_context}} plots the six training summaries at \texttt{{stations.csv}} working candidate coordinates without spatial interpolation and explicitly flags Saldanha Bay's unresolved secondary coordinate. Because deployments are asynchronous, uncalibrated, and unlabelled for natural vessel passages, Figure~\ref{{fig:site_context}} documents monitoring site context only and must not be interpreted as a validated shipping-noise pressure map.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.92\textwidth]{{figures/fig2_site_context_map.png}}
\caption{{Non-interpolated total-sound site context anchored at \texttt{{stations.csv}} candidate coordinates (marker area proportional to sampled runs; colour encodes median uncalibrated run-window level; red cross indicates Saldanha Bay's alternative candidate position).}}
\label{{fig:site_context}}
\end{{figure}}

\subsection{{Cross-Corpus Spectral Structure and Mismatch}}
Figure~\ref{{fig:spectral_mismatch}}a compares peak-normalized South African site spectra (which roll off steadily at 20--34\,dB from 31.6\,Hz to 7.94\,kHz) against external VTUAD and HearMyShip median vessel profiles (which maintain relatively stronger 100\,Hz--2\,kHz propulsion and small-boat cavitation plateaus). Median station-to-VTUAD shape distances range from 7.477\,dB at St Francis to 26.202\,dB at Durban (Figure~\ref{{fig:spectral_mismatch}}b), directly explaining why unwarped synthetic classifiers overfit to external sensor coloration and validating this propagation-warp strategy.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig3_spectral_structure_mismatch.png}}
\caption{{Cross-corpus spectral comparison: (a) peak-normalized South African station medians overlaid with 2s deci-decade VTUAD (525 clips) and HearMyShip (96 clips) profiles; (b) median VTUAD spectral-shape RMS distances alongside each site's P90--P10 run-level deviation spread.}}
\label{{fig:spectral_mismatch}}
\end{{figure}}

\subsection{{Missingness Profile, Representation Audits, and Experiment 007 Trade-offs}}
Read-only structural inspection of \texttt{{Test.csv}} (Figure~\ref{{fig:exp007}}a) shows 725/1,500 rows (48.3\%) contain missing cells strictly as whole-minute dropouts, concentrated at boundary minutes \texttt{{m01}} (41.6\%, 624 rows) and \texttt{{m15}} (6.7\%, 100 rows). Earlier V1 pipelines interpolated missing edges linearly and shifted by $\max(X)$; in Experiment~004 ($n=300$ frozen pairs), retaining original relative levels outperformed max-shifting across all five tested sites (RMSE 0.11443 vs 0.15089). Furthermore, Experiment~007 (Table~\ref{{tab:exp007}} and Figure~\ref{{fig:exp007}}b; 120 paired scenarios, 840 source-disjoint cases) demonstrated that contrast-varied training lowered share RMSE from 0.114236 to 0.094566 ($\sim 17.2\%$ relative point gain, though the 95\% run-bootstrap CI $[-0.00041, 0.04230]$ crossed zero) and lifted AUC from 0.848328 to 0.886421, yet nearly doubled non-injected false positives at a fixed 0.50 cutoff (0.16667 to 0.30000; threshold tuning to 0.55 lowered FPR to 0.23333).

\begin{{table}}[tbp]
\centering
\caption{{Frozen Experiment~007 source-disjoint assessment (840 synthetic cases; non-injected control FPR denominator $n=120$; observed positive TPR denominator $n=714$).}}
\label{{tab:exp007}}
\small
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Prespecified Decision Policy & Share RMSE $\downarrow$ & Share MAE $\downarrow$ & ROC AUC $\uparrow$ & Synth TPR $\uparrow$ & Control FPR $\downarrow$ \\
\midrule
{chr(10).join(table3_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig4_domain_gap_exp007.png}}
\caption{{Domain-gap diagnostics: (a) read-only \texttt{{Test.csv}} missingness concentrated at window endpoints (\texttt{{m01}}, \texttt{{m15}}); and (b) Experiment~007 trade-off showing RMSE reduction vs non-injected control false alarms under contrast augmentation.}}
\label{{fig:exp007}}
\end{{figure}}

\subsection{{Deterministic 11-Model Benchmark and Unseen-Station Generalization}}
Incorporating all architectural upgrades from my Experiments~004--007 audits into the leak-free Modeling~v2 framework yields decisive improvements across all metrics (Table~\ref{{tab:benchmark}} and Figure~\ref{{fig:benchmark}}). Notice two crucial diagnostics:
\begin{{enumerate}}
    \item \textbf{{Failure of naive analytical energy thresholding:}} Uncalibrated \texttt{{Physics\_q15}} fails badly (Val $1 - \mathrm{{RMSE}} = {val_bm.loc['Physics_q15','zindi_score_1_minus_rmse']:.4f}$, worse than \texttt{{Constant\_Mean}} {val_bm.loc['Constant_Mean','zindi_score_1_minus_rmse']:.4f}, with $\mathrm{{AUC}} = {val_bm.loc['Physics_q15','roc_auc']:.4f}$) because ambient wind swells, stationary traffic, and multi-minute dropouts distort single-quantile energy floors.
    \item \textbf{{Feature upgrade plus CNN2D--GBDT complementarity:}} Replacing Notebook~01's 96 max-shifted features (\texttt{{V1\_RF\_MaxShift}}: Val {val_bm.loc['V1_RF_MaxShift','zindi_score_1_minus_rmse']:.4f}, Holdout {ho_bm.loc['V1_RF_MaxShift','zindi_score_1_minus_rmse']:.4f}) with 138 shift-invariant mask-aware features raises random forest accuracy immediately (\texttt{{RF\_v2}}: {val_bm.loc['RF_v2','zindi_score_1_minus_rmse']:.4f} Val / {ho_bm.loc['RF_v2','zindi_score_1_minus_rmse']:.4f} Holdout). Gradient boosted trees push Val score to {val_bm.loc['LightGBM','zindi_score_1_minus_rmse']:.4f} (\texttt{{LightGBM}}), while \texttt{{CNN2D}} excels on spatial-temporal generalization, reaching {val_bm.loc['CNN2D','zindi_score_1_minus_rmse']:.4f} on Val and {ho_bm.loc['CNN2D','zindi_score_1_minus_rmse']:.4f} ($\mathrm{{ROC\ AUC}} = {ho_bm.loc['CNN2D','roc_auc']:.4f}$) on completely unseen stations (Fish Hoek and Sea Point). Blending via train-OOF simplex weights (share weights: \texttt{{CNN2D}} {sh_w['CNN2D']:.3f}, \texttt{{HistGBM}} {sh_w['HistGBM']:.3f}, \texttt{{CatBoost}} {sh_w['CatBoost']:.3f}, \texttt{{LightGBM}} {sh_w['LightGBM']:.3f}, \texttt{{XGBoost}} {sh_w['XGBoost']:.3f}; score weights: \texttt{{CNN2D}} {sc_w['CNN2D']:.3f}, \texttt{{HistGBM}} {sc_w['HistGBM']:.3f}, \texttt{{XGBoost}} {sc_w['XGBoost']:.3f}, \texttt{{Linear}} {sc_w['Linear']:.3f}) yields \texttt{{Ensemble\_v2}} at {val_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} Val ($\mathrm{{AUC}} = {val_bm.loc['Ensemble_v2','roc_auc']:.4f}$) and {ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} Unseen Holdout ($\mathrm{{AUC}} = {ho_bm.loc['Ensemble_v2','roc_auc']:.4f}$), reducing RMSE by 27.5\% relative to \texttt{{V1\_RF\_MaxShift}} on unseen sites.
\end{{enumerate}}

\begin{{table}}[tbp]
\centering
\caption{{Deterministic 11-model comparison evaluated on source-disjoint Validation (6 known sites, $n=1,500$) and Unseen-Station Holdout (Fish Hoek \& Sea Point, $n=541$). Zero \texttt{{Test.csv}} information used.}}
\label{{tab:benchmark}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
Model & Representation & Val $1-\mathrm{{RMSE}}\uparrow$ & Val RMSE $\downarrow$ & Val AUC $\uparrow$ & Holdout $1-\mathrm{{RMSE}}\uparrow$ & Holdout RMSE $\downarrow$ & Holdout AUC $\uparrow$ \\
\midrule
{chr(10).join(table4_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig5_benchmark_11_models.png}}
\caption{{Deterministic 11-model performance on known-station validation ($n=1,500$) and completely unseen holdout sites ($n=541$): (a) competition metric $1-\mathrm{{RMSE}}$ (dashed line marks legacy \texttt{{V1\_RF\_MaxShift}} holdout score 0.8390); and (b) vessel detection ROC AUC.}}
\label{{fig:benchmark}}
\end{{figure}}

\subsection{{Interpretability, Station Consistency, and Trajectory Diagnostics}}
Feature importance averaged across LightGBM, XGBoost, and CatBoost (Figure~\ref{{fig:interpretability}}a) confirms physical interpretability: the top predictors are high- and mid-frequency excess energy contributions together with temporal co-modulation and autocorrelation descriptors, specifically {", ".join(top_feats)}. Thus, trees rely simultaneously on HF/MF propulsion-cavitation emergence and coherent multi-band temporal evolution rather than recorder noise floors.

Across all eight stations individually (Table~\ref{{tab:stations}} and Figure~\ref{{fig:interpretability}}b), \texttt{{Ensemble\_v2}} cuts RMSE consistently compared to both \texttt{{Physics\_q15}} and \texttt{{V1\_RF\_MaxShift}}, achieving RMSE 0.1003 ($1-\mathrm{{RMSE}} = 0.8997$, $\mathrm{{AUC}} = 0.9604$) on unseen Sea Point and 0.1286 ($1-\mathrm{{RMSE}} = 0.8714$, $\mathrm{{AUC}} = 0.9165$) on unseen Fish Hoek. Unseen-holdout error diagnostics (Figure~\ref{{fig:holdout_diag}}) demonstrate tight calibration along the diagonal ($R^2 = {ho_bm.loc['Ensemble_v2','r2']:.3f}$), strong detection lift over legacy RF across false-positive thresholds, and lower error across every trajectory regime (\texttt{{entry\_exit}} RMSE 0.0898, \texttt{{cpa\_bell}} 0.1137, \texttt{{stationary\_drift}} 0.1516 vs legacy RF 0.1706, 0.1656, 0.1843) as well as masked ($<15$\,min) windows (RMSE 0.1237 vs 0.1770). Strictly inference-only \texttt{{Test.csv}} predictions (\texttt{{submissions/candidate\_v2\_ensemble.csv}}, $n=1,500$) span $[0.0005, 0.9394]$ ($\mathrm{{mean}} = {sanity['share_mean']:.4f}$, $\mathrm{{SD}} = {sanity['share_std']:.4f}$).

\begin{{table}}[tbp]
\centering
\caption{{Per-station performance breakdown across Validation and Unseen-Station Holdout comparing \texttt{{V1\_RF\_MaxShift}} vs deterministic \texttt{{Ensemble\_v2}}.}}
\label{{tab:stations}}
\footnotesize\setlength{{\tabcolsep}}{{4.5pt}}
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
Split & Monitoring Station & V1 $1-\mathrm{{RMSE}}$ & Ensemble $1-\mathrm{{RMSE}}\uparrow$ & Ensemble RMSE $\downarrow$ & V1 AUC & Ensemble AUC $\uparrow$ \\
\midrule
{chr(10).join(table5_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig6_interpretability_and_stations.png}}
\caption{{(a) Top-15 acoustic features ranked by mean normalized gain across LightGBM, XGBoost, and CatBoost; and (b) station-level RMSE across six validation stations and two completely unseen holdout sites.}}
\label{{fig:interpretability}}
\end{{figure}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\textwidth]{{figures/fig7_holdout_diagnostics_and_test.png}}
\caption{{Unseen-station holdout diagnostics ($n=541$): (a) predicted vs true \texttt{{share\_added\_vessel}} (\texttt{{Ensemble\_v2}}); (b) ROC comparison against baselines; and (c) holdout RMSE breakdown by simulated trajectory mode comparing legacy \texttt{{V1\_RF\_MaxShift}} and \texttt{{Ensemble\_v2}}.}}
\label{{fig:holdout_diag}}
\end{{figure}}

\section{{Validity Boundaries and Coastal Monitoring Recommendations}}
While \texttt{{Ensemble\_v2}} demonstrates strong shift-invariant generalization across unseen coastal sites, three evidentiary boundaries remain open in the released challenge data:
\begin{{enumerate}}
    \item \textbf{{Uncalibrated pressure and unverified positions:}} Without per-hydrophone end-to-end sensitivity curves (to convert source exports to calibrated $\mathrm{{dB\,re\,1\,\mu Pa}}$) and exact verified deployment coordinates, spatial plots remain descriptive telemetry summaries rather than absolute underwater sound-pressure maps.
    \item \textbf{{Absence of natural field annotations:}} Synthetic mixtures establish physical energy-share separation against real South African backgrounds, but true vessel-count or vessel-class attribution in unlabeled field hours requires clock-synchronized AIS/optical validation.
    \item \textbf{{Hidden test distribution:}} \texttt{{Test.csv}} ground truth is withheld; leaderboard confirmation of Candidate~v2 (0.8747 Val / 0.8833 Holdout) over my provisional probe (0.761536) awaits Zindi scoring after public external-data disclosure.
\end{{enumerate}}
For practical ecological monitoring across SAEON and SANBI networks, releasing per-recorder calibration tables, deployment intervals, and synchronized optical/AIS ground truth at candidate port stations would immediately convert these relative shift-invariant indicators into operational coastal noise budgets.

\section{{Reproducibility, Code Verification, and Data Compliance}}
Both research and predictive artifacts obey strict provenance and anti-leakage rules:
\begin{{itemize}}
    \item \textbf{{Clean-copy V6 research verification:}} Running \texttt{{scripts/build\_submission\_deliverables.py -{{}}-verify-only}} in an isolated temporary directory compiled all 17 archived science scripts in \texttt{{research\_code\_v6.zip}} ({v6_verif['code_archive']['bytes']} bytes, SHA-256 \texttt{{{v6_verif['code_archive']['sha256'][:16]}...}}) and regenerated Figure~2, all three Table~\ref{{tab:exp007}} CSVs, and all three Table~\ref{{tab:sa_summary}} CSVs byte-for-byte without bundling restricted audio.
    \item \textbf{{Deterministic Modeling~v2 verification:}} Dual cross-subprocess execution checks confirmed bit-exact repeatability ($\max|\Delta \hat{{y}}| = 0.0$, digest \texttt{{c58e79c022eb438d}}) across all seven learned model families under pinned dependencies (\texttt{{requirements.txt}}, 126 packages, zero AutoML). Both \texttt{{01\_south\_african\_acoustic\_eda.ipynb}} (72 cells, 15 PNGs, Finding-to-Choice log) and \texttt{{02\_vessel\_noise\_modeling.ipynb}} (22 cells, 9 PNGs, live 11-model runs) execute end-to-end with zero errors.
    \item \textbf{{Competition compliance:}} \texttt{{Test.csv}} remained strictly inference-only throughout; external-data disclosures for VTUAD v1.3.0 and HearMyShip (\texttt{{docs/EXTERNAL\_DATA\_DISCLOSURE.md}}) and my spatial-map scope clarification (\texttt{{docs/ORGANIZER\_MAP\_CLARIFICATION.md}}) accompany the submission.
\end{{itemize}}

\section{{Conclusion}}
I paired an empirically grounded, leak-free characterization of 460 South African acoustic runs with a deterministic 11-model benchmark addressing band-edge alignment, global-offset invariance, whole-minute missingness, and coastal small-craft diversity. Replacing max-shifted V1 heuristics with 2s base-10 deci-decade features, propagation-warped trajectories, and a CNN2D--GBDT ensemble elevated validation $1-\mathrm{{RMSE}}$ to {val_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} and unseen-station holdout performance to {ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} ($\mathrm{{AUC}} = {ho_bm.loc['Ensemble_v2','roc_auc']:.4f}$). At the same time, transparent separation of total recorded sound from calibrated shipping pressure ensures scientific rigor while guiding future field calibration in South African marine soundscapes.

\printbibliography[heading=bibintoc,title={{References}}]

\end{{document}}
"""
    out_path.write_text(tex, encoding="utf-8")


def write_readme_overleaf(path: Path) -> None:
    """Write concise step-by-step instructions for compiling in Overleaf."""
    txt = r"""# Overleaf Upload & Compilation Instructions

1. Upload `overleaf_research_manuscript.zip` directly in Overleaf (**New Project -> Upload Project**).
2. Set **Menu -> Compiler** to **`pdfLaTeX`** (Overleaf automatically runs **`biber`** for `biblatex` `style=apa` 7th-edition formatting).
3. In `main.tex`, update the `% TODO` `\author{...}` block with your name and affiliation before exporting the PDF for Zindi.
4. Confirm the compiled PDF renders between **4 and 10 pages** and is `< 10 MB` before uploading to Zindi separately from `research_code_v6.zip`.
"""
    path.write_text(txt, encoding="utf-8")


def validate_overleaf_package(tex_path: Path, bib_path: Path, fig_dir: Path) -> dict:
    """Strictly validate LaTeX syntax, unescaped characters, labels/refs, figures, and APA BibTeX citations."""
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    for label, txt_check in [("main.tex", tex), ("references.bib", bib)]:
        assert not any(ord(c) < 32 and c not in "\n\r\t" for c in txt_check), f"Control character found in {label}"
    if README_FILE.exists():
        readme_txt = README_FILE.read_text(encoding="utf-8")
        assert not any(ord(c) < 32 and c not in "\n\r\t" for c in readme_txt), "Control character found in README_OVERLEAF.md"

    # 1. Brace balance outside escaped \{ \} and comments
    cleaned = re.sub(r"(?<!\\)%.*", "", tex)
    cleaned_braces = cleaned.replace("\\{", "").replace("\\}", "")
    bal = 0
    for ch in cleaned_braces:
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
            assert bal >= 0, "Unmatched closing brace in main.tex"
    assert bal == 0, f"Unbalanced braces in main.tex: balance={bal}"

    # 2. Matching begin/end environments
    begins = re.findall(r"\\begin\{([a-zA-Z*]+)\}", cleaned)
    ends = re.findall(r"\\end\{([a-zA-Z*]+)\}", cleaned)
    assert len(begins) == len(ends), f"Mismatched env count: begin={len(begins)} vs end={len(ends)}"

    # 3. Check all referenced figures exist and are high-resolution PNGs
    inc_figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", cleaned)
    assert len(inc_figs) == 7, f"Expected 7 figures in main.tex, got {len(inc_figs)}"
    for rel in inc_figs:
        fp = tex_path.parent / rel
        assert fp.is_file(), f"Missing figure: {fp}"
        with Image.open(fp) as im:
            assert im.width >= 1200, f"Low-res figure {rel}: width={im.width}"

    # 4. Check all \ref and \eqref targets exist
    labels = set(re.findall(r"\\label\{([^}]+)\}", cleaned))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", cleaned))
    missing_refs = refs - labels
    assert not missing_refs, f"Unresolved LaTeX refs: {missing_refs}"

    # 5. Check all citations match references.bib AND zero uncited dead entries in bib
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited_keys = set()
    for grp in re.findall(r"\\(?:parencite|textcite|cite|citep|citet|citealt|citealp)\{([^}]+)\}", cleaned):
        for k in grp.split(","):
            cited_keys.add(k.strip())
    unknown_cites = cited_keys - bib_keys
    uncited_bib = bib_keys - cited_keys
    assert not unknown_cites, f"Unknown BibTeX keys: {unknown_cites}"
    assert not uncited_bib, f"Unused entries in references.bib: {uncited_bib}"

    # 6. Check prose lines for accidental unescaped _ or % (outside math/texttt/commands/URLs)
    in_math = False
    for ln_no, raw_ln in enumerate(cleaned.splitlines(), 1):
        line = re.sub(
            r"\\(?:texttt|url|href|includegraphics|label|ref|eqref|addbibresource|parencite|textcite|citealp)(?:\[[^\]]*\])?\{[^}]*\}",
            "",
            raw_ln,
        )
        line = re.sub(r"`[^`]+`", "", line)
        # Strip inline/display math expressions on single lines
        line = re.sub(r"\$\$[^$]*\$\$|\$[^$]*\$", "", line)
        if "\\begin{equation}" in raw_ln:
            in_math = True
            continue
        if "\\end{equation}" in raw_ln:
            in_math = False
            continue
        if in_math:
            continue
        # Remove escaped \_ and \%
        line_clean = line.replace("\\_", "").replace("\\%", "").replace("\\&", "")
        assert "_" not in line_clean, f"Unescaped underscore on line {ln_no}: {raw_ln}"
        assert "%" not in line_clean, f"Unescaped percent on line {ln_no}: {raw_ln}"

        # Enforce solo first-person voice (zero plural first-person pronouns in prose)
        line_voice = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", line_clean)
        line_voice = re.sub(r"\\[a-zA-Z*]+", " ", line_voice)
        plural_hits = re.findall(r"\b(?:we|our|ours|ourselves|us)\b", line_voice, flags=re.IGNORECASE)
        assert not plural_hits, f"First-person plural token(s) {plural_hits} on line {ln_no}: {raw_ln}"

    assert "`" not in cleaned, "Markdown backticks detected in main.tex; replace with \\texttt{...}"
    words = len(re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", cleaned).split())
    return {
        "figures_verified": len(inc_figs),
        "labels_verified": len(labels),
        "citations_verified": len(cited_keys),
        "bib_entries": len(bib_keys),
        "approx_body_words": words,
    }


def _find_tex_bin(name: str) -> str:
    """Locate pdflatex/biber on PATH or inside ~/Library/TinyTeX."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path.home() / "Library/TinyTeX/bin/universal-darwin" / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"Could not find TeX binary '{name}' on PATH or in {candidate.parent}")


def compile_local_pdf(overleaf_dir: Path, out_pdf: Path) -> dict:
    """Compile main.tex -> PDF in a temporary directory using pdflatex + biber."""
    pdflatex = _find_tex_bin("pdflatex")
    biber = _find_tex_bin("biber")

    with tempfile.TemporaryDirectory(prefix="tinytex_build_") as tmp:
        workdir = Path(tmp)
        shutil.copy2(overleaf_dir / "main.tex", workdir / "main.tex")
        shutil.copy2(overleaf_dir / "references.bib", workdir / "references.bib")
        shutil.copytree(overleaf_dir / "figures", workdir / "figures")

        steps = [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [biber, "main"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ]
        for cmd in steps:
            res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
            if res.returncode != 0:
                tail = (res.stdout + "\n" + res.stderr)[-3000:]
                raise RuntimeError(f"TeX step failed ({' '.join(cmd)}):\n{tail}")

        built_pdf = workdir / "main.pdf"
        assert built_pdf.is_file(), "Compiled main.pdf missing"

        blg_txt = (workdir / "main.blg").read_text(encoding="utf-8", errors="replace")
        blg_errors = [ln for ln in blg_txt.splitlines() if "ERROR" in ln]
        assert not blg_errors, f"Biber errors in main.blg: {blg_errors}"

        log_txt = (workdir / "main.log").read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"Citation .* undefined", log_txt), "Undefined citations detected in main.log"
        assert not re.search(r"Reference .* undefined|There were undefined references", log_txt), (
            "Undefined references detected in main.log"
        )
        overfull_hbox = len(re.findall(r"Overfull \\hbox", log_txt))

        shutil.copy2(built_pdf, out_pdf)

    reader = PdfReader(str(out_pdf))
    pdf_pages = len(reader.pages)
    pdf_bytes = out_pdf.stat().st_size
    assert 4 <= pdf_pages <= 10, f"PDF page count {pdf_pages} outside Zindi 4-10 page limit"
    assert pdf_bytes < 10_000_000, f"Compiled PDF exceeds 10 MB: {pdf_bytes}"
    return {
        "pdf_path": str(out_pdf.relative_to(ROOT)),
        "pdf_pages": pdf_pages,
        "pdf_bytes": pdf_bytes,
        "pdf_sha256": _sha256(out_pdf),
        "overfull_hbox_count": overfull_hbox,
    }


def build_zip_and_manifest(fig_info: dict, validation_info: dict, pdf_info: dict | None = None) -> dict:
    """Package Overleaf ZIP cleanly at deliverables/overleaf_manuscript/overleaf_research_manuscript.zip."""
    tmp_zip = OVERLEAF_ZIP.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [MAIN_TEX, BIB_FILE, README_FILE] + sorted(FIG_DIR.glob("*.png")):
            arc = path.relative_to(OVERLEAF_DIR)
            zf.write(path, arcname=str(arc))
    tmp_zip.replace(OVERLEAF_ZIP)

    zip_bytes = OVERLEAF_ZIP.stat().st_size
    assert zip_bytes < 10_000_000, f"Overleaf ZIP exceeds 10 MB: {zip_bytes}"
    with zipfile.ZipFile(OVERLEAF_ZIP) as zf:
        names = zf.namelist()
        assert len(names) == 10, f"Expected 10 clean ZIP entries, got {len(names)}: {names}"
        assert not any(".DS_Store" in n or "__MACOSX" in n for n in names)

    manifest = {
        "overleaf_dir": str(OVERLEAF_DIR.relative_to(ROOT)),
        "zip_path": str(OVERLEAF_ZIP.relative_to(ROOT)),
        "zip_bytes": zip_bytes,
        "zip_sha256": _sha256(OVERLEAF_ZIP),
        "zip_entries": len(names),
        "main_tex_sha256": _sha256(MAIN_TEX),
        "references_bib_sha256": _sha256(BIB_FILE),
        "validation": validation_info,
        "figures": fig_info,
    }
    if pdf_info:
        manifest.update(pdf_info)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_existing() -> dict:
    """Verify existing Overleaf assets, ZIP integrity, and LaTeX syntax without rewriting."""
    for p in (MAIN_TEX, BIB_FILE, README_FILE, OVERLEAF_ZIP, MANIFEST_JSON):
        assert p.is_file(), f"Missing Overleaf artifact: {p}"
    val_info = validate_overleaf_package(MAIN_TEX, BIB_FILE, FIG_DIR)
    rec = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    assert OVERLEAF_ZIP.stat().st_size == rec["zip_bytes"]
    assert _sha256(OVERLEAF_ZIP) == rec["zip_sha256"]
    assert _sha256(MAIN_TEX) == rec["main_tex_sha256"]
    assert _sha256(BIB_FILE) == rec["references_bib_sha256"]
    out = {"status": "PASS", "verify_only": True, **val_info, "zip_bytes": rec["zip_bytes"], "zip_sha256": rec["zip_sha256"]}
    if "pdf_path" in rec:
        pdf_p = ROOT / rec["pdf_path"]
        assert pdf_p.is_file(), f"Missing compiled PDF: {pdf_p}"
        assert pdf_p.stat().st_size == rec["pdf_bytes"]
        assert _sha256(pdf_p) == rec["pdf_sha256"]
        pages = len(PdfReader(str(pdf_p)).pages)
        assert pages == rec["pdf_pages"] and 4 <= pages <= 10
        out["pdf_pages"] = pages
        out["pdf_bytes"] = rec["pdf_bytes"]
        out["pdf_sha256"] = rec["pdf_sha256"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Verify existing Overleaf bundle without rebuilding.")
    parser.add_argument("--compile-pdf", action="store_true", help="Compile main.tex -> research_manuscript_latex.pdf locally via pdflatex + biber.")
    args = parser.parse_args()

    if args.verify_only:
        print(json.dumps(verify_existing(), indent=2))
        return

    OVERLEAF_DIR.mkdir(parents=True, exist_ok=True)
    fig_info = generate_figures(FIG_DIR)
    write_references_bib(BIB_FILE)
    build_main_tex(MAIN_TEX)
    write_readme_overleaf(README_FILE)
    val_info = validate_overleaf_package(MAIN_TEX, BIB_FILE, FIG_DIR)
    pdf_info = compile_local_pdf(OVERLEAF_DIR, PDF_FILE) if args.compile_pdf else None
    manifest = build_zip_and_manifest(fig_info, val_info, pdf_info=pdf_info)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
