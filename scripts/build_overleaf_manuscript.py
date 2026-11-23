#!/usr/bin/env python3
"""Build, validate, and package an Overleaf-ready APA 7th LaTeX research manuscript."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
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
CODE_ZIP = OVERLEAF_DIR / "research_and_modeling_code_final.zip"
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
        ("Constant_Mean", "Training-fold mean", "Constant baseline"),
        ("Physics_q15", "$q=15$ quantile floor", "Analytical baseline"),
        ("V1_RF_MaxShift", "96 max-shifted features", "Initial random forest"),
        ("Linear", "138 shift-invariant features", "Ridge + logistic"),
        ("RF_v2", "138 shift-invariant features", "Random forest"),
        ("CatBoost", "138 shift-invariant features", "CatBoost"),
        ("HistGBM", "138 shift-invariant features", "HistGBM"),
        ("XGBoost", "138 shift-invariant features", "XGBoost hist"),
        ("LightGBM", "138 shift-invariant features", "LightGBM"),
        ("CNN2D", "$(4,15,25)$ masked grid", "PyTorch 2D CNN"),
        ("Ensemble_v2", "Tabular + masked grid blend", "Simplex ensemble"),
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
        split_label = "Holdout" if split_key == "holdout" else "Validation"
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

\PassOptionsToPackage{{hyphens}}{{url}}
\usepackage[top=0.78in,bottom=0.78in,left=0.80in,right=0.80in]{{geometry}}
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
\usepackage[section]{{placeins}}
\usepackage{{xcolor}}
\usepackage[protrusion=true,expansion=true]{{microtype}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage{{titlesec}}
\usepackage{{hyperref}}

\definecolor{{AccentBlue}}{{HTML}}{{0072B2}}
\definecolor{{AccentOrange}}{{HTML}}{{D55E00}}
\hypersetup{{colorlinks=true,linkcolor=AccentBlue,citecolor=AccentBlue,urlcolor=AccentBlue}}

\linespread{{1.095}}
\setlength{{\parskip}}{{4.5pt}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.16}}
\setlength{{\textfloatsep}}{{12pt plus 2pt minus 2pt}}
\setlength{{\floatsep}}{{11pt plus 2pt minus 2pt}}
\setlength{{\intextsep}}{{11pt plus 2pt minus 2pt}}
\renewcommand{{\topfraction}}{{0.90}}
\renewcommand{{\bottomfraction}}{{0.80}}
\renewcommand{{\textfraction}}{{0.10}}
\renewcommand{{\floatpagefraction}}{{0.60}}
\setlength{{\bibitemsep}}{{2.0pt}}
\renewcommand*{{\bibfont}}{{\small}}

\AtBeginDocument{{%
  \setlength{{\abovedisplayskip}}{{10pt plus 2pt minus 2pt}}%
  \setlength{{\belowdisplayskip}}{{10pt plus 2pt minus 2pt}}%
  \setlength{{\abovedisplayshortskip}}{{7pt plus 2pt minus 1pt}}%
  \setlength{{\belowdisplayshortskip}}{{7pt plus 2pt minus 1pt}}%
}}

\tolerance=1200
\emergencystretch=2em
\hbadness=1500
\widowpenalty=10000
\clubpenalty=10000
\displaywidowpenalty=10000
\setcounter{{biburllcpenalty}}{{7000}}
\setcounter{{biburlucpenalty}}{{8000}}
\setcounter{{biburlnumpenalty}}{{9000}}
\def\UrlBreaks{{\do\/\do-\do.}}

\captionsetup{{font=small,labelfont=bf,skip=6pt,justification=justified,singlelinecheck=false}}
\titlespacing*{{\section}}{{0pt}}{{13pt plus 3pt minus 2pt}}{{5pt}}
\titlespacing*{{\subsection}}{{0pt}}{{9.5pt plus 2pt minus 1pt}}{{3.5pt}}
\setlist[itemize]{{leftmargin=1.6em,itemsep=3pt,topsep=3.5pt,parsep=0pt}}
\setlist[enumerate]{{leftmargin=1.8em,itemsep=3pt,topsep=3.5pt,parsep=0pt}}

\begin{{document}}

\begin{{center}}
{{\small\scshape\color{{AccentBlue}} Research Manuscript \enspace$\bullet$\enspace French Embassy Underwater Vessel Noise Challenge\par}}
\vspace{{0.55em}}
{{\Large\bfseries Acoustic Variability, Spectral Domain Shift, and Deterministic\\[0.22em] Vessel-Energy Modeling in South African Coastal Soundscapes\par}}
\vspace{{0.75em}}
{{\large\textbf{{Prepared by Emmanuel Kwaku Botchway}}\par}}
\vspace{{0.30em}}
{{\small Zindi Research Evaluation Track \enspace$\bullet$\enspace 23 November 2026\par}}
\end{{center}}
\vspace{{0.25em}}
\noindent\rule{{\textwidth}}{{0.5pt}}
\vspace{{0.35em}}

\begin{{center}}
{{\large\bfseries Abstract}}
\end{{center}}
\vspace{{0.15em}}

\noindent\textbf{{Objective:}} Separating anthropogenic shipping energy from ambient ocean fluctuations in uncalibrated marine recordings is difficult without sensor calibration curves or synchronized vessel logs. Using coastal acoustic telemetry from ten South African stations, I diagnose why standard spectral classifiers degrade on unseen sites and develop a shift-invariant framework for estimating added vessel energy.

\noindent\textbf{{Methods:}} I characterize 460 monitoring runs across six development sites with conditional run-bootstrap intervals, measure spectral shape differences against 525 Vessel-Type Underwater Acoustic Dataset (VTUAD) clips and 96 HearMyShip small-craft clips, and align feature extraction strictly to two-second Hann base-10 deci-decade periodograms. I then train 11 deterministic models on propagation-warped mixtures carrying observed test missingness masks and blend the strongest predictors with out-of-fold simplex weights.

\noindent\textbf{{Findings:}} Removing window-maximum shifting, aligning spectral band edges, and combining gradient-boosted trees with a masked 2D residual network (\texttt{{CNN2D}}) raises unseen-station score $1-\mathrm{{RMSE}}$ from 0.8390 (\texttt{{V1\_RF\_MaxShift}}) to {ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} (\texttt{{Ensemble\_v2}}, ROC AUC {ho_bm.loc['Ensemble_v2','roc_auc']:.4f}) across held-out Fish Hoek and Sea Point recordings ($n=541$), while lifting six-station validation score from 0.8270 to {val_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} (ROC AUC {val_bm.loc['Ensemble_v2','roc_auc']:.4f}).

\noindent\textbf{{Boundaries:}} Because the telemetry lacks per-hydrophone sensitivity constants, verified deployment coordinates, and ground-truth traffic logs, uncalibrated station summaries measure relative telemetry levels rather than a validated regional shipping-pressure map.

\noindent\textbf{{Keywords:}} underwater acoustics, anthropogenic ocean noise, spectral domain shift, third-octave spectra, deterministic ensembles, South Africa.

\vspace{{0.20em}}
\noindent\rule{{\textwidth}}{{0.5pt}}

\section{{Introduction and Research Objectives}}

Commercial shipping and coastal small craft elevate low- and mid-frequency underwater sound across global marine habitats, compressing biological communication space and altering marine animal behavior \parencite{{duarte2021soundscape,wenz1962ambient}}. Passive acoustic recorders supply long-term empirical baselines \parencite{{merchant2015measuring}}, yet automated vessel quantification remains fragile in coastal monitoring archives where hydrophone calibration curves, acoustic propagation paths, and Automatic Identification System (AIS) vessel logs are absent \parencite{{iso2017underwater}}.

The Zindi French Embassy Underwater Vessel Noise Challenge provides multi-year third-octave acoustic summaries from South African coastal stations \parencite{{zindi2026challenge,zindi2026rules}}. On the predictive track, participants estimate the added-vessel energy share \texttt{{share\_added\_vessel}} $\in [0,1]$ under the metric $1-\mathrm{{RMSE}}$ and output a vessel-presence score \texttt{{score\_added\_vessel}} $\in [0,1]$. On the research track, studies are evaluated on South African empirical evidence, uncertainty quantification, physical interpretability, limitation transparency, and spatial visualization \parencite{{zindi2026rules}}.

Sound scientific reporting separates three quantities that benchmarks often blur: total recorded sound exported in uncalibrated recorder units, synthetic energy fractions defined over observed mixture bins, and independently verified natural shipping activity. My initial leaderboard submission scored 0.761536 publicly ($1-\mathrm{{RMSE}}$) with a near-random 0.517533 diagnostic AUC despite scoring 0.112061 RMSE on clean internal splits, exposing critical mismatches in band edges, amplitude shifting, edge interpolation, and spectral contrast.

This paper resolves those failures and makes five contributions:
\begin{{enumerate}}
    \item \textbf{{Baseline characterization:}} Run-level acoustic medians, conditional bootstrap intervals, within-window temporal variability, and robust site anomalies across 460 South African monitoring runs.
    \item \textbf{{Bounded spatial reporting:}} A non-interpolated map anchored at candidate coordinates with explicit provenance safeguards against misinterpreting total telemetry as vessel noise.
    \item \textbf{{Domain-shift audits:}} Quantitative measurements of spectral-shape gaps across VTUAD and HearMyShip corpora, whole-minute test dropout geometry, and contrast-augmentation false alarms.
    \item \textbf{{Aligned deterministic modeling:}} Exact two-second base-10 deci-decade extraction, 138 global-offset-invariant features, and a reproducible 11-model benchmark including a masked 2D residual CNN.
    \item \textbf{{Out-of-site verification:}} Generalization checks across completely unseen South African stations (Fish Hoek and Sea Point) backed by verification archives.
\end{{enumerate}}

\section{{Data Provenance, Integrity, and Leak-Free Partitioning}}

\subsection{{South African Monitoring Archive}}

The competition archive contains 1,715,779 one-minute summaries spanning ten coastal South African sites \parencite{{zindi2026challenge}}. Raw station files record 62 nominal third-octave columns alongside recorder metadata and cadences ranging from 1\,s to 360\,s across 17 instrument configurations. Auditing the archive confirms zero duplicate timestamps and complete coverage across the 25 evaluation bands spanning 31.6\,Hz to 7.94\,kHz.

Two sites (Cape Recife and Riy Banks) never reach 15 continuous minutes and are excluded before windowing, leaving 88,143 complete 15-minute non-overlapping windows across 18,169 continuous runs and eight usable stations. Station tables list working site-level candidate coordinates rather than fixed hydrophone positions (including two unresolved candidate points at Saldanha Bay), omit deployment dates, and report arbitrary recorder units rather than calibrated sound pressure levels in dB\,re\,1\,$\mu\text{{Pa}}$ \parencite{{iso2017underwater}}.

\subsection{{External Vessel Corpora and Integrity Verification}}

To build controlled vessel mixtures without touching \texttt{{Test.csv}}, I draw source vessel spectra from two public underwater acoustic datasets:
\begin{{itemize}}
    \item \textbf{{VTUAD v1.3.0:}} Rebuilt from Ocean Networks Canada hydrophone \texttt{{ICLISTENAF2523}} in the Strait of Georgia (mono 32\,kHz WAV, 2016--2017) with AIS vessel associations under CC BY 4.0 \parencite{{yuan2026vtuad,onc2026oceans}}. All 525 strict-interval recordings passed SHA-256 verification against publisher checksums. Keeping metadata labels prevented label contamination because 22 clips stored in vessel folders carry background labels; isolating them alongside 16 explicit background clips yields 38 ambient references and 487 active vessel clips across 11 vessel classes (tugs represent 48.9\% of complete minutes). My local v1.3.0 copy is cited separately from the later v1.3.2 release \parencite{{yuan2026vtuad132}}.
    \item \textbf{{HearMyShip coastal small-craft corpus:}} Coastal optical-acoustic field recordings collected by LABUST (University of Zagreb FER) and ANL (University of Haifa) under EU Horizon Europe Grant 101086340 \parencite{{shipton2025hearmyship,hearmyship2025platform}}. Because VTUAD under-represents nearshore recreational craft, I downloaded a stratified sample of 96 valid mono 48\,kHz 24-bit WAV clips from the 1,149-record catalog spanning eight categories (Yacht: 23, Ferry: 23, Motorboat: 23, Tour boat: 21, Cargo Ship: 2, Fishing Vessel: 2, Auxiliary ship: 1, Diving tour boat: 1; 24 additional URLs returned HTTP 404).
\end{{itemize}}

\subsection{{Leak-Free Partitioning Design}}

Reliable generalization estimates require strict separation of continuous background runs, vessel hull identities, and coastal stations across splits. Table~\ref{{tab:inventory}} details the descriptive baseline partition and all predictive benchmark splits. Cross-validation folds ($k \in \{{0,\dots,4\}}$), the validation set (separated by 9--49 days between runs), and the unseen holdout stations share zero background \texttt{{run\_id}}, VTUAD \texttt{{hull\_id}}, or HearMyShip passage identifiers. The 1,500 evaluation windows in \texttt{{Test.csv}} remain isolated for read-only structural missingness checks and final frozen inference.

\begin{{table}}[!htbp]
\centering
\caption{{Data inventory and source-isolation boundaries across descriptive and predictive partitions.}}
\label{{tab:inventory}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllr@{{}}}}
\toprule
Dataset / Partition & Temporal or Vessel Scope & Isolation Boundary & Extracted Units & Count \\
\midrule
South African Archive & 10 sites, 17 recorders & Excludes Cape Recife \& Riy Banks & 1-minute rows / 15-minute windows & 1,715,779 / 88,143 \\
Descriptive Baseline Split & 6 development stations & 1 window per run, capped at 120/site & Indexed monitoring runs & 460 \\
VTUAD v1.3.0 & 172 active hulls + 38 bg. & Disjoint \texttt{{hull\_id}} groups & 2s minute rows / WAV files & 2,511 / 525 \\
HearMyShip & 8 coastal craft classes & Grouped by vessel passage & 48\,kHz clips (75 train / 21 val) & 96 (75 / 21) \\
Benchmark Training & 6 known SA stations & 5-fold disjoint runs and hulls & Synthetic windows (22.6\% zero) & 6,000 \\
Benchmark Validation & 6 known SA stations & 9--49 day gap; unseen hulls & Synthetic windows (23.2\% zero) & 1,500 \\
Unseen Station Holdout & Fish Hoek \& Sea Point & Unseen stations and unseen hulls & Synthetic windows (20.0\% zero) & 541 \\
Zindi Evaluation Set & Unknown SA deployments & Strictly inference-only & Evaluation windows & 1,500 \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\section{{Methodology: Spectral Extraction, Simulation, and Ensembles}}

\subsection{{Two-Second Base-10 Deci-Decade Spectral Alignment}}

My initial exploratory pipeline estimated band power via Welch's method (\texttt{{nperseg=32768}}, 50\% overlap) over base-2 third-octave boundaries $f_c 2^{{\pm 1/6}}$ \parencite{{welch1967fft}}, creating minor bin overlap and bandwidth offsets up to 2.6\,dB relative to the challenge specification. To eliminate that mismatch across all 525 VTUAD and 96 HearMyShip audio recordings, the aligned pipeline segments waveforms into non-overlapping $T_{{\mathrm{{seg}}}}=2$\,s Hann-windowed frames (\texttt{{detrend='constant'}}) with frequency resolution $\Delta f = 0.5$\,Hz, and integrates the one-sided power spectral density $S(f)$ over exact base-10 deci-decade bands \parencite{{zindi2026rules}}:
\begin{{equation}}
\label{{eq:decidecade}}
f_k = 10^{{1.5 + 0.1 k}}, \qquad
P_k = \Delta f \sum_{{f \in [f_k 10^{{-0.05}},\, f_k 10^{{0.05}})}} S(f), \qquad
k \in \{{0,\dots,24\}},
\end{{equation}}
where Equation~\eqref{{eq:decidecade}} integrates band $k$ whenever its upper edge satisfies $f_k 10^{{0.05}} \le f_s / 2$ for sampling rate $f_s$, $f_k$ is the $k$-th centre frequency in Hz, and $P_k$ is relative band power. VTUAD frames are averaged in blocks of 30 across complete minutes (2,511 minute rows), whereas 20\,s HearMyShip closest-point-of-approach (CPA) recordings yield exact 10-frame averages per clip after removing arbitrary digital gain.

\subsection{{Descriptive Site Variability and Anomaly Metrics}}

Within the six-station training baseline, sampled run-window means $x_{{is}}$ exhibit distinct site distributions and dispersion scales. To compare relative excursions without assuming Gaussian tails, Equation~\eqref{{eq:robust_z}} defines each run's site-relative robust $z$-score:
\begin{{equation}}
\label{{eq:robust_z}}
z_{{is}} = \frac{{x_{{is}} - \operatorname{{median}}_s(x)}}{{1.4826\,\mathrm{{MAD}}_s(x)}},
\end{{equation}}
where $\operatorname{{median}}_s(x)$ and $\mathrm{{MAD}}_s(x)$ are the site-specific sample median and median absolute deviation across runs $i$. Where sample dispersion cannot be estimated—specifically Durban, with a single recording run ($n=1$)—$z_{{is}}$ and conditional bootstrap confidence bounds remain explicitly undefined rather than zero-filled.

\subsection{{Additive Linear-Energy Mixtures Under Telemetry Dropouts}}

Given an observed $15 \times 25$ South African log-power background window $B_{{\mathrm{{dB}}}}(t,b)$ and a nonnegative external vessel template $V_0(t,b)$, per-window maximum shifting introduces artificial amplitude coupling whenever brief loud events occur. Instead, Equation~\eqref{{eq:energy_mix}} converts backgrounds relative to a fixed constant $R_0 = 100$\,dB, computes the scaling factor $\alpha$ for a target energy ratio $q \in [0,1]$, and sums background and vessel energies linearly:
\begin{{equation}}
\label{{eq:energy_mix}}
B(t,b) = 10^{{(B_{{\mathrm{{dB}}}}(t,b) - R_0)/10}}, \qquad
\alpha = \frac{{q \sum_{{t,b}} B(t,b)}}{{(1 - q)\sum_{{t,b}} V_0(t,b)}}, \qquad
M(t,b) = B(t,b) + \alpha V_0(t,b).
\end{{equation}}
Because structural dropouts remove entire edge minutes in the evaluation set, Equation~\eqref{{eq:share_target}} restricts target supervision strictly to observed time-frequency entries $\Omega \subseteq \{{1,\dots,15\}} \times \{{1,\dots,25\}}$ while setting missing cells to \texttt{{NaN}}:
\begin{{align}}
\label{{eq:share_target}}
y_{{\mathrm{{obs}}}} &= \frac{{\sum_{{(t,b)\in\Omega}} \alpha V_0(t,b)}}{{\sum_{{(t,b)\in\Omega}} \big(B(t,b) + \alpha V_0(t,b)\big)}} \in [0,1], \nonumber\\
X(t,b) &= \begin{{cases}}
R_0 + 10\log_{{10}}\!\max\!\big(M(t,b),\, 10^{{-30}}\big), & (t,b) \in \Omega, \\
\mathrm{{NaN}}, & (t,b) \notin \Omega.
\end{{cases}}
\end{{align}}
Notice that any constant site calibration offset $C$ cancels identically from linear ratios in Equation~\eqref{{eq:share_target}} as well as from median-centered predictors.

\subsection{{Propagation-Warped Trajectory Simulation}}

Diagnostic contrast experiments revealed that raw insertion of Strait-of-Georgia spectra creates recognizable sensor mismatches relative to South African stations. To span realistic field mixtures, the trajectory generator draws 22\% background-only windows ($y=0$) alongside positive mixtures spanning weak ($[0.003, 0.05]$), moderate ($[0.05, 0.35]$), strong ($[0.35, 0.90]$), and dominant ($[0.90, 0.992]$) shares across four temporal profiles:
\begin{{enumerate}}
    \item \textbf{{CPA passing bells (45\%):}} Geometric spreading $g(t) \propto \big(1 + ((t-t_{{\mathrm{{cpa}}}})/w)^2\big)^{{-p}}$ ($w \in [1.2, 4.2]$\,min, $p \in [0.85, 1.35]$) paired with distance-dependent high-frequency absorption above 300\,Hz.
    \item \textbf{{Harbour transits (27\%):}} Tapered cosine entry and exit ramps spanning 2 to 12 contiguous minutes.
    \item \textbf{{Stationary continuous traffic (18\%):}} Full-window presence modulated by a $\pm 4$\,dB random walk so detectors cannot rely solely on quiet-to-loud contrast within a window.
    \item \textbf{{Native multi-minute VTUAD sequences (10\%):}} Field sequences drawn directly when four or more contiguous minutes exist.
\end{{enumerate}}
Each positive template receives random propagation perturbations ($\pm 6$\,dB/decade spectral tilt, cosine modal ripple up to 2.5\,dB, $\mathcal{{N}}(0, 0.75)$\,dB minute jitter, and 35\% probability of partial blending toward local station background shape) followed by observed \texttt{{Test.csv}} missing-minute masks (41.6\% \texttt{{m01}}, 6.7\% \texttt{{m15}}, 2.0\% interior).

\subsection{{Shift-Invariant Features, Dual-Head Architectures, and Determinism}}

Because uncalibrated station offsets differ by nearly 30\,dB, every predictor operates solely on window-centered coordinates $\tilde{{X}}(t,b) = X(t,b) - \operatorname{{nanmedian}}_\Omega(X)$ over observed entries $\Omega$. Adding any arbitrary scalar $+37.5$\,dB leaves every model output identical within floating-point rounding ($\max|\Delta f| = 2.95 \times 10^{{-14}}$). From these centered matrices, Equation~\eqref{{eq:excess_share}} defines relative linear energy $\tilde{{E}}(t,b)$ and the multi-quantile excess energy shares $s_q$ for $q \in \{{5, 10, 15, 20, 25, 35, 50\}}$:
\begin{{equation}}
\label{{eq:excess_share}}
\tilde{{E}}(t,b) = 10^{{\tilde{{X}}(t,b)/10}}, \qquad
s_q = \frac{{\sum_{{(t,b)\in\Omega}} \max\!\big(\tilde{{E}}(t,b) - \hat{{B}}_q(b),\, 0\big)}}{{\sum_{{(t,b)\in\Omega}} \tilde{{E}}(t,b)}},
\end{{equation}}
where $\hat{{B}}_q(b)$ denotes the $q$-th temporal percentile in frequency band $b$ over observed minutes, and the analytical baseline \texttt{{Physics\_q15}} uses $s_{{15}}$ directly without learned parameters.

Around Equation~\eqref{{eq:excess_share}}, I assemble 138 mask-aware tabular predictors organized across four physical groups:
\begin{{itemize}}
    \item \textbf{{Quantile floors and excursions:}} Multi-quantile excess shares $s_q$, logit transforms, broadband range, upper-tail asymmetry, and above-floor excursion fractions.
    \item \textbf{{Sub-band cavitation emergence:}} Emergence shares and ranges in Very Low Frequency (31.6--100\,Hz), Low Frequency (125.9--501.2\,Hz), Mid Frequency (631--1995.3\,Hz), and High Frequency (2511.9--7943.3\,Hz) bands alongside spectral tilt, entropy, flatness, crest factor, and loud-minus-quiet cosine shifts.
    \item \textbf{{Temporal morphology and co-modulation:}} Three-minute smoothed range suppressing transient clicks, parabolic concavity ($R^2$ and curvature), lag-1 and lag-2 autocorrelation, and cross-band co-modulation fractions tracking broadband correlation ($r > 0.45$ and $r > 0.65$).
    \item \textbf{{Telemetry missingness flags:}} Observed minute fraction and boundary indicators for missing \texttt{{m01}} or \texttt{{m15}} frames.
\end{{itemize}}

Alongside tabular trees, a PyTorch dual-head residual network (\texttt{{CNN2D}}, adapted from residual blocks in \citealp{{he2016resnet}}) processes masked shift-invariant tensors $(N, 4, 15, 25)$ encoding scaled log levels, band-floor anomalies, per-minute spectral shape, and binary mask indicators. Using masked global mean and max pooling, the network optimizes the joint regression-and-detection objective in Equation~\eqref{{eq:cnn_loss}}:
\begin{{equation}}
\label{{eq:cnn_loss}}
\mathcal{{L}} = \mathrm{{MSE}}(\hat{{y}}, y_{{\mathrm{{obs}}}}) + 0.12\,\mathrm{{BCE}}\big(\hat{{z}},\, \mathbb{{I}}[y_{{\mathrm{{obs}}}} > 0]\big),
\end{{equation}}
where $\hat{{y}}$ estimates added vessel energy share, $\hat{{z}}$ is the predicted presence logit, and $\mathbb{{I}}[y_{{\mathrm{{obs}}}} > 0]$ indicates active vessel presence.

I benchmark 11 deterministic estimators on identical 5-fold splits: \texttt{{Constant\_Mean}}, \texttt{{Physics\_q15}}, the legacy max-shifted \texttt{{V1\_RF\_MaxShift}}, linear models (\texttt{{Linear}}: ridge regression and logistic regression), shift-invariant random forest (\texttt{{RF\_v2}}; \citealp{{breiman2001rf}}), histogram gradient boosting (\texttt{{HistGBM}}), \texttt{{LightGBM}} \parencite{{ke2017lightgbm}}, \texttt{{XGBoost}} \parencite{{chen2016xgboost}}, \texttt{{CatBoost}} \parencite{{prokhorenkova2018catboost}}, \texttt{{CNN2D}}, and \texttt{{Ensemble\_v2}}. Tree boosters and \texttt{{CNN2D}} use internal disjoint fold early stopping so out-of-fold predictions fit convex SLSQP simplex weights ($\sum_j w_j = 1$, $w_j \ge 0$) without validation leakage. Deterministic CPU execution locks identical outputs across independent runs.

\section{{Empirical Results}}

\subsection{{Six-Station South African Acoustic Variability}}

Across the 460 baseline monitoring runs (Table~\ref{{tab:sa_summary}} and Figure~\ref{{fig:sa_acoustics}}), median station levels span nearly 30 uncalibrated dB between Saldanha Bay (61.520) and Bird Island (90.604). Within-window temporal variability also differs by more than a factor of three (1.123\,dB at St Croix Island vs 3.623\,dB at St Francis), showing why fixed energy thresholds fail when transferred across coastal sites.

\begin{{table}}[!htbp]
\centering
\caption{{Training-only descriptive summary across six South African stations (source-export uncalibrated dB; 1,000-draw conditional run bootstrap, seed 42) and median spectral RMS distance to VTUAD.}}
\label{{tab:sa_summary}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrlrr@{{}}}}
\toprule
Monitoring Site & Runs & Recorders & Median Level & Conditional 95\% CI & Within-Window SD (dB) & VTUAD RMS Dist. (dB) \\
\midrule
{chr(10).join(table2_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig1_sa_real_acoustics.png}}
\caption{{Real South African six-station baseline ($n=460$ runs, capped at 120/site): (a) run counts per site; (b) median run-window level with 95\% conditional run-bootstrap intervals (Durban $n=1$ unestimable); and (c) median within-window temporal standard deviation.}}
\label{{fig:sa_acoustics}}
\end{{figure}}

\subsection{{Geographic Site Context and Spatial Boundaries}}

Figure~\ref{{fig:site_context}} maps the six training summaries strictly at working site-level candidate coordinates without spatial interpolation, marking Saldanha Bay's unresolved secondary coordinate explicitly. Because deployments are asynchronous across instruments, uncalibrated to physical pressure, and lack vessel-passage ground truth, this map records monitoring context only and must not be read as a shipping-pressure map.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.70\textwidth]{{figures/fig2_site_context_map.png}}
\caption{{Non-interpolated total-sound site context plotted at candidate coordinates (marker area proportional to sampled runs; colour indicates median uncalibrated level; red cross flags Saldanha Bay's unresolved alternative position).}}
\label{{fig:site_context}}
\end{{figure}}

\subsection{{Cross-Corpus Spectral Structure and Mismatch}}

Comparing peak-normalized spectra in Figure~\ref{{fig:spectral_mismatch}}a clarifies why naive classifiers struggle across regions: South African stations decline steadily by 20--34\,dB from 31.6\,Hz to 7.94\,kHz, whereas VTUAD and HearMyShip vessel clips maintain stronger propulsion and cavitation plateaus between 100\,Hz and 2\,kHz. As shown in Figure~\ref{{fig:spectral_mismatch}}b, median shape RMS distances between South African stations and VTUAD span 7.477\,dB (St Francis) to 26.202\,dB (Durban), directly motivating spectral propagation warping during synthetic training.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig3_spectral_structure_mismatch.png}}
\caption{{Cross-corpus spectral mismatch: (a) peak-normalized South African station medians overlaid with two-second deci-decade VTUAD (525 clips) and HearMyShip (96 clips) profiles; and (b) median VTUAD spectral-shape RMS distances compared against each site's P90--P10 run-level deviation spread.}}
\label{{fig:spectral_mismatch}}
\end{{figure}}

\subsection{{Missingness Profile, Amplitude Audits, and Contrast Trade-offs}}

Structural auditing of \texttt{{Test.csv}} shows that 725 of 1,500 evaluation windows (48.3\%) contain missing entries strictly as complete omitted minutes at boundary timestamps \texttt{{m01}} (41.6\%, 624 rows) and \texttt{{m15}} (6.7\%, 100 rows; Figure~\ref{{fig:exp007}}a). On 300 frozen diagnostic pairs across five sites, preserving true relative window levels reduced RMSE from 0.15089 (maximum-shifted baseline) to 0.11443.

In a separate source-disjoint synthetic experiment evaluating spectral contrast augmentation across 120 paired background-vessel configurations (840 total cases; Table~\ref{{tab:exp007}} and Figure~\ref{{fig:exp007}}b), training on contrast-varied mixtures reduced share RMSE from 0.114236 to 0.094566 and raised ROC AUC from 0.848328 to 0.886421, although the run-bootstrap 95\% CI for RMSE reduction ($[-0.00041, 0.04230]$) crossed zero and non-injected control false positives at a fixed 0.50 cutoff nearly doubled from 0.16667 to 0.30000 (calibration to cutoff 0.55 lowered false positives to 0.23333).

\begin{{table}}[!htbp]
\centering
\caption{{Source-disjoint contrast-augmentation experiment across 840 synthetic cases (non-injected control FPR $n=120$; observed positive TPR $n=714$).}}
\label{{tab:exp007}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Prespecified Decision Policy & Share RMSE $\downarrow$ & Share MAE $\downarrow$ & ROC AUC $\uparrow$ & Synth TPR $\uparrow$ & Control FPR $\downarrow$ \\
\midrule
{chr(10).join(table3_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig4_domain_gap_exp007.png}}
\caption{{Domain-gap diagnostics: (a) read-only evaluation missingness concentrated at window edges (\texttt{{m01}} and \texttt{{m15}}); and (b) synthetic experiment trade-off between RMSE improvement and non-injected false-alarm rates under contrast augmentation.}}
\label{{fig:exp007}}
\end{{figure}}

\subsection{{Deterministic 11-Model Benchmark and Unseen-Station Generalization}}

Evaluating all 11 deterministic models across source-disjoint validation ($n=1,500$) and unseen holdout stations (Fish Hoek and Sea Point, $n=541$) establishes two clear empirical results (Table~\ref{{tab:benchmark}} and Figure~\ref{{fig:benchmark}}):
\begin{{enumerate}}
    \item \textbf{{Failure of uncalibrated analytical floors:}} Direct energy thresholding via \texttt{{Physics\_q15}} performs worse than predicting the training mean (validation $1-\mathrm{{RMSE}}$ 0.5282 vs 0.6953; ROC AUC 0.5326) because wind swell, stationary ships, and boundary dropouts violate quiet-baseline assumptions.
    \item \textbf{{Shift-invariant representations and CNN--GBDT synergy:}} Moving from 96 max-shifted features (\texttt{{V1\_RF\_MaxShift}}: validation 0.8270, holdout 0.8390) to 138 shift-invariant features improves random-forest generalization immediately (\texttt{{RF\_v2}}: 0.8444 validation, 0.8556 holdout). Gradient boosters reach 0.8632 validation (\texttt{{LightGBM}}), while \texttt{{CNN2D}} leads single models on unseen sites at {ho_bm.loc['CNN2D','zindi_score_1_minus_rmse']:.4f} (AUC {ho_bm.loc['CNN2D','roc_auc']:.4f}). Convex simplex blending (share weights: \texttt{{CNN2D}} {sh_w['CNN2D']:.3f}, \texttt{{HistGBM}} {sh_w['HistGBM']:.3f}, \texttt{{CatBoost}} {sh_w['CatBoost']:.3f}, \texttt{{LightGBM}} {sh_w['LightGBM']:.3f}, \texttt{{XGBoost}} {sh_w['XGBoost']:.3f}; score weights: \texttt{{CNN2D}} {sc_w['CNN2D']:.3f}, \texttt{{HistGBM}} {sc_w['HistGBM']:.3f}, \texttt{{XGBoost}} {sc_w['XGBoost']:.3f}, \texttt{{Linear}} {sc_w['Linear']:.3f}) yields \texttt{{Ensemble\_v2}} at {val_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} validation (AUC {val_bm.loc['Ensemble_v2','roc_auc']:.4f}) and {ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} holdout (AUC {ho_bm.loc['Ensemble_v2','roc_auc']:.4f}), cutting unseen-site RMSE by 27.5\% relative to \texttt{{V1\_RF\_MaxShift}}.
\end{{enumerate}}

\begin{{table}}[!htbp]
\centering
\caption{{Deterministic 11-model benchmark evaluated on known-station validation ($n=1,500$) and unseen holdout stations (Fish Hoek and Sea Point, $n=541$) without using evaluation windows.}}
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

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig5_benchmark_11_models.png}}
\caption{{Deterministic 11-model performance on validation ($n=1,500$) and unseen holdout sites ($n=541$): (a) competition metric $1-\mathrm{{RMSE}}$ relative to legacy \texttt{{V1\_RF\_MaxShift}} holdout (0.8390 dashed line); and (b) vessel presence ROC AUC.}}
\label{{fig:benchmark}}
\end{{figure}}

\subsection{{Interpretability, Station Consistency, and Trajectory Diagnostics}}

Mean normalized gain across LightGBM, XGBoost, and CatBoost (Figure~\ref{{fig:interpretability}}a) confirms that tree predictions track genuine physical acoustics rather than recorder gain artifacts. The five highest-ranked predictors combine high- and mid-frequency emergence with temporal coherence: {", ".join(top_feats)}.

Across all eight stations individually (Table~\ref{{tab:stations}} and Figure~\ref{{fig:interpretability}}b), \texttt{{Ensemble\_v2}} lowers RMSE at every location, reaching 0.1003 ($1-\mathrm{{RMSE}} = 0.8997$, AUC 0.9604) on unseen Sea Point and 0.1286 ($1-\mathrm{{RMSE}} = 0.8714$, AUC 0.9165) on unseen Fish Hoek. Unseen-holdout error checks (Figure~\ref{{fig:holdout_diag}}) show strong alignment along the identity line ($R^2 = {ho_bm.loc['Ensemble_v2','r2']:.3f}$) and consistent RMSE gains across every vessel trajectory (\texttt{{entry\_exit}} 0.0898 vs 0.1706, \texttt{{cpa\_bell}} 0.1137 vs 0.1656, \texttt{{stationary\_drift}} 0.1516 vs 0.1843) and incomplete ($<15$-minute) windows (0.1237 vs 0.1770). Frozen predictions across the 1,500 evaluation windows span $[0.0005, 0.9394]$ ($\mathrm{{mean}} = {sanity['share_mean']:.4f}$, $\mathrm{{SD}} = {sanity['share_std']:.4f}$).

\begin{{table}}[!htbp]
\centering
\caption{{Per-station accuracy across Validation and Unseen-Station Holdout comparing the initial random forest baseline (\texttt{{V1\_RF\_MaxShift}}) against \texttt{{Ensemble\_v2}}.}}
\label{{tab:stations}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
Split & Monitoring Station & Initial RF $1-\mathrm{{RMSE}}$ & Ensemble $1-\mathrm{{RMSE}}\uparrow$ & Ensemble RMSE $\downarrow$ & Initial RF AUC & Ensemble AUC $\uparrow$ \\
\midrule
{chr(10).join(table5_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig6_interpretability_and_stations.png}}
\caption{{(a) Top-15 acoustic predictors ranked by mean normalized gain across LightGBM, XGBoost, and CatBoost; and (b) error reduction across all six validation sites and both unseen holdout stations.}}
\label{{fig:interpretability}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig7_holdout_diagnostics_and_test.png}}
\caption{{Unseen-station holdout diagnostics ($n=541$): (a) predicted versus true \texttt{{share\_added\_vessel}} for \texttt{{Ensemble\_v2}}; (b) ROC curves against baselines; and (c) holdout RMSE across simulated trajectory regimes.}}
\label{{fig:holdout_diag}}
\end{{figure}}

\section{{Validity Boundaries and Coastal Monitoring Recommendations}}

Three evidentiary boundaries remain open in the released monitoring archive and constrain ecological interpretation:
\begin{{enumerate}}
    \item \textbf{{Uncalibrated exports and approximate positions:}} Without per-hydrophone sensitivity tables to convert digital levels into $\mathrm{{dB\,re\,1\,\mu Pa}}$ alongside verified deployment coordinates, spatial plots represent descriptive site telemetry rather than physical sound-pressure fields.
    \item \textbf{{Absence of field passage labels:}} Synthetic mixtures verify energy-share separation over authentic South African backgrounds, whereas estimating true vessel counts or traffic intensity in ambient hours requires clock-synchronized AIS or optical ground truth.
    \item \textbf{{Withheld evaluation targets:}} Test labels remain hidden until public leaderboard scoring confirms whether holdout improvements ({ho_bm.loc['Ensemble_v2','zindi_score_1_minus_rmse']:.4f} vs 0.761536 initial submission) transfer identically under unseen field conditions.
\end{{enumerate}}

Releasing recorder calibration tables, exact deployment dates, and synchronized AIS or visual logs for port stations would immediately convert these shift-invariant estimates into calibrated coastal noise budgets for South African marine networks.

\section{{Reproducibility, Code Verification, and Data Compliance}}

All analyses, external datasets, and submission archives follow strict provenance and anti-leakage controls:
\begin{{itemize}}
    \item \textbf{{Unified verification archives:}} The complete runnable package \path{{research_and_modeling_code_final.zip}} (31 verified scripts, both notebooks, and LaTeX sources, $<10$\,MB, zero raw audio or competition data) complements \path{{research_code_v6.zip}} ({v6_verif['code_archive']['bytes']} bytes, SHA-256 \texttt{{{v6_verif['code_archive']['sha256'][:16]}...}}), whose 17 science scripts pass isolated clean-copy regeneration of Figure~2 and six CSV tables via \texttt{{-{{}}-verify-only}}.
    \item \textbf{{Deterministic modeling verification (Modeling~v2):}} Dual-subprocess checks confirmed bit-exact repeatability ($\max|\Delta\hat{{y}}| = 0.0$, digest \texttt{{c58e79c022eb438d}}) across all seven learned families under 126 pinned dependencies (\path{{requirements.txt}}, zero AutoML). Both \path{{notebooks/01_south_african_acoustic_eda.ipynb}} and \path{{notebooks/02_vessel_noise_modeling.ipynb}} execute end-to-end with zero errors, exporting predictions to \path{{submissions/candidate_v2_ensemble.csv}}.
    \item \textbf{{Competition compliance:}} \texttt{{Test.csv}} remained strictly inference-only throughout; public external-data disclosures for VTUAD v1.3.0 and HearMyShip (\path{{docs/EXTERNAL_DATA_DISCLOSURE.md}}) and my site-map clarification (\path{{docs/ORGANIZER_MAP_CLARIFICATION.md}}) accompany the archive.
\end{{itemize}}

\section{{Conclusion}}

By uniting a descriptive baseline across 460 South African monitoring runs with aligned two-second deci-decade extraction, propagation-warped trajectories, and global-offset-invariant modeling, this study diagnoses and overcomes severe cross-site domain shift. Shifting from maximum-normalized heuristics to a deterministic CNN--GBDT simplex ensemble lowers RMSE by 27.5\% on completely unseen South African stations while preserving explicit evidentiary boundaries around uncalibrated coastal telemetry.

\printbibliography[heading=bibintoc,title={{References}}]

\end{{document}}
"""
    out_path.write_text(tex, encoding="utf-8")


def write_readme_overleaf(path: Path) -> None:
    """Write concise step-by-step instructions for compiling in Overleaf or locally."""
    txt = r"""# Overleaf Upload & Compilation Instructions

1. Upload `overleaf_research_manuscript.zip` directly in Overleaf (**New Project -> Upload Project**).
2. Set **Menu -> Compiler** to **`pdfLaTeX`** (Overleaf automatically runs **`biber`** for `biblatex` `style=apa` 7th-edition formatting).
3. `main.tex` is already authored as `Prepared by Emmanuel Kwaku Botchway` and compiles cleanly in Overleaf or locally via `python scripts/build_overleaf_manuscript.py --compile-pdf`.
4. Confirm the exported PDF renders between **4 and 10 pages** and is `< 10 MB` before uploading to Zindi alongside `research_and_modeling_code_final.zip`.
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

    # 4. Check all \ref and \eqref targets exist (6 eqs + 5 tables + 7 figs = 18 labels)
    labels = set(re.findall(r"\\label\{([^}]+)\}", cleaned))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", cleaned))
    assert len(labels) == 18, f"Expected 18 labels (6 eq, 5 tab, 7 fig), got {len(labels)}: {sorted(labels)}"
    missing_refs = refs - labels
    unreferenced = labels - refs
    assert not missing_refs, f"Unresolved LaTeX refs: {missing_refs}"
    assert not unreferenced, f"Unreferenced LaTeX labels: {sorted(unreferenced)}"

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
            r"\\(?:texttt|path|url|href|includegraphics|label|ref|eqref|addbibresource|parencite|textcite|citealp)(?:\[[^\]]*\])?\{[^}]*\}",
            "",
            raw_ln,
        )
        line = re.sub(r"`[^`]+`", "", line)
        # Strip inline/display math expressions on single lines
        line = re.sub(r"\$\$[^$]*\$\$|\$[^$]*\$", "", line)
        if re.search(r"\\begin\{(?:equation|align)\*?\}", raw_ln):
            in_math = True
            continue
        if re.search(r"\\end\{(?:equation|align)\*?\}", raw_ln):
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
        overfull_lines = [ln.strip() for ln in log_txt.splitlines() if "Overfull \\hbox" in ln]
        overfull_hbox = len(overfull_lines)

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
        "overfull_hbox_lines": overfull_lines[:10],
    }


def verify_code_archive(zip_path: Path) -> dict:
    """Extract unified code ZIP in isolation, compile all 31 scripts, parse notebooks, and run deterministic self-test."""
    assert zip_path.is_file(), f"Missing code archive: {zip_path}"
    zip_bytes = zip_path.stat().st_size
    assert zip_bytes < 10_000_000, f"Code archive exceeds 10 MB: {zip_bytes}"

    banned_tokens = (".wav", ".parquet", ".npz", ".joblib", ".ds_store", "__macosx", "test.csv", "south_african_real_acoustics.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    for n in names:
        low = n.lower()
        assert not any(tok in low for tok in banned_tokens), f"Prohibited entry in code archive: {n}"

    with tempfile.TemporaryDirectory(prefix="code_zip_verify_") as tmp:
        workdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(workdir)

        py_files = sorted((workdir / "scripts").glob("*.py"))
        assert len(py_files) == 31, f"Expected 31 extracted scripts, got {len(py_files)}"
        for py in py_files:
            py_compile.compile(str(py), doraise=True)

        nb_files = sorted((workdir / "notebooks").glob("*.ipynb"))
        assert len(nb_files) == 2, f"Expected 2 notebooks, got {len(nb_files)}"
        for nb_p in nb_files:
            nb = json.loads(nb_p.read_text(encoding="utf-8"))
            assert len(nb.get("cells", [])) > 0, f"Empty notebook: {nb_p.name}"

        res = subprocess.run(
            [sys.executable, "scripts/modeling_v2_train.py", "--self-test-only"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0 and "PASS" in res.stdout, (
            f"Clean-copy modeling_v2_train.py --self-test-only failed:\n{res.stdout}\n{res.stderr}"
        )

    return {
        "code_zip_path": str(zip_path.relative_to(ROOT)),
        "code_zip_bytes": zip_bytes,
        "code_zip_sha256": _sha256(zip_path),
        "code_zip_entries": len(names),
        "code_zip_scripts": 31,
        "code_zip_self_test": "PASS",
    }


def build_unified_code_archive(out_zip: Path) -> dict:
    """Build unified <10 MB runnable code + notebooks + TeX archive combining V6 and Modeling v2."""
    v6_zip = ROOT / "deliverables/research_submission/research_code_v6.zip"
    assert v6_zip.is_file(), f"Missing V6 archive: {v6_zip}"
    with zipfile.ZipFile(v6_zip) as zf:
        v6_scripts = [n for n in zf.namelist() if n.startswith("scripts/") and n.endswith(".py")]

    extra_scripts = [
        "scripts/extract_external_decidecade.py",
        "scripts/extract_vtuad_band_features.py",
        "scripts/fetch_hearmyship_sample.py",
        "scripts/download_hearmyship.py",
        "scripts/build_vtuad_source_splits_v2.py",
        "scripts/apply_vtuad_source_split_v2.py",
        "scripts/verify_publisher_assets_v2.py",
        "scripts/modeling_v2_features.py",
        "scripts/modeling_v2_simulator.py",
        "scripts/modeling_v2_train.py",
        "scripts/build_modeling_notebook_02.py",
        "scripts/polish_research_notebook_01.py",
        "scripts/build_submission_deliverables.py",
        "scripts/build_overleaf_manuscript.py",
    ]
    all_script_rels = sorted(set(v6_scripts + extra_scripts))
    assert len(all_script_rels) == 31, f"Expected 31 unique scripts, got {len(all_script_rels)}"

    # Verify via AST that no local scripts/*.py dependency is missing
    available_stems = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    bundled_stems = {Path(r).stem for r in all_script_rels}
    missing_local = set()
    for rel in all_script_rels:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in available_stems and top not in bundled_stems:
                        missing_local.add(top)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in available_stems and top not in bundled_stems:
                    missing_local.add(top)
    assert not missing_local, f"Missing transitive local script imports: {sorted(missing_local)}"

    files_to_bundle: list[tuple[Path, str]] = [(ROOT / rel, rel) for rel in all_script_rels]
    files_to_bundle.extend([
        (ROOT / "README.md", "README.md"),
        (ROOT / "requirements.txt", "requirements.txt"),
        (ROOT / "docs/EXTERNAL_DATA_DISCLOSURE.md", "docs/EXTERNAL_DATA_DISCLOSURE.md"),
        (ROOT / "docs/ORGANIZER_MAP_CLARIFICATION.md", "docs/ORGANIZER_MAP_CLARIFICATION.md"),
        (ROOT / "docs/protocols/RESEARCH_METHODS_AND_DISCLOSURE_GATES.md", "docs/protocols/RESEARCH_METHODS_AND_DISCLOSURE_GATES.md"),
        (ROOT / "docs/protocols/EXPERIMENT007_PREDECLARED_PROTOCOL.md", "docs/protocols/EXPERIMENT007_PREDECLARED_PROTOCOL.md"),
        (ROOT / "notebooks/01_south_african_acoustic_eda.ipynb", "notebooks/01_south_african_acoustic_eda.ipynb"),
        (ROOT / "notebooks/02_vessel_noise_modeling.ipynb", "notebooks/02_vessel_noise_modeling.ipynb"),
        (ROOT / "deliverables/research_submission/training_only_site_aggregates.csv", "site_context/training_only_site_aggregates.csv"),
        (ROOT / "data/external/hearmyship/metadata/hearmyship_split_v1.csv", "metadata/hearmyship_split_v1.csv"),
        (MAIN_TEX, "manuscript/main.tex"),
        (BIB_FILE, "manuscript/references.bib"),
    ])

    tmp_zip = out_zip.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_bundle:
            assert src.is_file(), f"Missing file for unified code archive: {src}"
            zf.write(src, arcname=arc)
    tmp_zip.replace(out_zip)

    return verify_code_archive(out_zip)


def build_zip_and_manifest(
    fig_info: dict,
    validation_info: dict,
    code_info: dict | None = None,
    pdf_info: dict | None = None,
) -> dict:
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
    if code_info:
        manifest.update(code_info)
    if pdf_info:
        manifest.update(pdf_info)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_existing() -> dict:
    """Verify existing Overleaf assets, ZIP integrity, unified code archive, and LaTeX syntax without rewriting."""
    for p in (MAIN_TEX, BIB_FILE, README_FILE, OVERLEAF_ZIP, MANIFEST_JSON):
        assert p.is_file(), f"Missing Overleaf artifact: {p}"
    val_info = validate_overleaf_package(MAIN_TEX, BIB_FILE, FIG_DIR)
    rec = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    assert OVERLEAF_ZIP.stat().st_size == rec["zip_bytes"]
    assert _sha256(OVERLEAF_ZIP) == rec["zip_sha256"]
    assert _sha256(MAIN_TEX) == rec["main_tex_sha256"]
    assert _sha256(BIB_FILE) == rec["references_bib_sha256"]
    out = {"status": "PASS", "verify_only": True, **val_info, "zip_bytes": rec["zip_bytes"], "zip_sha256": rec["zip_sha256"]}
    if "code_zip_path" in rec:
        code_p = ROOT / rec["code_zip_path"]
        code_verif = verify_code_archive(code_p)
        assert code_verif["code_zip_bytes"] == rec["code_zip_bytes"]
        assert code_verif["code_zip_sha256"] == rec["code_zip_sha256"]
        out.update(code_verif)
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
    code_info = build_unified_code_archive(CODE_ZIP)
    pdf_info = compile_local_pdf(OVERLEAF_DIR, PDF_FILE) if args.compile_pdf else None
    manifest = build_zip_and_manifest(fig_info, val_info, code_info=code_info, pdf_info=pdf_info)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
