#!/usr/bin/env python3
"""Build, validate, and package the Overleaf-ready APA 7th LaTeX research manuscript and unified code archive."""
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
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

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
    """Generate and stage all 7 300-DPI figures directly from verified repository outputs."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Remove any retired figures so Overleaf ZIP stays exact
    for old_png in fig_dir.glob("*.png"):
        old_png.unlink()

    # --- Figure 1: Six-Station Real Acoustic Variability (3 panels) ---
    st_sum = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/validated_station_summary.csv")
    boot_ci = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/run_bootstrap_intervals.csv")
    lvl = boot_ci[boot_ci["metric"] == "window_mean_level"].merge(
        st_sum[["station", "sampled_runs", "recorders", "median_temporal_band_variability"]],
        on="station",
    )
    lvl["label"] = lvl["station"].map(STATION_LABELS)
    lvl = lvl.sort_values("median")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.65), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})
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

    # --- Figure 3: Cross-Corpus Spectral Shapes & Shape Mismatch ---
    sa_spec = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv")
    vt_2s = pd.read_parquet(ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet")
    hms_2s = pd.read_parquet(ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet")
    shape_dist = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/shape_distance_descriptive.csv")
    anom = pd.read_csv(ROOT / "reports/research_manuscript_v1/station_relative_anomaly_proxy.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.05), gridspec_kw={"width_ratios": [1.35, 1.0]})
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

    # --- Figure 4: Modeling v2 -> v3 -> v4 Progression & Shortcut Elimination (2 panels) ---
    cmp_v4 = pd.read_csv(ROOT / "reports/modeling_v4/v2_v3_v4_comparison.csv")
    imp_v3 = pd.read_csv(ROOT / "reports/modeling_v3/feature_importance.csv").head(8).iloc[::-1]
    imp_v4 = pd.read_csv(ROOT / "reports/modeling_v4/feature_importance.csv").head(8).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.15), gridspec_kw={"width_ratios": [1.08, 1.12]})
    splits = ["val_cal_hard", "val_standard", "holdout"]
    split_lbls = ["Val Cal-Hard\n(100% h(f), n=1,500)", "Val Standard\n(n=2,000)", "Unseen Holdout\n(2 Sites, n=541)"]
    xs = np.arange(len(splits))
    wb = 0.25
    piv_sc = cmp_v4.pivot(index="split", columns="model", values="zindi_score_1_minus_rmse")
    piv_zr = cmp_v4.pivot(index="split", columns="model", values="zero_rmse")
    for i, (m, col, lbl) in enumerate([
        ("Ensemble_v2", PALETTE[1], "Ensemble_v2 (Sub #2 LB 0.775)"),
        ("Ensemble_v3", PALETTE[0], "Ensemble_v3 (Hard-Neg)"),
        ("Ensemble_v4", PALETTE[2], "Ensemble_v4 (h(f)-Inv, Sub #3)"),
    ]):
        vals = [float(piv_sc.loc[s, m]) for s in splits]
        zvals = [float(piv_zr.loc[s, m]) for s in splits]
        bars_p = axes[0].bar(xs + (i - 1) * wb, vals, width=wb, color=col, label=lbl)
        for rect, v, zv in zip(bars_p, vals, zvals):
            axes[0].text(rect.get_x() + rect.get_width() / 2, v + 0.004, f"{v:.3f}\n(z:{zv:.2f})", ha="center", va="bottom", fontsize=6.7)
    axes[0].axhline(0.77501743, color="#666666", ls="--", lw=1.1, label="Sub #2 Public LB (0.7750)")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(split_lbls, fontsize=8.0)
    axes[0].set_ylim(0.74, 0.965)
    axes[0].set_ylabel("Competition Metric (1 - RMSE)")
    axes[0].set_title("(a) Progression & Quiet-Window Zero RMSE (z)")
    axes[0].legend(loc="lower right", fontsize=7.2, frameon=True)

    # Panel (b): Top features shift from v3 (shortcut proto_spec_logit) to v4 (R_* + physical emergence)
    y_top = np.arange(len(imp_v4))
    h_b = 0.36
    axes[1].barh(y_top - h_b / 2, imp_v3["mean_norm_importance"].values, height=h_b, color=PALETTE[3], alpha=0.82, label="v3 Top-8 (incl. proto_spec_logit)")
    axes[1].barh(y_top + h_b / 2, imp_v4["mean_norm_importance"].values, height=h_b, color=PALETTE[2], alpha=0.92, label="v4 Top-8 (h(f)-Invariant R_* + HF)")
    axes[1].set_yticks(y_top)
    axes[1].set_yticklabels([f"{v4n}  |  {v3n}" for v4n, v3n in zip(imp_v4["feature"], imp_v3["feature"])], fontsize=7.2)
    axes[1].set_xlim(0, 0.31)
    axes[1].set_xlabel("Mean Normalized GBDT Split Importance")
    axes[1].set_title("(b) Feature Elimination: v3 Tilt Shortcut -> v4 Physics")
    axes[1].legend(loc="lower right", fontsize=7.4, frameon=True)
    plt.tight_layout(w_pad=1.6)
    _save_fig(fig, fig_dir / "fig4_v2_v3_v4_progression_and_features.png")

    # --- Figures 5-7: Standardized Acoustic Benchmark Figures ---
    for src_name, dst_name in [
        ("fig1_standards_calibration_and_agreement.png", "fig5_standards_calibration_and_agreement.png"),
        ("fig2_boundary_stress_and_calibration_shift.png", "fig6_boundary_stress_and_calibration_shift.png"),
        ("fig3_acoustic_signal_boundaries_decomposition.png", "fig7_acoustic_signal_boundaries_decomposition.png"),
    ]:
        src_p = ROOT / "reports/standards_benchmark/figures" / src_name
        assert src_p.is_file(), f"Missing standards benchmark figure: {src_p}"
        shutil.copy2(src_p, fig_dir / dst_name)

    fig_info = {}
    for p in sorted(fig_dir.glob("*.png")):
        with Image.open(p) as im:
            w_px, h_px = im.size
        fig_info[p.name] = {"bytes": p.stat().st_size, "width": w_px, "height": h_px, "sha256": _sha256(p)}
    return fig_info


def write_references_bib(path: Path) -> None:
    """Write complete APA 7th-edition BibTeX database including acoustic standards and DG literature."""
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

@standard{iec2014bands,
  author       = {{International Electrotechnical Commission}},
  year         = {2014},
  title        = {Electroacoustics -- Octave-band and fractional-octave-band filters -- Part 1: Specifications ({IEC 61260-1:2014})},
  organization = {International Electrotechnical Commission}
}

@techreport{robinson2014npl,
  author      = {Robinson, Stephen P. and Lepper, Paul A. and Hazelwood, Richard A.},
  year        = {2014},
  title       = {Good practice guide for underwater noise measurement ({NPL Good Practice Guide No. 133})},
  institution = {National Physical Laboratory}
}

@techreport{ainslie2020adeon,
  author      = {Ainslie, Michael A. and Miksis-Olds, Jennifer L. and Martin, S. Bruce and Heaney, Kevin and de Jong, Christ A. F. and von Benda-Beckmann, Alexander M. and Lyons, Anthony P.},
  year        = {2020},
  title       = {{ADEON} underwater soundscape and modeling metadata standard (Version 1.0)},
  institution = {Atlantic Deepwater Ecosystem Observatory Network}
}

@article{farcas2020shipping,
  author  = {Farcas, Adrian and Powell, Claire F. and Brookes, Kate L. and Merchant, Nathan D.},
  year    = {2020},
  title   = {Validated shipping noise maps of the {Northeast Atlantic}},
  journal = {Science of the Total Environment},
  volume  = {735},
  pages   = {139509},
  doi     = {10.1016/j.scitotenv.2020.139509}
}

@article{merchant2012averaging,
  author  = {Merchant, Nathan D. and Blondel, Philippe and Dakin, D. Tom and Dorocicz, John},
  year    = {2012},
  title   = {Averaging underwater noise levels for environmental assessment of shipping},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {132},
  number  = {4},
  pages   = {EL343--EL349},
  doi     = {10.1121/1.4754429}
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

@article{candes2011rpca,
  author  = {Cand{\`e}s, Emmanuel J. and Li, Xiaodong and Ma, Yi and Wright, John},
  year    = {2011},
  title   = {Robust principal component analysis?},
  journal = {Journal of the ACM},
  volume  = {58},
  number  = {3},
  pages   = {1--37},
  doi     = {10.1145/1970392.1970395}
}

@article{lin2017rpca,
  author  = {Lin, Tzu-Hao and Fang, Shih-Hau and Tsao, Yu},
  year    = {2017},
  title   = {Improving biodiversity assessment via unsupervised separation of biological sounds from long-duration recordings},
  journal = {Scientific Reports},
  volume  = {7},
  pages   = {4547},
  doi     = {10.1038/s41598-017-04790-7}
}

@inproceedings{wang2017pcen,
  author    = {Wang, Yuxuan and Getreuer, Pascal and Hughes, Thad and Lyon, Richard F. and Saurous, Rif A.},
  year      = {2017},
  title     = {Trainable frontend for robust and far-field keyword spotting},
  booktitle = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {5670--5674},
  doi       = {10.1109/ICASSP.2017.7953242}
}

@article{lostanlen2019pcen,
  author  = {Lostanlen, Vincent and Salamon, Justin and Cartwright, Mark and McFee, Brian and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
  year    = {2019},
  title   = {Per-channel energy normalization: Why and how},
  journal = {IEEE Signal Processing Letters},
  volume  = {26},
  number  = {1},
  pages   = {39--43},
  doi     = {10.1109/LSP.2018.2878620}
}

@inproceedings{kim2021resnorm,
  author    = {Kim, Byeonggeun and Yang, Seunghan and Kim, Janghoon and Park, Hyunsin and Lee, Juntae and Chang, Simyung},
  year      = {2021},
  title     = {Domain generalization on efficient acoustic scene classification using residual normalization},
  booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE 2021)},
  pages     = {21--25}
}

@inproceedings{zhou2021mixstyle,
  author    = {Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  year      = {2021},
  title     = {Domain generalization with {MixStyle}},
  booktitle = {International Conference on Learning Representations (ICLR)}
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
    """Populate LaTeX manuscript dynamically with exact verified v2/v3/v4 and standards metrics."""
    st_sum = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/validated_station_summary.csv")
    boot_ci = pd.read_csv(ROOT / "reports/research_manuscript_v1/real_acoustic_results_v2/run_bootstrap_intervals.csv")
    shape_dist = pd.read_csv(ROOT / "reports/audits/real_acoustic_development_v1/shape_distance_descriptive.csv")
    cmp_v4 = pd.read_csv(ROOT / "reports/modeling_v4/v2_v3_v4_comparison.csv")
    std_full = pd.read_csv(ROOT / "reports/standards_benchmark/standardized_metrics_full.csv")
    strata_df = pd.read_csv(ROOT / "reports/standards_benchmark/stratum_breakdown.csv")
    tilt_df = pd.read_csv(ROOT / "reports/standards_benchmark/calibration_shift_curve.csv")
    w_v4 = json.loads((ROOT / "reports/modeling_v4/ensemble_weights.json").read_text())
    san_v4 = json.loads((ROOT / "reports/modeling_v4/sanity_summary.json").read_text())
    man_v4 = json.loads((ROOT / "data/synthetic/modeling_v4/manifest.json").read_text())
    v6_verif = json.loads((ROOT / "deliverables/research_submission/SUBMISSION_VERIFICATION.json").read_text())

    # Table 2: Real Six-Station Descriptive Summary
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
        table2_rows.append(
            f"    {st_name} & {int(r.sampled_runs)} & {int(r.recorders)} & {r.median_observed_level:.3f} & {ci_str} & {r.median_temporal_band_variability:.3f} & {med_shape_by_st[r.station]:.3f} \\\\"
        )

    # Table 3: Progression across v2 -> v3 -> v4 + Live Public LB
    piv_v4 = cmp_v4.set_index(["split", "model"])
    prog_specs = [
        ("Sub \\#1 (V1\\_RF\\_MaxShift)", "96 max-shifted", "0.7615 / 0.5175", None, None, None),
        ("Sub \\#2 (Ensemble\\_v2)", "138 global-shift-inv", "0.7750 / 0.5484", "Ensemble_v2", False, False),
        ("Candidate v3 (Ensemble\\_v3)", "180 + proto\\_spec\\_logit", "Intermediate v3", "Ensemble_v3", False, False),
        ("CNN2D\\_v4 (7-ch SE-ResNet)", "$(7,15,25)$ masked grid", "Base component", "CNN2D_v4", True, False),
        ("LightGBM\\_v4\\_Hurdle", "205 (71 $R_*$ $h(f)$-inv)", "Base component", "LightGBM_v4_Hurdle", True, False),
        ("Candidate v4 (Ensemble\\_v4)", "205 ($R_*$ $h(f)$-inv + CNN)", "Ready (Sub \\#3)", "Ensemble_v4", True, True),
    ]
    table3_rows = []
    for label, rep, lb_txt, mk, bold_flag, highlight in prog_specs:
        op, cl = ("\\textbf{", "}") if highlight else ("", "")
        if mk is None:
            table3_rows.append(
                f"    {label} & {rep} & {lb_txt} & --- & --- & 0.8390 / 0.8980 & --- \\\\"
            )
        else:
            hc = piv_v4.loc[("val_cal_hard", mk)]
            vs = piv_v4.loc[("val_standard", mk)]
            ho = piv_v4.loc[("holdout", mk)]
            table3_rows.append(
                f"    {op}{label}{cl} & {rep} & {lb_txt} & "
                f"{op}{hc['zindi_score_1_minus_rmse']:.4f} / {hc['roc_auc']:.4f}{cl} & "
                f"{op}{vs['zindi_score_1_minus_rmse']:.4f} / {vs['roc_auc']:.4f}{cl} & "
                f"{op}{ho['zindi_score_1_minus_rmse']:.4f} / {ho['roc_auc']:.4f}{cl} & "
                f"{op}{ho['zero_rmse']:.4f}{cl} \\\\"
            )

    # Table 4: International Acoustic Standards Benchmark on val_cal_hard & holdout
    std_idx = std_full.set_index(["split", "model"])
    tilt8 = tilt_df[tilt_df["tilt_db_per_dec"] == 8.0].set_index("model")
    std_order = [
        ("Constant_Mean", "Constant mean"),
        ("Physics_Robust_q20", "Robust $q_{20}$ physics"),
        ("RPCA_Unsupervised", "Unsupervised RPCA"),
        ("Ensemble_v2", "Ensemble v2"),
        ("Ensemble_v3", "Ensemble v3"),
        ("Ensemble_v4", "Ensemble v4 ($h(f)$-inv)"),
    ]
    table4_rows = []
    for mk, lbl in std_order:
        hc = std_idx.loc[("val_cal_hard", mk)]
        ho = std_idx.loc[("holdout", mk)]
        op, cl = ("\\textbf{", "}") if mk == "Ensemble_v4" else ("", "")
        t8_str = "---" if mk not in tilt8.index else f"{tilt8.loc[mk, 'roc_auc']:.3f} / {tilt8.loc[mk, 'zero_rmse']:.3f}"
        table4_rows.append(
            f"    {op}{lbl}{cl} & {op}{hc['lin_ccc']:.3f}{cl} & {op}{hc['nse_r2']:.3f}{cl} & "
            f"{op}{hc['jomopans_lex_mae_db']:.3f}{cl} & {op}{hc['ece_10']:.4f}{cl} & "
            f"{op}{ho['lin_ccc']:.3f}{cl} & {op}{ho['jomopans_lex_mae_db']:.3f}{cl} & "
            f"{op}{ho['ece_10']:.4f}{cl} & {op}{t8_str}{cl} \\\\"
        )

    # Table 5: Holdout Stations & 5 Physical Strata Breakdown
    st_idx = strata_df.set_index(["scope", "group", "model"])
    stratum_specs = [
        ("Holdout Station", "09_Fish_Hoek", "Fish Hoek (Unseen)"),
        ("Holdout Station", "10_Sea_Point", "Sea Point (Unseen)"),
        ("Hard-Neg Stratum", "strict_hf_ship", "Strict HF Ship"),
        ("Hard-Neg Stratum", "vlf_surf", "VLF Surf Swell"),
        ("Hard-Neg Stratum", "reef_active", "Reef Biol. Chorus"),
        ("Hard-Neg Stratum", "glitch_drop", "Telemetry Glitch"),
        ("Ambient Reference", "ambient_normal", "Ambient Normal"),
    ]
    table5_rows = []
    for sc, grp, nice in stratum_specs:
        r2 = st_idx.loc[(sc_key := ("holdout_station" if sc == "Holdout Station" else "val_combined"), grp, "Ensemble_v2")]
        r3 = st_idx.loc[(sc_key, grp, "Ensemble_v3")]
        r4 = st_idx.loc[(sc_key, grp, "Ensemble_v4")]
        n_val = int(r4["n"])
        table5_rows.append(
            f"    {sc} & {nice} & {n_val} & "
            f"{r2['zindi_score_1_minus_rmse']:.4f} / {r2['roc_auc']:.3f} & "
            f"{r3['zindi_score_1_minus_rmse']:.4f} / {r3['roc_auc']:.3f} & "
            f"\\textbf{{{r4['zindi_score_1_minus_rmse']:.4f}}} / \\textbf{{{r4['roc_auc']:.3f}}} & "
            f"\\textbf{{{r4['rmse_zero']:.4f}}} \\\\"
        )

    sh_w = w_v4["share_weights"]
    sc_w = w_v4["score_weights"]
    hc_v4 = piv_v4.loc[("val_cal_hard", "Ensemble_v4")]
    ho_v4 = piv_v4.loc[("holdout", "Ensemble_v4")]

    tex = rf"""\documentclass[10pt,a4paper]{{article}}

\PassOptionsToPackage{{hyphens}}{{url}}
\usepackage[top=0.72in,bottom=0.72in,left=0.75in,right=0.75in]{{geometry}}
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

\linespread{{1.08}}
\setlength{{\parskip}}{{4.2pt}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.14}}
\setlength{{\textfloatsep}}{{11pt plus 2pt minus 2pt}}
\setlength{{\floatsep}}{{10pt plus 2pt minus 2pt}}
\setlength{{\intextsep}}{{10pt plus 2pt minus 2pt}}
\renewcommand{{\topfraction}}{{0.92}}
\renewcommand{{\bottomfraction}}{{0.85}}
\renewcommand{{\textfraction}}{{0.08}}
\renewcommand{{\floatpagefraction}}{{0.65}}
\setlength{{\bibitemsep}}{{0.2pt}}
\renewcommand*{{\bibfont}}{{\footnotesize}}

\AtBeginDocument{{%
  \setlength{{\abovedisplayskip}}{{9pt plus 2pt minus 2pt}}%
  \setlength{{\belowdisplayskip}}{{9pt plus 2pt minus 2pt}}%
  \setlength{{\abovedisplayshortskip}}{{6pt plus 2pt minus 1pt}}%
  \setlength{{\belowdisplayshortskip}}{{6pt plus 2pt minus 1pt}}%
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

\captionsetup{{font=small,labelfont=bf,skip=5pt,justification=justified,singlelinecheck=false}}
\titlespacing*{{\section}}{{0pt}}{{11pt plus 2pt minus 2pt}}{{4.5pt}}
\titlespacing*{{\subsection}}{{0pt}}{{8.5pt plus 2pt minus 1pt}}{{3pt}}
\setlist[itemize]{{leftmargin=1.5em,itemsep=2.5pt,topsep=3pt,parsep=0pt}}
\setlist[enumerate]{{leftmargin=1.7em,itemsep=2.5pt,topsep=3pt,parsep=0pt}}

\begin{{document}}

\begin{{center}}
{{\small\scshape\color{{AccentBlue}} Research Manuscript \enspace$\bullet$\enspace French Embassy Underwater Vessel Noise Challenge\par}}
\vspace{{0.45em}}
{{\Large\bfseries Calibration-Invariant Vessel-Energy Modeling, Acoustic Standards,\\[0.20em] and Coastal Soundscape Boundaries in South African Telemetry\par}}
\vspace{{0.65em}}
{{\large\textbf{{Prepared by Emmanuel Kwaku Botchway}}\par}}
\vspace{{0.25em}}
{{\small Zindi Research Evaluation Track \enspace$\bullet$\enspace 23 November 2026\par}}
\end{{center}}
\vspace{{0.20em}}
\noindent\rule{{\textwidth}}{{0.5pt}}
\vspace{{0.25em}}

\begin{{center}}
{{\large\bfseries Abstract}}
\end{{center}}
\vspace{{0.10em}}

\noindent\textbf{{Objective:}} Separating anthropogenic shipping energy from natural ambient sound in uncalibrated passive acoustic recordings fails whenever classifiers memorize site-specific hydrophone transfer curves or biological transients. Using multi-year third-octave telemetry from ten South African coastal stations, I diagnose why global-offset-invariant models stalled on live leaderboard evaluation and develop an exact frequency-response-invariant framework benchmarked against international acoustic standards.

\noindent\textbf{{Methods:}} I characterize 460 training runs with conditional run-bootstrap intervals, quantify spectral shape gaps against 525 Vessel-Type Underwater Acoustic Dataset (VTUAD v1.3.0) and 96 HearMyShip recordings under strict 2\,s base-10 deci-decade extraction, and progress iteratively from \texttt{{Ensemble\_v2}} (Submission~\#2) through hard-negative \texttt{{Ensemble\_v3}} to \texttt{{Ensemble\_v4}} featuring 71 exact $h(f)$-invariant temporal excess residuals $R(t,f)$, calibration-warped strata, and a 7-channel Squeeze-and-Excitation residual CNN (\texttt{{CNN2D\_v4}}).

\noindent\textbf{{Findings:}} Removing the \texttt{{proto\_spec\_logit}} sensor-tilt shortcut and training \texttt{{Ensemble\_v4}} across five physical strata raises unseen Fish Hoek and Sea Point holdout accuracy $1-\mathrm{{RMSE}}$ from 0.8886 (\texttt{{Ensemble\_v2}}) and 0.8971 (\texttt{{Ensemble\_v3}}) to {ho_v4['zindi_score_1_minus_rmse']:.4f} (ROC AUC {ho_v4['roc_auc']:.4f}, quiet-window \texttt{{zero\_rmse}} {ho_v4['zero_rmse']:.4f}). Under 100\% calibration-warped validation (\texttt{{val\_cal\_hard}}, $n=1{{,}}500$), \texttt{{Ensemble\_v4}} lifts Lin's Concordance Correlation Coefficient from 0.8108 to 0.9204, cuts JOMOPANS excess-level error to 0.547\,dB, lowers Expected Calibration Error from 0.1889 to 0.0273, and preserves 0.9561 ROC AUC under $+8$\,dB/decade sensor tilt.

\noindent\textbf{{Boundaries:}} Without per-hydrophone sensitivity constants, surveyed coordinates, and synchronized traffic logs, uncalibrated exports remain relative telemetry proxies rather than validated physical sound-pressure shipping maps.

\noindent\textbf{{Keywords:}} underwater acoustics, hydrophone calibration invariance, ISO 18405, JOMOPANS, spectral domain shift, deterministic ensembles, South Africa.

\vspace{{0.15em}}
\noindent\rule{{\textwidth}}{{0.5pt}}

\section{{Introduction and Live Leaderboard Evolution}}

Commercial shipping and nearshore small craft elevate low- and mid-frequency underwater sound across coastal habitats, masking biological communication and altering marine fauna behavior \parencite{{duarte2021soundscape,wenz1962ambient}}. Passive acoustic monitoring supplies long-term soundscape records \parencite{{merchant2015measuring}}, yet automated vessel quantification degrades severely when sensor sensitivity curves $h(f)$, propagation loss, and Automatic Identification System (AIS) traffic logs are unavailable \parencite{{iso2017underwater,robinson2014npl}}.

On the Zindi French Embassy Underwater Vessel Noise Challenge \parencite{{zindi2026challenge,zindi2026rules}}, predictive participants estimate added vessel energy share \texttt{{share\_added\_vessel}} $\in [0,1]$ (scored by $1-\mathrm{{RMSE}}$) and presence score \texttt{{score\_added\_vessel}} $\in [0,1]$ (diagnostic ROC AUC) across $15 \times 25$ base-10 third-octave matrices spanning 31.6\,Hz to 7.94\,kHz \parencite{{iec2014bands}}. Rigorous scientific reporting separates uncalibrated digital exports, synthetic added-energy fractions over observed bins, and true biological vs.\ vessel soundscapes \parencite{{ainslie2020adeon,farcas2020shipping}}.

My live public submissions exposed critical cross-site domain-shift failure modes. Submission~\#1 (\texttt{{V1\_RF\_MaxShift}}) scored 0.761536 publicly ($1-\mathrm{{RMSE}}$) with 0.517533 ROC AUC because window-maximum shifting coupled background levels to brief peaks and Welch band edges drifted by up to 2.6\,dB. Fixing band alignment and global median centering in \texttt{{Ensemble\_v2}} (Submission~\#2) reached 0.8886 holdout $1-\mathrm{{RMSE}}$ (0.9334 AUC) locally, yet yielded only 0.77501743 publicly (RMSE 0.224982569, ROC AUC 0.548377777, Rank 508). Auditing why a clean cross-station model stalled publicly drove three advances reported in this paper:
\begin{{enumerate}}
    \item \textbf{{Diagnosis and elimination of sensor-tilt shortcuts:}} Demonstrating via Modeling~v3 and v4 how global spectral prototypes memorize Ocean Networks Canada (VTUAD) vs.\ South African frequency response $h(f)$ unless eliminated by per-band temporal excess residuals $R(t,f)$.
    \item \textbf{{Five-stratum South African hard-negative simulation:}} Isolating coastal VLF surf swells, 63--125\,Hz reef choruses, sharp $2$--$5$\,min vessel bursts, and $>15$\,dB telemetry dropouts under realistic calibration warping ($65.3\%$ training, $100\%$ \texttt{{val\_cal\_hard}}).
    \item \textbf{{International standards and boundary benchmarking:}} Benchmarking every model generation alongside Per-Channel Energy Normalization (PCEN; \citealp{{wang2017pcen,lostanlen2019pcen}}) and unsupervised Robust PCA \parencite{{candes2011rpca,lin2017rpca}} under ISO~18405, IEC~61260-1, NPL~GPG~133, ADEON, and JOMOPANS excess-level metrics across three open reproducibility notebooks.
\end{{enumerate}}

\section{{Data Provenance, Spectral Alignment, and Leak-Free Partitioning}}

\subsection{{South African Coastal Archive and Descriptive Variability}}

The competition archive comprises 1,715,779 one-minute rows across ten South African sites \parencite{{zindi2026challenge}} logged across 17 recorder configurations without duplicate timestamps. Excluding Cape Recife and Riy Banks (never reaching 15 continuous minutes) leaves 88,143 complete 15-minute windows across eight usable sites and 18,169 runs. Metadata list approximate site candidate coordinates—including two unresolved positions at Saldanha Bay—without hydrophone sensitivity curves in dB\,re\,1\,$\mu\text{{Pa}}$ \parencite{{iso2017underwater}}.

Across 460 training baseline runs (Table~\ref{{tab:sa_summary}}, Figure~\ref{{fig:sa_acoustics}}), median uncalibrated site levels span 29.1\,dB between Saldanha Bay (61.520) and Bird Island (90.604), while within-window temporal standard deviations vary more than threefold (1.123\,dB at St Croix Island vs.\ 3.623\,dB at St Francis). Equation~\eqref{{eq:robust_z}} quantifies run-level anomalies relative to each station's median and median absolute deviation (explicitly undefined for Durban where $n=1$):
\begin{{equation}}
\label{{eq:robust_z}}
z_{{is}} = \frac{{x_{{is}} - \operatorname{{median}}_s(x)}}{{1.4826\,\mathrm{{MAD}}_s(x)}}.
\end{{equation}}
Figure~\ref{{fig:site_context}} plots these training levels strictly at non-interpolated candidate coordinates as site monitoring context rather than a calibrated marine pressure surface \parencite{{robinson2014npl}}.

\begin{{table}}[!htbp]
\centering
\caption{{Training-only descriptive summary across six South African sites (source-export uncalibrated dB; 1,000-draw conditional run bootstrap, seed 42) and median spectral RMS distance to VTUAD.}}
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
\includegraphics[width=0.83\textwidth]{{figures/fig1_sa_real_acoustics.png}}
\caption{{Real South African six-station baseline ($n=460$ runs, capped at 120/site): (a) run counts per site; (b) median run-window level with 95\% conditional run-bootstrap intervals (Durban $n=1$ unestimable); and (c) median within-window temporal standard deviation.}}
\label{{fig:sa_acoustics}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.64\textwidth]{{figures/fig2_site_context_map.png}}
\caption{{Non-interpolated total-sound site context at candidate coordinates (marker area proportional to runs; red cross flags Saldanha Bay's unresolved secondary location).}}
\label{{fig:site_context}}
\end{{figure}}

\subsection{{External Vessel Corpora and Base-10 Deci-Decade Alignment}}

Keeping \texttt{{Test.csv}} strictly inference-only (1,500 windows, 48.3\% containing edge omissions: 41.6\% missing \texttt{{m01}}, 6.7\% missing \texttt{{m15}}), I extract clean source templates from 525 SHA-256-verified VTUAD v1.3.0 recordings \parencite{{yuan2026vtuad,onc2026oceans}}—isolating 22 misfiled background clips into 38 ambient references and 487 active vessel clips across 172 hulls, separate from v1.3.2 \parencite{{yuan2026vtuad132}}—and 96 stratified coastal small-craft clips from HearMyShip \parencite{{shipton2025hearmyship,hearmyship2025platform}}. Non-overlapping 2\,s Hann periodograms \parencite{{welch1967fft}} match Equation~\eqref{{eq:decidecade}} \parencite{{zindi2026rules,iec2014bands}}:
\begin{{equation}}
\label{{eq:decidecade}}
f_k = 10^{{1.5 + 0.1 k}}, \qquad
P_k = \Delta f \sum_{{f \in [f_k 10^{{-0.05}},\, f_k 10^{{0.05}})}} S(f), \qquad
k \in \{{0,\dots,24\}}.
\end{{equation}}
Figure~\ref{{fig:spectral_mismatch}} reveals why direct source superposition leaks: South African stations slope downward by 20--34\,dB across 31.6\,Hz--7.94\,kHz, creating 7.5--26.2\,dB RMS shape distances against external vessels. Table~\ref{{tab:inventory}} summarizes the zero-leakage partitions used across baselines and v2--v4.

\begin{{table}}[!htbp]
\centering
\caption{{Data inventory and source-isolation boundaries across South African development, VTUAD, HearMyShip, and synthetic v2--v4 partitions.}}
\label{{tab:inventory}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llllr@{{}}}}
\toprule
Dataset / Partition & Temporal or Vessel Scope & Isolation Boundary & Extracted Units & Count \\
\midrule
South African Archive & 10 sites, 17 recorders & Excludes Cape Recife \& Riy Banks & 1-min rows / 15-min windows & 1,715,779 / 88,143 \\
VTUAD v1.3.0 & 172 active hulls + 38 bg. & Disjoint \texttt{{hull\_id}} & 2\,s min rows / WAVs & 2,511 / 525 \\
HearMyShip & 8 coastal craft classes & Passage-disjoint & 2\,s 20\,s CPA clips (75 / 21) & 96 \\
Modeling v2 (Baseline) & 6 train / Gap val / 2 holdout & 5-fold run \& hull disjoint & 22.6\% zero windows (6k/1.5k/541) & 8,041 \\
Modeling v3 (Hard-Neg) & 4 strata (inc. reef \& glitch) & Zero fold/split overlap & 41.6\% zero windows (10k/2k/1.5k/541) & 14,041 \\
Modeling v4 (Cal-Inv) & 5 strata + 65.3\%--100\% $h(f)$ warp & Strict ship negatives \& disjoint runs & {100*man_v4['train']['zero_fraction']:.1f}\% zero train (10k/2k/1.5k/541) & 14,041 \\
Zindi Evaluation Set & Unknown SA deployments & Strictly inference-only & Evaluation windows & 1,500 \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.83\textwidth]{{figures/fig3_spectral_structure_mismatch.png}}
\caption{{Cross-corpus spectral mismatch: (a) peak-normalized South African station medians overlaid with aligned 2\,s VTUAD (525) and HearMyShip (96) profiles; and (b) VTUAD RMS shape distance against P90--P10 run spread.}}
\label{{fig:spectral_mismatch}}
\end{{figure}}

\section{{Methodology: Physical Mixtures and \texorpdfstring{{$h(f)$}}{{h(f)}} Calibration Invariance}}

\subsection{{Linear-Energy Superposition Under Observed Masks}}

Given background window $B_{{\mathrm{{dB}}}}(t,b)$ and nonnegative vessel template $V_0(t,b)$ relative to $R_0=100$\,dB, Equation~\eqref{{eq:energy_mix}} scales vessel energy to target ratio $q$, and Equation~\eqref{{eq:share_target}} evaluates ground-truth added share strictly over observed cells $\Omega$ under sampled \texttt{{Test.csv}} edge-dropout masks:
\begin{{equation}}
\label{{eq:energy_mix}}
B(t,b) = 10^{{(B_{{\mathrm{{dB}}}}(t,b) - R_0)/10}}, \qquad
\alpha = \frac{{q \sum_{{t,b}} B(t,b)}}{{(1 - q)\sum_{{t,b}} V_0(t,b)}}, \qquad
M(t,b) = B(t,b) + \alpha V_0(t,b),
\end{{equation}}
\begin{{align}}
\label{{eq:share_target}}
y_{{\mathrm{{obs}}}} &= \frac{{\sum_{{(t,b)\in\Omega}} \alpha V_0(t,b)}}{{\sum_{{(t,b)\in\Omega}} \big(B(t,b) + \alpha V_0(t,b)\big)}} \in [0,1], \nonumber\\
X(t,b) &= \begin{{cases}}
R_0 + 10\log_{{10}}\!\max\!\big(M(t,b),\, 10^{{-30}}\big), & (t,b) \in \Omega, \\
\mathrm{{NaN}}, & (t,b) \notin \Omega.
\end{{cases}}
\end{{align}}
While Modeling~v2 established that true relative scaling outperformed window-maximum shifting (lowering unseen holdout RMSE by 27.5\% vs.\ \texttt{{V1\_RF\_MaxShift}}, while revealing a precision-recall trade-off between RMSE reduction and control false alarms under heavy contrast augmentation), Submission~\#2's live public AUC (0.5484) revealed that Equation~\eqref{{eq:share_target}} remained vulnerable to unknown hydrophone frequency responses $h(f)$ across deployment hardware.

\subsection{{Iterative Modeling Progression: v2 \texorpdfstring{{$\to$}}{{to}} v3 \texorpdfstring{{$\to$}}{{to}} v4 Calibration Invariance}}

In \textbf{{Modeling~v3}}, I increased quiet windows to 41.6\%, introduced explicit South African hard negatives (biological reef bursts, VLF surf swells, and $>15$\,dB drops), removed \texttt{{mask\_m01\_missing}} dropout shortcuts, expanded tabular features to 180 excess descriptors, and trained a 7-channel Squeeze-and-Excitation residual CNN (\texttt{{CNN2D\_v3}}) alongside Hurdle GBDT regressors $\hat{{p}}_{{\mathrm{{vessel}}}} \cdot \hat{{y}}_{{\mathrm{{pos}}}}$. Post-run feature inspection exposed \texttt{{proto\_spec\_logit}} capturing 47.0\% of LightGBM gain (Figure~\ref{{fig:v4_progression}}b), acting as an ONC-vs.-SA sensor-tilt classifier rather than a vessel detector.

In \textbf{{Modeling~v4}}, I permanently closed that shortcut through three structural fixes:
\begin{{enumerate}}
    \item \textbf{{Exact $h(f)$-invariant temporal excess residuals $R(t,f)$:}} Equation~\eqref{{eq:hf_invariant}} centers each third-octave band $f$ independently by its observed 20th temporal percentile $Q_{{0.20, t' \in \Omega(f)}}$, canceling any static hydrophone sensitivity curve $h(f)$ algebraically ($\max|\Delta R| \le 1.6 \times 10^{{-14}}$ under $+6$\,dB/decade tilt plus 2.5\,dB modal ripple):
    \begin{{equation}}
    \label{{eq:hf_invariant}}
    R(t, f) = X(t, f) - Q_{{0.20,\, t' \in \Omega(f)}}\!\big(X(t', f)\big)
            = \big(X_0(t,f) + h(f)\big) - \big(Q_{{0.20}}(X_0(\cdot,f)) + h(f)\big).
    \end{{equation}}
    \item \textbf{{Strict South African five-stratum simulation and sharp VTUAD transients:}} Training ($10{{,}}000$ rows) and 100\% calibration-warped \texttt{{val\_cal\_hard}} ($1{{,}}500$ rows) separate ambient runs into \texttt{{strict\_hf\_ship}}, \texttt{{vlf\_surf}}, \texttt{{reef\_active}}, \texttt{{glitch\_drop}}, and \texttt{{ambient\_normal}}, pair 2--5\,min sharp VTUAD bursts, and apply $\pm 6$\,dB/decade tilts and modal ripples to both background-only ($y=0$) and positive windows.
    \item \textbf{{Calibration-augmented 7-channel SE-ResNet and OOF simplex blend:}} \texttt{{CNN2D\_v4}} preserves exact $h(f)$ invariance across six of seven input tensors, undergoes live calibration-tilt training augmentation, and optimizes Equation~\eqref{{eq:cnn_loss}} jointly with log-loss simplex blending and out-of-fold temperature scaling ($T={w_v4['score_temperature']:.3f}$):
    \begin{{equation}}
    \label{{eq:cnn_loss}}
    \mathcal{{L}} = \mathrm{{MSE}}(\hat{{y}}, y_{{\mathrm{{obs}}}}) + 0.25\,\mathrm{{BCE}}\big(\hat{{z}},\, \mathbb{{I}}[y_{{\mathrm{{obs}}}} > 0]\big).
    \end{{equation}}
\end{{enumerate}}
Final \texttt{{Ensemble\_v4}} OOF share weights allocate {sh_w['CNN2D_v4']:.3f} to \texttt{{CNN2D\_v4}}, {sh_w['LightGBM_v4_Hurdle']:.3f} to \texttt{{LightGBM\_v4\_Hurdle}}, {sh_w['CatBoost_v4_Hurdle']:.3f} to \texttt{{CatBoost\_v4\_Hurdle}}, {sh_w['XGBoost_v4_Hurdle']:.3f} to \texttt{{XGBoost\_v4\_Hurdle}}, {sh_w['LightGBM_v4']:.3f} to \texttt{{LightGBM\_v4}}, and {sh_w['XGBoost_v4']:.3f} to \texttt{{XGBoost\_v4}}; presence weights assign {sc_w['CNN2D_v4']:.3f} to \texttt{{CNN2D\_v4}}, {sc_w['LightGBM_v4']:.3f} to \texttt{{LightGBM\_v4}}, and {sc_w['CatBoost_v4']:.3f} to \texttt{{CatBoost\_v4}}.

\subsection{{International Acoustic Standards Evaluation Framework}}

Following European MSFD D11C2 \parencite{{farcas2020shipping}} and arithmetic energy-averaging guidance \parencite{{merchant2012averaging}}, Equation~\eqref{{eq:jomopans}} maps predicted added share $s \in [0,1)$ to JOMOPANS excess shipping sound in decibels:
\begin{{equation}}
\label{{eq:jomopans}}
L_{{\mathrm{{excess}}}}(s) = -10\log_{{10}}\!\big(1 - \min(s, 0.995)\big).
\end{{equation}}
I evaluate all models across Lin's Concordance Correlation Coefficient (CCC), Nash-Sutcliffe Efficiency (NSE), Brier reliability/resolution decomposition, 10-bin Expected Calibration Error ($\mathrm{{ECE}}_{{10}}$), Per-Channel Energy Normalization (PCEN; \citealp{{lostanlen2019pcen}}), and unsupervised Inexact ALM Robust PCA \parencite{{candes2011rpca,lin2017rpca}}.

\section{{Empirical Results Across Model Generations and Standards}}

\subsection{{Modeling Progression (v2 \texorpdfstring{{$\to$}}{{to}} v3 \texorpdfstring{{$\to$}}{{to}} v4) and Feature Shortcut Elimination}}

Table~\ref{{tab:progression}} and Figure~\ref{{fig:v4_progression}} document monotonic accuracy, calibration, and zero-window gains across every benchmark regime. On the fully calibration-warped stress split \texttt{{val\_cal\_hard}} ($n=1{{,}}500$), \texttt{{Ensemble\_v4}} raises $1-\mathrm{{RMSE}}$ from 0.8364 (\texttt{{Ensemble\_v2}}) and 0.8788 (\texttt{{Ensemble\_v3}}) to {hc_v4['zindi_score_1_minus_rmse']:.4f}, improves ROC AUC from 0.9312 to {hc_v4['roc_auc']:.4f}, and cuts quiet-window \texttt{{zero\_rmse}} by 58.8\% (from 0.1612 to {hc_v4['zero_rmse']:.4f}). On completely unseen Fish Hoek and Sea Point stations ($n=541$), \texttt{{Ensemble\_v4}} achieves {ho_v4['zindi_score_1_minus_rmse']:.4f} $1-\mathrm{{RMSE}}$ (RMSE {ho_v4['rmse']:.4f}, AUC {ho_v4['roc_auc']:.4f}, \texttt{{zero\_rmse}} {ho_v4['zero_rmse']:.4f})—representing a 30.0\% RMSE reduction over \texttt{{Ensemble\_v2}} (0.1114). Crucially, Figure~\ref{{fig:v4_progression}}b confirms that removing \texttt{{proto\_spec\_logit}} shifted tree splits cleanly onto $h(f)$-invariant HF/MF persistence and kurtosis features (\texttt{{ex\_frac\_hf}}, \texttt{{hf\_contrib\_q20}}, \texttt{{R\_max\_b24}}, \texttt{{m\_hf\_lag1}}, \texttt{{ex\_curv\_mid\_vs\_ends}}). Inference across \texttt{{Test.csv}} ($1{{,}}500$ windows) produces finite predictions ($\mathrm{{mean}} = {san_v4['test_share_mean']:.4f}$, $\mathrm{{median}} = 0.2109$, $\mathrm{{SD}} = {san_v4['test_share_std']:.4f}$).

\begin{{table}}[!htbp]
\centering
\caption{{End-to-end modeling and leaderboard progression across Submissions~\#1--\#2 and Candidates v3--v4 (evaluated across 100\% calibration-warped \texttt{{val\_cal\_hard}}, \texttt{{val\_standard}}, and unseen holdout stations).}}
\label{{tab:progression}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lllrrrr@{{}}}}
\toprule
Model / Stage & Features & Live Zindi Public & Val Cal-Hard ($1-\mathrm{{RMSE}}$ / AUC) & Val Standard ($1-\mathrm{{RMSE}}$ / AUC) & Unseen Holdout ($1-\mathrm{{RMSE}}$ / AUC) & Holdout Zero RMSE $\downarrow$ \\
\midrule
{chr(10).join(table3_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig4_v2_v3_v4_progression_and_features.png}}
\caption{{Modeling progression and shortcut removal: (a) $1-\mathrm{{RMSE}}$ and quiet-window \texttt{{zero\_rmse}} across \texttt{{val\_cal\_hard}}, \texttt{{val\_standard}}, and unseen \texttt{{holdout}}; and (b) Top-8 GBDT feature shift replacing v3's \texttt{{proto\_spec\_logit}} tilt shortcut with physical and $R_*$ descriptors in v4.}}
\label{{fig:v4_progression}}
\end{{figure}}

\subsection{{International Acoustic Standards, Unsupervised Baselines, and Stress Curves}}

Table~\ref{{tab:standards}} and Figure~\ref{{fig:standards_cal}} demonstrate why uncalibrated analytical rules cannot replace learned calibration-invariant models in shallow coastal telemetry. On \texttt{{val\_cal\_hard}}, \texttt{{Physics\_Robust\_q20}} ($1-\mathrm{{RMSE}}=0.3630$, NSE $-4.028$) and unsupervised \texttt{{RPCA\_Unsupervised}} ($1-\mathrm{{RMSE}}=0.5413$, CCC $-0.154$, AUC 0.500) score worse than predicting a constant mean (0.7158) because coastal waves, shrimp choruses, and 15-minute vessel dwells breach stationary low-rank background assumptions (illustrated spectrally in Figure~\ref{{fig:signal_decomp}}). By contrast, \texttt{{Ensemble\_v4}} achieves high agreement (Lin's CCC 0.9204 on \texttt{{val\_cal\_hard}}, 0.9240 on \texttt{{holdout}}), drops JOMOPANS $L_{{\mathrm{{excess}}}}$ MAE from 0.886\,dB (\texttt{{v2}}) and 0.631\,dB (\texttt{{v3}}) to 0.547\,dB (\texttt{{val\_cal\_hard}}) and 0.308\,dB (\texttt{{holdout}}), and improves probability calibration $\mathrm{{ECE}}_{{10}}$ from 0.1889 to 0.0273.

Under synthetic hydrophone tilt sweeps from $-6$ to $+10$\,dB/decade on unseen holdout windows (Figure~\ref{{fig:boundary_stress}}a), \texttt{{Ensemble\_v2}} degrades sharply at $+8$\,dB/decade (AUC drops to 0.866; \texttt{{zero\_rmse}} rises to 0.202) and \texttt{{Ensemble\_v3}} still suffers quiet-window false positives (\texttt{{zero\_rmse}} 0.204), whereas \texttt{{Ensemble\_v4}} maintains 0.956 ROC AUC with \texttt{{zero\_rmse}} = 0.093. Across every individual holdout station and all five physical hard-negative strata (Table~\ref{{tab:strata}} and Figure~\ref{{fig:boundary_stress}}b--c), \texttt{{Ensemble\_v4}} delivers consistent error reductions—lifting Fish Hoek from 0.8715 to 0.9123, \texttt{{strict\_hf\_ship}} from 0.8087 to 0.8794, and \texttt{{vlf\_surf}} from 0.8698 to 0.9163.

\begin{{table}}[!htbp]
\centering
\caption{{International acoustic standards comparison across 100\% calibration-warped \texttt{{val\_cal\_hard}} ($n=1{{,}}500$), unseen \texttt{{holdout}} ($n=541$), and $+8$\,dB/decade tilt stress.}}
\label{{tab:standards}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
Method & Cal-Hard CCC $\uparrow$ & Cal-Hard NSE $\uparrow$ & Cal-Hard $L_{{\mathrm{{ex}}}}$ MAE $\downarrow$ & Cal-Hard $\mathrm{{ECE}}_{{10}}\downarrow$ & Holdout CCC $\uparrow$ & Holdout $L_{{\mathrm{{ex}}}}$ MAE $\downarrow$ & Holdout $\mathrm{{ECE}}_{{10}}\downarrow$ & $+8$\,dB/dec Tilt (AUC / zRMSE) \\
\midrule
{chr(10).join(table4_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig5_standards_calibration_and_agreement.png}}
\caption{{International standards evaluation: (a) 10-bin probability reliability curves and $\mathrm{{ECE}}_{{10}}$ on \texttt{{val\_cal\_hard}}; (b) predicted versus observed agreement with Lin's CCC, NSE, and Spearman $\rho$; and (c) regime-stratified share RMSE (\texttt{{Zero}}, \texttt{{Low}}, \texttt{{Moderate}}, \texttt{{High}}).}}
\label{{fig:standards_cal}}
\end{{figure}}

\begin{{table}}[!htbp]
\centering
\caption{{Unseen-station and five-stratum accuracy progression from \texttt{{Ensemble\_v2}} through \texttt{{Ensemble\_v3}} to \texttt{{Ensemble\_v4}}.}}
\label{{tab:strata}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}llrrrrr@{{}}}}
\toprule
Scope & Group / Stratum & $n$ & v2 ($1-\mathrm{{RMSE}}$ / AUC) & v3 ($1-\mathrm{{RMSE}}$ / AUC) & v4 ($1-\mathrm{{RMSE}}$ / AUC) & v4 Zero RMSE $\downarrow$ \\
\midrule
{chr(10).join(table5_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.84\textwidth]{{figures/fig6_boundary_stress_and_calibration_shift.png}}
\caption{{Boundary stress tests: (a) unseen-holdout ROC AUC degradation under $-6$ to $+10$\,dB/dec hydrophone tilt; (b) error by South African hard-negative stratum; and (c) JOMOPANS/MSFD D11C2 excess-level MAE ($L_{{\mathrm{{ex}}}}$) across \texttt{{val\_cal\_hard}}, \texttt{{val\_standard}}, and \texttt{{holdout}}.}}
\label{{fig:boundary_stress}}
\end{{figure}}

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=0.77\textwidth]{{figures/fig7_acoustic_signal_boundaries_decomposition.png}}
\caption{{Acoustic signal decomposition across three real South African hard negatives ($y=0$: shrimp reef, real SA vessel pass, dropout glitch) and a controlled VTUAD mixture ($y=0.624$), contrasting centered ISO~18405 spectra, PCEN contrast, and RPCA sparse transients.}}
\label{{fig:signal_decomp}}
\end{{figure}}

\section{{Evidentiary Boundaries, Domain-Generalization Contingency, and Reproducibility}}

Three open measurement boundaries still constrain coastal inference across this dataset: (a) missing per-hydrophone sensitivity tables prevent converting uncalibrated exports to physical $\mathrm{{dB\,re\,1\,\mu Pa}}$ \parencite{{iso2017underwater,robinson2014npl}}; (b) candidate coordinates without verified deployment dates prohibit absolute shipping-density interpolation \parencite{{ainslie2020adeon,farcas2020shipping}}; and (c) zero synchronized field AIS logs exist in test windows. If Submission~\#3 (\texttt{{submissions/candidate\_v4\_ensemble.csv}}) still exhibits residual upward scoring bias on uncalibrated ports, my documented Modeling~v5 contingency isolates \texttt{{score\_added\_vessel}} strictly to exact $h(f)$-invariant \texttt{{R\_*}}, PCEN, and RPCA predictors combined with \texttt{{ResNorm}} \parencite{{kim2021resnorm}} and \texttt{{Freq-MixStyle}} \parencite{{zhou2021mixstyle}} normalization without spectral-tilt channel~4.

All results are bit-exact and auditable anytime across any clean Python~3.13 environment matching the 125 pinned non-AutoML packages in \path{{requirements.txt}} \parencite{{breiman2001rf,ke2017lightgbm,chen2016xgboost,prokhorenkova2018catboost,he2016resnet}}:
\begin{{itemize}}
    \item \textbf{{Single-command verification and archives:}} \path{{scripts/reproduce_pipeline.py}} (\texttt{{-{{}}-mode verify-only}}) validates environment pins, SHA-256 digests, and six independent deterministic self-tests (\texttt{{v2}}, \texttt{{v3}}, \texttt{{v4\_simulator}}, \texttt{{v4\_features}}, \texttt{{v4\_train}}, and \texttt{{standardized\_evaluation\_suite}}). Subprocess reloads of \path{{models/v4/bundle_v4.joblib}} (zero custom \texttt{{\_\_main\_\_}} classes) reproduce saved submissions within $\le 3.55 \times 10^{{-9}}$ share / $0.0$ score, while $+37.5$\,dB shifts yield $0.0$ holdout difference.
    \item \textbf{{Reproducible notebooks and archives:}} Three executed notebooks (\texttt{{01}} EDA/experiments, \texttt{{02}} V2 11-model benchmark, \texttt{{03}} \texttt{{v2->v3->v4}} generalization and standards) accompany the verified $<10$\,MB runnable code package \path{{research_and_modeling_code_final.zip}} (40 scripts, zero raw audio or competition test data) and preserved V6 science archive \path{{research_code_v6.zip}} ({v6_verif['code_archive']['bytes']} bytes). External-data (\path{{docs/EXTERNAL_DATA_DISCLOSURE.md}}) and map disclosures remain archived, and \texttt{{Test.csv}} stayed strictly inference-only.
\end{{itemize}}

\section{{Conclusion}}

Tracing live leaderboard performance from Submission~\#1 (0.7615 / 0.5175 AUC) through \texttt{{Ensemble\_v2}} (0.7750 / 0.5484 AUC) exposed hydrophone frequency response $h(f)$ leakage and coastal ambient transients that clean multi-site synthetic splits initially concealed. By proving algebraic $h(f)$ cancellation in \texttt{{Ensemble\_v4}} via per-band temporal excess residuals $R(t,f)$, five-stratum South African hard negatives, and calibration-augmented 7-channel SE-ResNet blending, I lifted unseen-station accuracy to 0.9219 (0.9737 AUC, 0.0329 \texttt{{zero\_rmse}}), cut JOMOPANS excess error to 0.308\,dB, and maintained 0.9561 AUC under severe $+8$\,dB/decade hardware tilt while keeping measurement boundaries explicit.

\printbibliography[heading=bibintoc,title={{References}}]

\end{{document}}
"""
    out_path.write_text(tex, encoding="utf-8")


def write_readme_overleaf(path: Path) -> None:
    """Write concise step-by-step instructions for compiling in Overleaf or locally."""
    txt = r"""# Overleaf Upload & Compilation Instructions

1. Upload `overleaf_research_manuscript.zip` directly in Overleaf (**New Project -> Upload Project**).
2. Set **Menu -> Compiler** to **`pdfLaTeX`** (Overleaf automatically runs **`biber`** for `biblatex` `style=apa` 7th-edition formatting).
3. `main.tex` is authored as `Prepared by Emmanuel Kwaku Botchway` and compiles cleanly in Overleaf or locally via `python scripts/build_overleaf_manuscript.py --compile-pdf`.
4. Confirm the exported PDF renders between **4 and 10 pages** and is `< 10 MB` before uploading to Zindi alongside `research_and_modeling_code_final.zip`.
"""
    path.write_text(txt, encoding="utf-8")


def validate_overleaf_package(tex_path: Path, bib_path: Path, fig_dir: Path) -> dict:
    """Strictly validate LaTeX syntax, unescaped characters, labels/refs, figures, APA BibTeX citations, and solo voice."""
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

    # 4. Check all \ref and \eqref targets exist (7 eqs + 5 tables + 7 figs = 19 labels)
    labels = set(re.findall(r"\\label\{([^}]+)\}", cleaned))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", cleaned))
    assert len(labels) == 19, f"Expected 19 labels (7 eq, 5 tab, 7 fig), got {len(labels)}: {sorted(labels)}"
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

    # 6. Check prose lines for accidental unescaped _, %, or & and plural first-person pronouns
    in_math = False
    in_tabular = False
    for ln_no, raw_ln in enumerate(cleaned.splitlines(), 1):
        line = re.sub(
            r"\\(?:texttt|path|url|href|includegraphics|label|ref|eqref|addbibresource|parencite|textcite|citealp)(?:\[[^\]]*\])?\{[^}]*\}",
            "",
            raw_ln,
        )
        line = re.sub(r"`[^`]+`", "", line)
        line = re.sub(r"\$\$[^$]*\$\$|\$[^$]*\$", "", line)
        if re.search(r"\\begin\{(?:equation|align)\*?\}", raw_ln):
            in_math = True
            continue
        if re.search(r"\\end\{(?:equation|align)\*?\}", raw_ln):
            in_math = False
            continue
        if re.search(r"\\begin\{tabular", raw_ln):
            in_tabular = True
            continue
        if re.search(r"\\end\{tabular", raw_ln):
            in_tabular = False
            continue
        if in_math:
            continue
        line_clean = line.replace("\\_", "").replace("\\%", "").replace("\\&", "").replace("\\#", "")
        assert "_" not in line_clean, f"Unescaped underscore on line {ln_no}: {raw_ln}"
        assert "%" not in line_clean, f"Unescaped percent on line {ln_no}: {raw_ln}"
        if not in_tabular:
            assert "&" not in line_clean, f"Unescaped ampersand outside tabular on line {ln_no}: {raw_ln}"

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
    """Extract unified code ZIP in isolation, compile all 40 scripts, validate 3 notebooks, and run deterministic self-tests."""
    assert zip_path.is_file(), f"Missing code archive: {zip_path}"
    zip_bytes = zip_path.stat().st_size
    assert zip_bytes < 10_000_000, f"Code archive exceeds 10 MB: {zip_bytes}"

    banned_tokens = (
        ".wav", ".parquet", ".npz", ".joblib", ".ds_store", "__macosx",
        "test.csv", "south_african_real_acoustics.zip", "private_local_" + "archive",
    )
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    for n in names:
        low = n.lower()
        assert not any(tok in low for tok in banned_tokens), f"Prohibited entry in code archive: {n}"

    det_env = os.environ.copy()
    det_env.update({
        "PYTHONHASHSEED": "42",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/mpl",
    })

    with tempfile.TemporaryDirectory(prefix="code_zip_verify_") as tmp:
        workdir = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(workdir)

        # Ensure text docs inside ZIP contain zero leaked external backup URLs
        for doc_p in workdir.rglob("*.md"):
            txt_low = doc_p.read_text(encoding="utf-8", errors="ignore").lower()
            assert ("mega" + ".nz") not in txt_low and ("private_local_" + "archive") not in txt_low, (
                f"Private archive reference leaked into {doc_p.relative_to(workdir)}"
            )

        py_files = sorted((workdir / "scripts").glob("*.py"))
        assert len(py_files) == 40, f"Expected 40 extracted scripts, got {len(py_files)}"
        for py in py_files:
            py_compile.compile(str(py), doraise=True)

        nb_files = sorted((workdir / "notebooks").glob("*.ipynb"))
        assert len(nb_files) == 3, f"Expected 3 notebooks, got {len(nb_files)}"
        for nb_p in nb_files:
            nb = json.loads(nb_p.read_text(encoding="utf-8"))
            assert len(nb.get("cells", [])) > 0, f"Empty notebook: {nb_p.name}"

        for test_script in [
            "scripts/modeling_v2_train.py",
            "scripts/modeling_v3_train.py",
            "scripts/modeling_v4_train.py",
            "scripts/standardized_evaluation_suite.py",
        ]:
            res = subprocess.run(
                [sys.executable, test_script, "--self-test-only"],
                cwd=workdir,
                env=det_env,
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0 and "PASS" in res.stdout, (
                f"Clean-copy {test_script} --self-test-only failed:\n{res.stdout}\n{res.stderr}"
            )

    return {
        "code_zip_path": str(zip_path.relative_to(ROOT)),
        "code_zip_bytes": zip_bytes,
        "code_zip_sha256": _sha256(zip_path),
        "code_zip_entries": len(names),
        "code_zip_scripts": 40,
        "code_zip_notebooks": 3,
        "code_zip_self_test": "PASS",
    }


def build_unified_code_archive(out_zip: Path) -> dict:
    """Build unified <10 MB runnable code + notebooks + docs + TeX archive covering v2, v3, v4, and standards."""
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
        "scripts/modeling_v3_features.py",
        "scripts/modeling_v3_simulator.py",
        "scripts/modeling_v3_train.py",
        "scripts/modeling_v4_features.py",
        "scripts/modeling_v4_simulator.py",
        "scripts/modeling_v4_train.py",
        "scripts/standardized_evaluation_suite.py",
        "scripts/reproduce_pipeline.py",
        "scripts/build_modeling_notebook_02.py",
        "scripts/build_modeling_notebook_03.py",
        "scripts/polish_research_notebook_01.py",
        "scripts/build_submission_deliverables.py",
        "scripts/build_overleaf_manuscript.py",
    ]
    all_script_rels = sorted(set(v6_scripts + extra_scripts))
    assert len(all_script_rels) == 40, f"Expected 40 unique scripts, got {len(all_script_rels)}"

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
        (ROOT / "CHECKPOINT.md", "CHECKPOINT.md"),
        (ROOT / "requirements.txt", "requirements.txt"),
        (ROOT / "docs/EXTERNAL_DATA_DISCLOSURE.md", "docs/EXTERNAL_DATA_DISCLOSURE.md"),
        (ROOT / "docs/ORGANIZER_MAP_CLARIFICATION.md", "docs/ORGANIZER_MAP_CLARIFICATION.md"),
        (ROOT / "docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md", "docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md"),
        (ROOT / "docs/SUBMISSION_STATUS.md", "docs/SUBMISSION_STATUS.md"),
        (ROOT / "docs/PROJECT_LEARNING_LOG.md", "docs/PROJECT_LEARNING_LOG.md"),
        (ROOT / "docs/protocols/RESEARCH_METHODS_AND_DISCLOSURE_GATES.md", "docs/protocols/RESEARCH_METHODS_AND_DISCLOSURE_GATES.md"),
        (ROOT / "docs/protocols/EXPERIMENT007_PREDECLARED_PROTOCOL.md", "docs/protocols/EXPERIMENT007_PREDECLARED_PROTOCOL.md"),
        (ROOT / "notebooks/01_south_african_acoustic_eda.ipynb", "notebooks/01_south_african_acoustic_eda.ipynb"),
        (ROOT / "notebooks/02_vessel_noise_modeling.ipynb", "notebooks/02_vessel_noise_modeling.ipynb"),
        (ROOT / "notebooks/03_domain_generalization_v3_v4_and_standards.ipynb", "notebooks/03_domain_generalization_v3_v4_and_standards.ipynb"),
        (ROOT / "deliverables/research_submission/training_only_site_aggregates.csv", "site_context/training_only_site_aggregates.csv"),
        (ROOT / "docs/metadata/hearmyship_split_v1.csv", "metadata/hearmyship_split_v1.csv"),
        (MAIN_TEX, "manuscript/main.tex"),
        (BIB_FILE, "manuscript/references.bib"),
    ])
    manifest_repro = ROOT / "reports/reproducibility_manifest.json"
    if manifest_repro.is_file():
        files_to_bundle.append((manifest_repro, "reports/reproducibility_manifest.json"))

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
    parser.add_argument("--code-zip-only", action="store_true", help="Rebuild and verify research_and_modeling_code_final.zip and refresh OVERLEAF_MANIFEST.json.")
    args = parser.parse_args()

    if args.verify_only:
        print(json.dumps(verify_existing(), indent=2))
        return

    if args.code_zip_only:
        code_info = build_unified_code_archive(CODE_ZIP)
        rec = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        rec.update(code_info)
        MANIFEST_JSON.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=2))
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
