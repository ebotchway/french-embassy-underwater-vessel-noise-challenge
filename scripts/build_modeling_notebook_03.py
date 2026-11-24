#!/usr/bin/env python3
"""Build notebooks/03_domain_generalization_v3_v4_and_standards.ipynb documenting Modeling v3, v4, and international acoustic standards."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks/03_domain_generalization_v3_v4_and_standards.ipynb"


def build_notebook() -> nbf.NotebookNode:
    """Construct Notebook 03 in solo first-person voice with deterministic CPU verification."""
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))

    def code(src: str) -> None:
        cells.append(nbf.v4.new_code_cell(src.strip() + "\n"))

    md(
        """
# 03 · Domain Generalization (`v2 -> v3 -> v4`), Calibration Invariance & Acoustic Standards

In this notebook, I document and verify my full domain-generalization progression from the live Zindi leaderboard results (`Submission #1` and `Submission #2 / v2`) through **Modeling v3** (`180` features, hard-negative strata, 7-channel SE-ResNet CNN), **Modeling v4** (`205` features, `71` exact hydrophone calibration $h(f)$-invariant `R_*` features, strict SA ship hard negatives), and my **International Acoustic Standards Benchmark** (`ISO 18405:2017`, `IEC 61260-1`, `NPL GPG 133`, `ADEON`, `JOMOPANS`, `PCEN`, and `RPCA`). Every metric, live invariance assertion, and bundle reload check below executes deterministically without ever fitting on `Test.csv`.
"""
    )

    code(
        """
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

# Import torch before GBDT C extensions to prevent macOS OpenMP runtime clashes.
import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path.cwd().resolve()
if not (ROOT / "scripts").is_dir():
    ROOT = ROOT.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from extract_external_decidecade import EXACT_CENTRES
from modeling_v4_features import build_cnn_tensors_v4, extract_features_v4
from modeling_v4_train import predict_from_bundle_v4, seed_everything
from standardized_evaluation_suite import compute_pcen_map, eval_rpca_unsupervised

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
seed_everything(42, torch_threads=1)
print(f"Project root: {ROOT.name} | Deterministic CPU environment locked (seed=42)")
"""
    )

    md(
        """
## 1. Live Leaderboard Diagnosis (`Submission #1` & `Submission #2 / v2`)

Even though `Ensemble_v2` achieved strong synthetic generalization (`0.8747` known-station validation, `0.8833` unseen Fish Hoek + Sea Point holdout `1-RMSE`), my two live Zindi public leaderboard submissions revealed a critical real-world residual gap: `ROC AUC` improved only modestly (`0.517533 -> 0.548377777`) alongside a small public `1-RMSE` gain (`0.761536 -> 0.77501743`, Rank `508`). Auditing feature importances and `Test.csv` window structures pinpointed two mechanisms behind that gap:
1. **Unmodeled hydrophone frequency response $h(f)$:** Different recorders impose site-specific transfer curves (e.g., spectral tilt and modal ripple), letting global spectral-shape prototypes (`proto_spec_logit`) separate Ocean Networks Canada (VTUAD) ships from South African backgrounds by sensor signature rather than transient vessel physics.
2. **Sharp SA hard negatives & missingness shortcuts:** Short 2–5 minute VTUAD blocks, intense 63–125 Hz reef activity, 31.6–100 Hz VLF surf swells, and sharp `>15 dB` telemetry glitches triggered false alarms, while missing `m01`/`m15` boundary flags introduced spurious missingness associations.
"""
    )

    code(
        """
lb_history = pd.DataFrame([
    {
        "Stage": "Submission #1 (V1_RF_MaxShift)",
        "Feature Dim": 96,
        "Public 1-RMSE": 0.761536,
        "Public Share RMSE": 0.238464,
        "Public ROC AUC": 0.517533,
        "Unseen Holdout 1-RMSE": 0.8390,
        "Unseen Holdout AUC": 0.8980,
        "Status": "Scored (Submitted)",
    },
    {
        "Stage": "Submission #2 (Ensemble_v2)",
        "Feature Dim": 138,
        "Public 1-RMSE": 0.77501743,
        "Public Share RMSE": 0.224982569,
        "Public ROC AUC": 0.548377777,
        "Unseen Holdout 1-RMSE": 0.8886,
        "Unseen Holdout AUC": 0.9334,
        "Status": "Scored (Rank 508)",
    },
    {
        "Stage": "Candidate v3 (Hard-Negative + SE-ResNet)",
        "Feature Dim": 180,
        "Public 1-RMSE": None,
        "Public Share RMSE": None,
        "Public ROC AUC": None,
        "Unseen Holdout 1-RMSE": 0.8971,
        "Unseen Holdout AUC": 0.9686,
        "Status": "Diagnostic Intermediate (Superseded by v4)",
    },
    {
        "Stage": "Candidate v4 (h(f)-Invariant R(t,f))",
        "Feature Dim": 205,
        "Public 1-RMSE": None,
        "Public Share RMSE": None,
        "Public ROC AUC": None,
        "Unseen Holdout 1-RMSE": 0.9219,
        "Unseen Holdout AUC": 0.9737,
        "Status": "Verified & Ready for Submission #3",
    },
])
display(lb_history)
"""
    )

    md(
        """
## 2. Modeling v3: Hard-Negative Strata, Linear-Excess Features & 7-Channel SE-ResNet CNN

In **Modeling v3**, I expanded the simulator to `10,000` training / `2,000` `val_standard` / `1,500` `val_hard_event` / `541` unseen `holdout` windows (`0` cross-split or cross-fold `bg_run_id` and `hull_id` overlap), raised `zero_fraction` from `~22%` to `~42%`, introduced four stratified hard-negative groups (`ambient_normal`, `hard_event`, `reef_active`, `glitch_drop`), removed `mask_m01_missing` shortcut predictors, added `180` linear-excess features, and upgraded the neural branch to a 7-channel Squeeze-and-Excitation ResNet (`CNN2D_v3`) alongside Hurdle GBDT heads (`P(vessel) * E[share | positive]`). Auditing the resulting v3 feature rankings uncovered `proto_spec_logit` taking `47.0%` of LightGBM split gain (`25.8%` mean normalized importance), exposing residual sensitivity to fixed VTUAD vs. South African calibration tilt.
"""
    )

    code(
        """
man_v3 = json.loads((ROOT / "data/synthetic/modeling_v3/manifest.json").read_text())
cmp_v3 = pd.read_csv(ROOT / "reports/modeling_v3/v2_vs_v3_comparison.csv")
imp_v3 = pd.read_csv(ROOT / "reports/modeling_v3/feature_importance.csv")
w_v3 = json.loads((ROOT / "reports/modeling_v3/ensemble_weights.json").read_text())
san_v3 = json.loads((ROOT / "reports/modeling_v3/sanity_summary.json").read_text())

v3_splits = pd.DataFrame([
    {
        "Split": k,
        "Rows": v["rows"],
        "Zero Fraction": round(v["zero_fraction"], 4),
        "Mean Share": round(v["mean_share"], 4),
        "Mean Obs Min": round(v["mean_obs_minutes"], 2),
        "Strata": v["strata_counts"],
    }
    for k, v in man_v3.items()
])
display(v3_splits)

display(
    cmp_v3[cmp_v3["model"].isin(["Ensemble_v2", "CNN2D_v3", "Ensemble_v3", "LightGBM_v3_Hurdle"])]
    .sort_values(["split", "zindi_score_1_minus_rmse"], ascending=[True, False])
    .round(4)
)

fig, ax = plt.subplots(figsize=(9.5, 4.2))
top12_v3 = imp_v3.head(12).iloc[::-1]
cols_v3 = ["#D55E00" if f == "proto_spec_logit" else "#0072B2" for f in top12_v3["feature"]]
ax.barh(top12_v3["feature"], top12_v3["mean_norm_importance"], color=cols_v3)
ax.set_title("Modeling v3 Top-12 GBDT Features (Orange: Calibration-Tilt Shortcut Discovery)")
ax.set_xlabel("Mean Normalized Importance Across LightGBM, XGBoost & CatBoost")
plt.tight_layout()
plt.show()

print("Modeling v3 OOF Share Weights:", {k: round(v, 4) for k, v in w_v3["share_weights"].items() if v > 1e-4})
print("Modeling v3 Sanity:", {k: round(v, 4) if isinstance(v, float) else v for k, v in san_v3.items() if not isinstance(v, list)})
"""
    )

    md(
        r"""
## 3. Modeling v4: Exact $h(f)$ Calibration-Invariant `R(t,f)` & Strict SA Hard Negatives

To eliminate calibration-tilt leakage permanently, in **Modeling v4** I dropped `proto_spec_logit`, replaced `ex_tot_db_*` raw-spectrum features with `71` exact **frequency-response-invariant temporal excess residuals**:
$$R(t, f) = X(t, f) - Q_{0.20,\, t' \in \Omega(f)}\!\left(X(t', f)\right) = \left(X_0(t,f) + h(f)\right) - \left(Q_{0.20}(X_0(\cdot, f)) + h(f)\right)$$
so any time-invariant hydrophone sensitivity curve $h(f)$ cancels identically. I further split the synthetic benchmark into five strict physical strata (`strict_hf_ship`, `vlf_surf`, `reef_active`, `glitch_drop`, `ambient_normal`), simulated sharp 2–5 minute VTUAD blocks, and injected calibration warping (`±6 dB/dec` + cosine modal ripple + broadband jitter) into `65.3%` of training windows and `100%` of `val_cal_hard` (`n=1,500`). Below I prove exact $h(f)$ invariance live on unseen `holdout` windows before comparing `v2 -> v3 -> v4`.
"""
    )

    code(
        """
ho_v4 = np.load(ROOT / "data/synthetic/modeling_v4/holdout.npz")
X_sub = ho_v4["X"][:32].astype(np.float64)

# Live Invariance Proof: +37.5 dB global offset AND +6 dB/dec + 2.5 dB cosine ripple h(f)
norm_b = np.linspace(0.0, 1.0, 25)
h_curve = 6.0 * np.log10(EXACT_CENTRES / 500.0) + 2.5 * np.cos(2.0 * np.pi * norm_b + 0.7)

f_base = extract_features_v4(X_sub)
f_glob = extract_features_v4(X_sub + 37.5)
f_hf = extract_features_v4(X_sub + h_curve[None, None, :])
r_cols = [c for c in f_base.columns if c.startswith("R_")]

t_base = build_cnn_tensors_v4(X_sub)
t_glob = build_cnn_tensors_v4(X_sub + 37.5)
t_hf = build_cnn_tensors_v4(X_sub + h_curve[None, None, :])

inv_proof = pd.DataFrame([
    {"Transform": "+37.5 dB Global Offset", "Scope": f"All {f_base.shape[1]} Tabular Features", "Max |Diff|": float(np.max(np.abs(f_base.to_numpy() - f_glob.to_numpy())))},
    {"Transform": "+6 dB/dec + 2.5 dB Ripple h(f)", "Scope": f"71 Cal-Invariant R_* Features", "Max |Diff|": float(np.max(np.abs(f_base[r_cols].to_numpy() - f_hf[r_cols].to_numpy())))},
    {"Transform": "+37.5 dB Global Offset", "Scope": "7-Channel CNN Tensor", "Max |Diff|": float(np.max(np.abs(t_base - t_glob)))},
    {"Transform": "+6 dB/dec + 2.5 dB Ripple h(f)", "Scope": "CNN h(f)-Inv Channels [0,1,2,3,5,6]", "Max |Diff|": float(np.max(np.abs(t_base[:, [0,1,2,3,5,6]] - t_hf[:, [0,1,2,3,5,6]])))},
])
display(inv_proof)
assert inv_proof["Max |Diff|"].max() < 1e-5

man_v4 = json.loads((ROOT / "data/synthetic/modeling_v4/manifest.json").read_text())
cmp_v4 = pd.read_csv(ROOT / "reports/modeling_v4/v2_v3_v4_comparison.csv")
imp_v4 = pd.read_csv(ROOT / "reports/modeling_v4/feature_importance.csv")
test_audit_v4 = json.loads((ROOT / "reports/modeling_v4/test_prediction_audit.json").read_text())

focus_models = ["Ensemble_v2", "Ensemble_v3", "CNN2D_v4", "LightGBM_v4_Hurdle", "Ensemble_v4"]
display(
    cmp_v4[cmp_v4["model"].isin(focus_models)]
    [["split", "model", "n", "zindi_score_1_minus_rmse", "rmse", "zero_rmse", "roc_auc", "logloss", "brier"]]
    .sort_values(["split", "zindi_score_1_minus_rmse"], ascending=[True, False])
    .round(4)
)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
top12_v4 = imp_v4.head(12).iloc[::-1]
c_v4 = ["#009E73" if f.startswith("R_") else ("#CC79A7" if f.startswith(("m_", "ex_curv")) else "#0072B2") for f in top12_v4["feature"]]
axes[0].barh(top12_v4["feature"], top12_v4["mean_norm_importance"], color=c_v4)
axes[0].set_title("Modeling v4 Top-12 Features (Green: h(f)-Invariant R_*)")
axes[0].set_xlabel("Mean Normalized GBDT Importance")

tau_df = pd.DataFrame(test_audit_v4).set_index("subgroup")
sub_plot = tau_df.drop(index="ALL_1500")
x = np.arange(len(sub_plot))
w = 0.27
axes[1].bar(x - w, sub_plot["v2_share_mean"], width=w, color="#E69F00", label="Ensemble_v2")
axes[1].bar(x, sub_plot["v3_share_mean"], width=w, color="#0072B2", label="Ensemble_v3")
axes[1].bar(x + w, sub_plot["v4_share_mean"], width=w, color="#009E73", label="Ensemble_v4")
axes[1].set_xticks(x)
axes[1].set_xticklabels(sub_plot.index, rotation=20, ha="right", fontsize=8)
axes[1].set_ylabel("Mean Predicted Test Share")
axes[1].set_title("Inference-Only Test.csv Subgroup Behavior (Zero Fitting)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    )

    md(
        """
## 4. International Acoustic Standards & Boundary Stress Benchmark

Beyond Zindi's competition metrics, I evaluated `Constant_Mean`, `Physics_Robust_q20`, unsupervised `RPCA_Unsupervised` (Candès et al., 2011; Lin et al., 2017), `Ensemble_v2`, `Ensemble_v3`, and `Ensemble_v4` across **16+ international underwater acoustic metrics**: `ISO 18405:2017` terminology, `IEC 61260-1` base-10 deci-decades, `NPL GPG 133` & `ADEON` amplitude/calibration checks, `JOMOPANS` excess shipping sound $L_{\\text{excess}} = -10\\log_{10}(1 - s)$ MAE in dB, Lin's Concordance Correlation Coefficient (`CCC`), Nash-Sutcliffe Efficiency (`NSE`), regime-wise RMSE (`Zero`, `Low`, `Mod`, `High`), Brier reliability/resolution/uncertainty decomposition, Expected Calibration Error (`ECE_10`), and a synthetic `-6` to `+10 dB/dec` hydrophone calibration-tilt stress test on unseen `holdout`.
"""
    )

    code(
        """
# Live PCEN & RPCA shift-invariance verification
pcen_diff = float(np.nanmax(np.abs(compute_pcen_map(X_sub[0]) - compute_pcen_map(X_sub[0] + 37.5))))
s_rp1, c_rp1, _ = eval_rpca_unsupervised(X_sub[:8])
s_rp2, c_rp2, _ = eval_rpca_unsupervised(X_sub[:8] + 37.5)
rpca_diff = max(float(np.max(np.abs(s_rp1 - s_rp2))), float(np.max(np.abs(c_rp1 - c_rp2))))
print(f"Live Standards Preprocessor Check -> PCEN +37.5 dB max diff: {pcen_diff:.2e} | RPCA max diff: {rpca_diff:.2e}")

std_full = pd.read_csv(ROOT / "reports/standards_benchmark/standardized_metrics_full.csv")
tilt_df = pd.read_csv(ROOT / "reports/standards_benchmark/calibration_shift_curve.csv")
strata_df = pd.read_csv(ROOT / "reports/standards_benchmark/stratum_breakdown.csv")

cols_show = [
    "split", "model", "zindi_score_1_minus_rmse", "lin_ccc", "nse_r2",
    "rmse_zero", "jomopans_lex_mae_db", "roc_auc", "pr_auc", "ece_10", "brier"
]
display(std_full[std_full["split"].isin(["val_cal_hard", "holdout"])][cols_show].round(4))

print("Holdout Calibration-Tilt Stress (+8 dB/dec extreme sensor tilt):")
display(tilt_df[tilt_df["tilt_db_per_dec"] == 8.0].round(4))

print("Holdout Per-Station & Combined Stratum Breakdown (Ensemble_v4 vs Ensemble_v2):")
sub_st = strata_df[strata_df["model"].isin(["Ensemble_v2", "Ensemble_v4"])][
    ["scope", "group", "model", "n", "zindi_score_1_minus_rmse", "rmse_zero", "roc_auc", "jomopans_lex_mae_db", "ece_10"]
]
display(sub_st.sort_values(["scope", "group", "model"]).round(4))
"""
    )

    code(
        """
for title, rel in [
    ("Figure 1 · Standards Agreement, JOMOPANS L_excess & Calibration", "reports/standards_benchmark/figures/fig1_standards_calibration_and_agreement.png"),
    ("Figure 2 · Hydrophone Tilt Stress, Hard Strata & Unseen Holdout Sites", "reports/standards_benchmark/figures/fig2_boundary_stress_and_calibration_shift.png"),
    ("Figure 3 · Acoustic Signal Boundaries (ISO 18405, PCEN & RPCA Decomposition)", "reports/standards_benchmark/figures/fig3_acoustic_signal_boundaries_decomposition.png"),
]:
    print(title)
    display(Image(filename=str(ROOT / rel), width=880))
"""
    )

    md(
        """
## 5. End-to-End Reproducibility Verification (`v2`, `v3`, `v4`) & `v5` Blueprint Hook

Every submission artifact is audited below against `sample_submission.csv` (`1,500` rows, exact `ID` order, finite values in `[0, 1]`, SHA-256 checksums). I also reload `models/v4/bundle_v4.joblib` from disk and check `predict_from_bundle_v4` on `holdout.npz[:32]` for finite predictions and `+37.5 dB` invariance. If Submission #3 reveals residual false positives on uncalibrated ports, my documented **Modeling v5 Blueprint** (`docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md`) isolates `score_added_vessel` to strict `h(f)`-invariant `R_*` + `PCEN` + `RPCA` features and `Freq-MixStyle` / `ResNorm` without spectral tilt channel `ch4`.
"""
    )

    code(
        """
sample = pd.read_csv(ROOT / "data/zindi/raw/sample_submission.csv")
sub_rows = []
subs = {}
for tag, rel in [
    ("candidate_v2_ensemble (Submitted #2)", "submissions/candidate_v2_ensemble.csv"),
    ("candidate_v3_ensemble (Intermediate v3)", "submissions/candidate_v3_ensemble.csv"),
    ("candidate_v4_ensemble (Recommended #3)", "submissions/candidate_v4_ensemble.csv"),
]:
    p = ROOT / rel
    df = pd.read_csv(p)
    assert list(df.columns) == list(sample.columns) and len(df) == 1500
    assert (df["ID"] == sample["ID"]).all()
    assert df["share_added_vessel"].between(0.0, 1.0).all()
    assert df["score_added_vessel"].between(0.0, 1.0).all()
    subs[tag] = df
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    sub_rows.append({
        "Candidate": tag,
        "Rows": len(df),
        "Share Mean": round(float(df["share_added_vessel"].mean()), 4),
        "Share Median": round(float(df["share_added_vessel"].median()), 4),
        "Score Mean": round(float(df["score_added_vessel"].mean()), 4),
        "Score > 0.95": round(float((df["score_added_vessel"] > 0.95).mean()), 4),
        "SHA-256 (16)": sha,
    })
display(pd.DataFrame(sub_rows))

# Live bundle_v4 reload + continuous +37.5 dB check
bundle_v4 = joblib.load(ROOT / "models/v4/bundle_v4.joblib")
sh_a, sc_a = predict_from_bundle_v4(bundle_v4, X_sub)
sh_b, sc_b = predict_from_bundle_v4(bundle_v4, X_sub + 37.5)
print(
    f"Live bundle_v4 reload check -> Share max|diff|={np.max(np.abs(sh_a - sh_b)):.2e}, "
    f"Score max|diff|={np.max(np.abs(sc_a - sc_b)):.2e} (n={len(X_sub)} holdout windows)"
)
assert np.max(np.abs(sh_a - sh_b)) < 1e-6 and np.max(np.abs(sc_a - sc_b)) < 1e-6
"""
    )

    md(
        """
## 6. Held-Out South African Site-Level Relative Acoustic Activity & Observed-Signal Support (Zindi Staff Compliance)

Per official Zindi staff spatial-rubric guidance (`daniil_zindi`, 24 Nov 2026), an uninterpolated total-sound map provides context only. Below, I load `reports/standards_benchmark/sa_site_relative_vessel_activity.csv` and `reports/standards_benchmark/sa_heldout_signal_support.csv` (generated strictly over **17,487 held-out real 15-minute windows** across `6` future development periods and `2` unseen holdout sites `09_Fish_Hoek` & `10_Sea_Point`, with zero training or `Test.csv` overlap), contrast `h(f)`-invariant **relative acoustic activity** against total-sound Baseline `B0` and analytical `Physics_Robust_q20` with **1,000-draw calendar-day block bootstrap 95% CIs**, and display the upgraded 3-panel **Figure 2**.
"""
    )

    code(
        """
sa_site = pd.read_csv(ROOT / "reports/standards_benchmark/sa_site_relative_vessel_activity.csv")
sa_sup = pd.read_csv(ROOT / "reports/standards_benchmark/sa_heldout_signal_support.csv")
display(sa_site[[
    "display_name", "eval_scope", "windows_heldout", "calendar_days", "provisional_low_days",
    "rel_vessel_activity_pct", "rel_ci95_low_pct", "rel_ci95_high_pct",
    "total_sound_median_db", "physics_q20_mean_share", "ensemble_v4_mean_share"
]].round(4))
display(sa_sup.round(4))
display(Image(filename=str(ROOT / "reports/standards_benchmark/figures/fig2_site_relative_acoustic_activity_map.png"), width=920))
"""
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NOTEBOOK_PATH)
    args = parser.parse_args()
    nb = build_notebook()
    md_text = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    plural_hits = re.findall(r"\b(?:we|our|ours|ourselves|us)\b", md_text, flags=re.IGNORECASE)
    assert not plural_hits, f"First-person plural pronouns found in Notebook 03 markdown: {plural_hits}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, args.out)
    print(f"Wrote {args.out} ({len(nb['cells'])} cells, {len(md_text)} markdown chars)")


if __name__ == "__main__":
    main()
