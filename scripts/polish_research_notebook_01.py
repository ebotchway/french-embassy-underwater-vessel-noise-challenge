#!/usr/bin/env python3
"""Update Notebook 01 markdown cells so every section states Finding -> Choice Driven cleanly."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB01_DEFAULT = ROOT / "notebooks/01_south_african_acoustic_eda.ipynb"

UPDATES: dict[int, str] = {
    0: """# South African Underwater Acoustic EDA & Decision Log

Audit of `1,715,779` rows across 10 monitoring stations, 25 competition bands (`31.6–7943.3 Hz`), split construction, and Experiments 001–007. Each finding explicitly states the modeling choice carried forward into `02_vessel_noise_modeling.ipynb`. (`Test.csv` headers/masks only — zero fitting or calibration.)""",
    6: """## 2. Station Coverage & Missing Bands

- **Finding:** `1,715,779` rows span 10 sites, heavily skewed toward `01_Richards_Bay` (`58.35%`); extra raw `TOL_*` columns outside `31.6–7943.3 Hz` are mostly missing.
- **Choice driven:** Enforce station-balanced sampling (prevent Richards Bay dominance) and restrict features to the 25 active competition bands.""",
    9: """## 3. Competition 25-Band Schema Check

- **Finding:** All 25 competition third-octave columns exist across the station parquets with zero missing names.
- **Choice driven:** Lock canonical column order `TOL_31.6` .. `TOL_7943.3` identically across extraction, simulation, and `Test.csv` inference.""",
    13: """## 4. 25-Band Coverage Across Stations

- **Finding:** `100.00%` of raw rows in the South African archives contain complete, finite values across all 25 competition bands.
- **Choice driven:** Partial-band hardware dropouts do not happen in SA telemetry; `Test.csv` missingness comes solely from dropped whole minutes.""",
    17: """## 5. Temporal Continuity & 15-Minute Window Availability

- **Finding:** Zero duplicate timestamps exist, yielding `88,143` non-overlapping continuous 15-minute windows across 8 stations.
- **Choice driven:** Slice `(15, 25)` matrices strictly inside unbroken runs (`never stitch across timestamp gaps`).""",
    21: """## 6. Recording Interval Diagnostics

- **Finding:** `17` recorder configurations vary widely in cadence (`1–360 s`) and source-export dB baseline offsets without physical hydrophone calibration.
- **Choice driven:** Prohibit raw uncentered dB across sites; engineer strictly global-offset-invariant features (`X - median(X)`).""",
    24: """## 7. Continuous Run-Length Distributions

- **Finding:** `03_Cape_Recife` and `06_Riy_Banks` have max continuous run length `< 15` minutes (`0` complete 15-minute windows).
- **Choice driven:** Exclude Cape Recife and Riy Banks cleanly from window simulation pools.""",
    28: """---
# Part II — Splits, VTUAD Extraction & Synthetic Experiment Decisions

Structured experimental evidence diagnosing V1 baseline limits and locking the physics and validation choices for Notebook 02.""",
    30: """## 8. 15-Minute Background Window Index

- **Finding:** Window supply spans `52,522` (`Richards_Bay`) down to `297` (`Fish_Hoek`) and `244` (`Sea_Point`) across `18,169` continuous runs.
- **Choice driven:** Equalize background representation per station inside `train` (`6,000`) and `val` (`1,500`).""",
    32: """## 9. Chronological & Run-Isolated Background Split (v2)

- **Finding:** Run-grouped splitting with 9–49 day chronological gaps leaves `67,463` train / `16,946` val / `3,193` excluded, plus `541` untouched `station_holdout` windows (`Fish_Hoek`, `Sea_Point`).
- **Choice driven:** Evaluate every candidate model on both in-station `val` and completely unseen `station_holdout` stations.""",
    34: """## 10. VTUAD v1.3.0 Verification & Metadata Audit

- **Finding:** All `525` WAVs passed SHA-256 checks, `22` files stored inside `vessel/` are actually `background`, and `167` WAV durations differ `> 60 s` from metadata.
- **Choice driven:** Filter strictly by verified metadata `label != 'background'` and read exact WAV frames rather than trusting folders or durations.""",
    36: """## 11. Hull-Disjoint Vessel Source Split (v2)

- **Finding:** Grouping by `hull_id` yields zero overlap between `train` (`410` clips, `137` hulls) and `validation` (`77` clips, `35` hulls), isolating `38` background references.
- **Choice driven:** Enforce zero `bg_run_id` and zero `hull_id` / `vessel_group` overlap across CV folds, `val`, and `holdout`.""",
    38: """## 12. VTUAD 25-Band Relative Power Extraction

- **Finding:** Welch (`nperseg=32768`, `2^(1/6)` base-2 edges) extracted `1,892` vessel minutes (`49%` `tug`, zero small coastal craft).
- **Choice driven:** Upgrade extraction to starter-matched 2s Hann deci-decade (`10^0.05` base-10) and add `96` HearMyShip coastal craft WAVs across `8` classes.""",
    40: """## 13. Synthetic Experiment 001 (Discrete Target Shares)

- **Finding:** Exp 001 (`200/80` windows, discrete shares `0–0.70`) subtracted per-window `max(background_db)`, warping ambient levels.
- **Choice driven:** Establish controlled Exp 001 baseline before scaling features or data.""",
    42: """## 14. Experiment 001 Baseline Check

- **Finding:** Raw 375-feature RF beat Constant Mean (`0.197845` vs `0.249648` RMSE), whereas unscaled `ridge_alpha_10` degraded (`0.266875`).
- **Choice driven:** Standardize regularized linear models and replace high-variance raw 375 grids with structured summary features.""",
    43: """## 15. Experiment 001 Provenance Inspection

- **Finding:** Exp 001 inherited Richards Bay skew (`119/200` train rows) and tug oversampling.
- **Choice driven:** Stratify both background stations and vessel classes evenly in Exp 002 and Modeling v2.""",
    45: """## 16. Experiment 001 Prediction Diagnostics

- **Finding:** Raw-feature RF compressed predictions (`0.045–0.583`), under-predicting high traffic shares and over-predicting zeros.
- **Choice driven:** Inspect error by target share bin and train separate regression (`share`) and detection (`score`) heads.""",
    47: """## 17. Experiment 001 Error Breakdown by Target Share

- **Finding:** Zero-addition windows suffered `+0.190` mean positive bias while `0.70` windows were systematically under-predicted.
- **Choice driven:** Allocate `~22%` exact `0.0` background windows and expand continuous targets up to `0.992`.""",
    49: """## 18. Feature Family Ablation (Exploratory)

- **Finding:** Hand-crafted `combined_96` features outperformed `raw_375` (`0.119687` vs `0.196201` RMSE).
- **Choice driven:** Build richer shift-invariant spectral/temporal/co-modulation features (`138` dims) plus a spatial 2D CNN over `(15, 25)` grids.""",
    51: """## 19A. Experiment 002 Plan (Completed)

- **Finding:** Scaling to `1,200/300` station-balanced continuous-share windows confirmed data-pipeline integrity.
- **Choice driven:** Scale Notebook 02 simulator to `6,000` train / `1,500` val / `541` unseen-station holdout windows.""",
    52: """## 19B. Experiment 002 Synthetic Benchmarks

- **Finding:** `rf_combined_96` achieved `0.112061` clean synthetic RMSE, yet leaked generalization gaps when deployed to Submission 001 (`0.7615` LB, `0.5175` AUC).
- **Choice driven:** Root-cause the synthetic-to-Test gap via representation, missingness, and contrast audits (Sections 19D–25).""",
    54: """## 19C. Experiment 002 Failure Analysis by Duration & Station

- **Finding:** Errors spiked on 1-minute transients (`RMSE 0.1391`) and high-variance stations.
- **Choice driven:** Engineer smooth 3-point envelope, quadratic CPA concavity, lag-1 autocorrelation, and cross-band co-modulation features.""",
    56: """## 19D. Per-Window Max-Level Shift Assumption

- **Finding:** Subtracting `max(background_db)` anchors silent windows near loud spikes and distorts energy ratios (`Exp 001–002`).
- **Choice driven:** Remove per-window max-shift (`use fixed REF_DB=100` in simulator; center features by `X - median(X)`).""",
    57: """## 19E. Data-to-Model Alignment Audit

- **Finding:** Grouping errors by hull and continuous run revealed variance hidden by pooled window metrics.
- **Choice driven:** Partition `train` into 5 run- and hull-disjoint CV folds so OOF ensemble selection cannot memorize stations or vessels.""",
    59: """## 20. Phase 3: Test Missingness, Mask Robustness & Submission 001

- **Finding:** `725/1,500` `Test.csv` rows have whole-minute `NaN` gaps (`m01` in `624` rows, `m15` in `100`); Submission 001 scored `0.761536` (`0.517533` AUC) due to max-shift + naive linear edge interpolation.
- **Choice driven:** Inject matching whole-minute `NaN` dropouts in training/eval, recompute targets over observed cells only, and make all features/CNN masks natively handle `NaN`s.""",
    61: """## 21. Experiment 004: Frozen Source-Pool Baselines

- **Finding:** Retaining original relative levels beat `submitted_max_shift` across all 5 test stations (`0.11443` vs `0.15089` RMSE).
- **Choice driven:** Benchmark legacy `V1_RF_MaxShift` explicitly in Notebook 02 to verify that `V2` features strictly beat it on identical splits.""",
    63: """## 22. Experiment 005: Spectral-Contrast Realism Sensitivity

- **Finding:** Unwarped ONC Strait-of-Georgia spectra differ sharply from SA background shapes, inflating synthetic detection separability.
- **Choice driven:** Inject propagation warps (`±6 dB/dec` tilt, modal ripple, partial background-shape blending) so models cannot cheat on sensor color.""",
    65: """## 23. Experiments 006 & 006B: Contrast Augmentation vs False Alarms

- **Finding:** Naive contrast variation improved RMSE (`-0.01887`) and AUC (`+0.02916`) while raising non-injected background false positives (`+0.14167` FPR).
- **Choice driven:** Combine channel-warped CPA/entry-exit/stationary transits with clean `0.0` background runs and co-modulation features to control false alarms.""",
    67: """## 24. Experiment 007: Source-Disjoint Policy Comparison

- **Finding:** Under strict run/hull-disjoint assessment, contrast-varied training cut RMSE to `0.09457` (AUC `0.8864`) and proved tuned detection thresholding is essential.
- **Choice driven:** Fit OOF simplex meta-weights strictly inside `train` CV across GBDTs + `CNN2D` with separate share and log-loss score heads.""",
    69: """## 25. Six-Station Real-Acoustic Synthesis, Spectral Mismatch & Site Context

- **Finding:** Background baselines vary by ~`29 dB` across SA sites (`Saldanha 61.5` vs `Bird Island 90.6` source-export dB) and differ spectrally from Strait of Georgia VTUAD.
- **Choice driven:** Reserve `09_Fish_Hoek` and `10_Sea_Point` strictly as unseen-station testbeds in Notebook 02.""",
    71: """## 26. Research-to-Modeling Decision Matrix (Handoff to `02_vessel_noise_modeling.ipynb`)

| Notebook 01 Finding | Modeling Decision Carried Into Notebook 02 |
| --- | --- |
| Welch base-2 differed from starter 2s base-10 deci-decade; VTUAD skewed to large tugs | Re-extracted all `525` VTUAD WAVs + `96` HearMyShip coastal craft WAVs via exact 2s Hann base-10 deci-decade (`10^0.05`) |
| Uncalibrated SA recorder offsets span `~29 dB` across stations; V1 max-shift hurt Exp 004 (`0.151` vs `0.114`) | Fixed `REF_DB=100` in linear simulation + strict `+37.5 dB` shift-invariant centering (`X - median(X)`) in features & 2D CNN |
| `Test.csv` has whole-minute `NaN` dropouts (`m01` `41.6%`, `m15` `6.7%`) | Simulated matching whole-minute masks, recomputed ground truth on observed cells only, added native mask channel & nan-safe reductions |
| Unwarped Strait-of-Georgia spectra & abrupt steps inflated Exp 005–006 false alarms | Added CPA bell, entry/exit, stationary drift trajectories, `±6 dB/dec` tilts, modal ripple, and `35%` background-shape blending |
| Single `RandomForest` weak at extremes & blind to 2D spatiotemporal envelopes | Benchmarked `11` deterministic models (`Constant_Mean`, `Physics_q15`, `V1_RF_MaxShift`, `RF_v2`, `Linear`, `HistGBM`, `LightGBM`, `XGBoost`, `CatBoost`, `CNN2D`, `Ensemble_v2`) |
| Pooled random splits risk leaking runs, hulls, or stations | Disjoint `bg_run_id` & `hull_id`/`vessel_group` across 5 CV folds, `val`, and unseen stations (`Fish_Hoek`, `Sea_Point`) |""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NB01_DEFAULT)
    args = parser.parse_args()

    nb = nbf.read(args.notebook, as_version=4)
    for idx, text in UPDATES.items():
        assert nb.cells[idx].cell_type == "markdown", f"Expected markdown at cell {idx}"
        nb.cells[idx].source = text.strip() + "\n"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_chars = sum(len(c.source) for c in nb.cells if c.cell_type == "markdown")
    nbf.write(nb, args.notebook)
    print(f"Updated {args.notebook}: cells={len(nb.cells)} (code={len(code_cells)}), md_chars={md_chars}")


if __name__ == "__main__":
    main()
