# Project Learning Log

Concise chronological record of verified data findings, modelling experiments, and safeguards.

## 1. Data & Integrity Audit
- **South African archive:** 1,715,779 1-minute rows across 10 sites (`stations.csv` lists working coordinate candidates; site 13 Saldanha Bay has two unresolved alternatives). 62 `TOL_*` columns include all 25 competition bands (31.6–7943.3 Hz). Nominal `12000` Hz recorder rows contain finite 6.3/7.9 kHz values (export convention unverified).
- **15-min background index (`reports/eda/background_window_index.csv`):** 88,143 non-overlapping complete windows across 8 stations (Cape Recife max run = 13, Riy Banks max run = 13 -> excluded).
- **Chronological split v2 (`reports/eda/background_split_manifest_v2.csv`):** 67,463 train / 16,946 validation / 3,193 gap-excluded / 541 station holdout (Fish Hoek 297, Sea Point 244). Zero run overlap; 9–49 day train/val gaps.
- **`Test.csv` (inference-only):** 1,500 × 375 features; 18,350 missing cells across 725 rows, all occurring as whole 25-band minute omissions (624 missing `m01`, 100 missing `m15`). Never used for fitting or calibration.
- **VTUAD (`v1.3.0`, Concept DOI `10.5281/zenodo.22274549`):** 525 mono 32 kHz PCM_16 WAVs passed SHA-256 verification. Key pitfalls found:
  1. 22 files in `vessel/` have `label == 'background'` (`class_code == 0.0`) -> isolated with the 16 `background/` files (38 total).
  2. Strict-interval WAV durations match `span_sec`, whereas 168 differ from `metadata.csv`'s `duration_sec` by up to 300 s -> always read actual WAV frames.
  3. Publisher split leaks 57/178 hulls across splits -> rebuilt hull-disjoint split v2 (`410` train / `77` val clips, 0 shared hulls; `dredger` & `pilotvessel` have 1 hull each -> train-only).

## 2. Synthetic & Leaderboard Progression
- **Exp 001 (200 train / 80 val, discrete shares):** 375-feature RF RMSE `0.197845` vs constant `0.249648`. Feature-family ablation showed 96 combined global/spectral/temporal summaries dropped exploratory RMSE to `0.119687`.
- **Exp 002 (1,200 train / 300 val, station-balanced, continuous shares):** Preselected 96-feature RF reached RMSE `0.112061`, MAE `0.073236` (constant `0.250364`, raw 375 `0.144499`, 50 spectral `0.147717`). Errors concentrated on 1–2 min weak additions (22/30 worst errors).
- **Submission 001 (`pBUapzRu`, commit `aaf4748`):** Fixed per-window max-shift pipeline scored provisional public `1 - RMSE = 0.761536` (`share RMSE = 0.238464`), diagnostic `score_added_vessel` AUC `0.517533` (shared same output as `share_added_vessel`, no separate classifier). On synthetic Exp 002 val, that same pipeline gave RMSE `0.166477` and AUC `0.953386` — confirming severe synthetic-to-public domain shift.
- **Exp 003 (48 matched pairs × 6 masks):** Original representation beat `submitted_max_shift` on 4/6 masks (complete RMSE `0.127155` vs `0.178687`), while first-minute and both-edge masks were sensitive to single-source deletions.
- **Exp 004 (frozen source pools, `exp004_eval` n=300):** Original RF RMSE `0.114430` (AUC `0.981296`) vs max-shift RF `0.150893` (AUC `0.978653`).

## 3. Spectral Contrast, Source-Disjoint Exp 007 & Descriptive Synthesis
- **Real-acoustic 6-station audit (`exp004_train`, 460 runs, cap 120/site):** Median run-window means span `61.520` (Saldanha Bay) to `90.604` (Bird Island) in source-export units; peak-normalized spectral-shape RMS distances vs VTUAD range from `7.477` (St Francis) to `26.202` (Durban, `n=1`).
- **Exp 005 (5,520 scenarios, 1,840 triplets):** Replacing VTUAD spectra with background-matched controls significantly reduced within-window temporal spectral excursion (median `+3.50` to `+6.57` dB on 1-min events), showing models could exploit unnatural source–background spectral contrast.
- **Exp 006 / 006B (120 internal dev pairs):** Contrast-varied training reduced RMSE (`0.133224 -> 0.114353`) and lifted AUC (`0.847977 -> 0.877137`), but raised non-injected-control FPR at 0.50 (`0.233333 -> 0.375000`). Exp 006C found 112/120 pairs in one connected source component, prompting a clean pre-pairing split in Exp 007.
- **Exp 007 (source-disjoint train/calibration/assessment, 120 assessment pairs):**
  - Contrast-varied RF reduced assessment RMSE `0.114236 -> 0.094566` (`-0.019670`, ~17.2%; run-bootstrap 95% CI `[-0.00041, 0.04230]`) and raised AUC `0.848328 -> 0.886421`.
  - Fixed-0.50 non-injected FPR increased `0.16667 -> 0.30000`; calibration-selected threshold (`0.55`, targeting FPR ≤ 0.25 on calibration) yielded assessment FPR `0.23333` and TPR `0.85294`.
  - **Hybrid policy** (contrast-varied regressor + original 0.50 detector) pairs `0.094566` RMSE with `0.16667` FPR, capturing the accuracy/specificity trade-off.
