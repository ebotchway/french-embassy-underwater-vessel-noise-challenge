# Project Checkpoint

Updated: 25 November 2026. Status: **ALL 5 LIVE ZINDI SUBMISSIONS SCORED (BEST: SUBMISSION #5 `a6JJCya4` AT `0.787467941` `1 - RMSE` / `0.572400000` `ROC AUC`), COMPLETE OFFLINE ACTUALS RECONCILED, 10-PAGE RESEARCH PDF & 43-SCRIPT CODE ARCHIVE VERIFIED & SUBMITTED** (deadline **13 December 2026, 11:30 UTC**).

## End-to-End Offline & Live Leaderboard Summary

| Stage / ID | Architecture & Features | Val Cal-Hard (`1-RMSE`/AUC) | Unseen Holdout (`1-RMSE`/AUC) | `+8 dB/dec` Tilt (AUC / zRMSE / LL) | Live Zindi `1-RMSE` | Live Share `RMSE` $\downarrow$ | Live `ROC AUC` $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: |
| **Sub #1 (`pBUapzRu`)** | `V1_RF_MaxShift` (`96` max-shifted) | — | `0.8390 / 0.8980` | — | `0.761535762` | `0.238464237` | `0.517533333` |
| **Sub #2 (`1zX9R2QL`)** | `Ensemble_v2` (`138` shift-inv + CNN; v2 ho `0.8833`) | `0.8364 / 0.9312` | `0.8886 / 0.9334` | `0.8657 / 0.2023 / 1.1879` | `0.775017430` | `0.224982569` | `0.548377777` |
| **Cand v3 (`---`)** | `Ensemble_v3` (`180` + `proto_spec_logit` + CNN) | `0.8788 / 0.9634` | `0.8971 / 0.9686` | `0.9477 / 0.2045 / 0.4625` | — | — | — |
| **Sub #3 (`ovjzdeb8`)** | `Ensemble_v4` (`205` incl. `71` $R_*$ + 7-ch CNN) | **`0.8933 / 0.9800`** | **`0.9219 / 0.9737`** | **`0.9561`** / `0.0929 / 0.3704` | `0.775235755` | `0.224764244` | `0.556555555` |
| **Sub #4 (`MAKNM3KB`)** | `Ensemble_v5_Simple` (`42` pure $h(f)$-inv, 3 reg.) | `0.7843 / 0.8765` | `0.8382 / 0.9194` | `0.9194` / **`0.0787`** / `0.3578` | `0.771209221` | `0.228790778` | `0.563911111` |
| **Sub #5 (`a6JJCya4`)** | `Ensemble_v5_Blend` (`50/50` `v4` + `v5_simple`) | `0.8542 / 0.9662` | `0.8914 / 0.9629` | `0.9490 / 0.0807` / **`0.3310`** | **`0.787467941`** | **`0.212532058`** | **`0.572400000`** |

## Key Verified Actuals & Physical Reconciliation

- **Submission #1–#2 (`V1` -> `V2`):** Replacing window max-shifting with $R_0 = 100$\,dB physical superposition and 2s Hann base-10 deci-decades cut unseen holdout RMSE by `27.5%` (`0.8390 -> 0.8833`) and lifted live LB `0.7615 -> 0.7750` (Rank `508`).
- **Candidate v3 -> Submission #3 (`ovjzdeb8`):** Catching `proto_spec_logit` (`47.0%` LGB tilt gain in v3) and deriving `71` exact $h(f)$-invariant `R(t,f)` residuals lifted local `holdout` to `0.9219` (`zero_rmse 0.0329`, `Lin CCC 0.924`, `JOMOPANS L_ex MAE 0.308 dB`, `val_cal_hard CCC 0.920` / `ECE_10 0.0273`), yet yielded only `+0.00022` live lift (`0.775235755`) because `134/205` features + CNN `ch4` remained tilt-sensitive (`Test.csv` score median `0.9753`) and `y=0` `strict_hf_ship` suppressed real coastal ships.
- **Submission #4 (`MAKNM3KB`, `scripts/modeling_v5_simple.py`):** Stripping strictly to `42` purely $h(f)$-invariant features (`R(t,f)`, smooth sigmoids, centered `PCEN`, `RPCA`) without real-ship suppression intentionally traded offline fit (`v5_val_100pct_cal 0.8302 / 0.9064 AUC`, `v5_holdout 0.8357 / 0.8881 AUC`, `v4_holdout 0.8382`) for `2.89e-33` `Test.csv` warp drift, flat `+8 dB/dec` `zero_rmse 0.0787`, restored real SA ship sensitivity (`0.5005` vs `0.2357`), and lifted live diagnostic `ROC AUC` to `0.563911111` (at `0.771209221` `1 - RMSE` due to lost inter-band loudness scale).
- **Submission #5 (`a6JJCya4`, `scripts/modeling_v5_blend.py`):** Because `ovjzdeb8` and `MAKNM3KB` have `0.1672` `Test.csv` RMS difference (`corr = 0.6864` share / `0.5424` score), a zero-tuning `50/50` blend achieves **`0.8390` `1 - RMSE`** (`logloss 0.3310`, `Brier 0.1026`) under `+8 dB/dec` stress (beating both parents offline) and jumps live to **`0.787467941` (`RMSE = 0.212532058`)** and **`0.572400000` ROC AUC** (`25,758 / 45,000`), matching the exact MSE diversity identity (`0.1583` implied public-split RMS diff).

## Research Write-up, Notebooks & Reproducibility Track

- **Primary Research PDF (`deliverables/overleaf_manuscript/research_manuscript_latex.pdf`):** `10` pages (`3,291,507` bytes, `<10 MB`), `0` `Overfull \hbox`, `0` undefined refs/citations, `28/28` APA 7th citations, `23/23` labeled equations/tables/figures (`8` figures, `7` tables, `8` equations), SHA-256 `268ca0b915105736775f4e0831653a14449852a1128ef4f402e9d339de3ff1b5`.
- **Overleaf Upload ZIP (`deliverables/overleaf_manuscript/overleaf_research_manuscript.zip`):** `2,679,440` bytes (`11` clean entries), SHA-256 `f14e623e18020bd35d76e329a1d8fec0360db21012c6b672a899680f4938edd9`.
- **Unified Code Archive (`deliverables/overleaf_manuscript/research_and_modeling_code_final.zip`):** `< 10 MB`, packages `43` verified scripts, all `3` executed notebooks (`01`, `02`, `03`), documentation, and TeX sources; clean-copy extraction passes `py_compile` and `--self-test-only` across `v2`, `v3`, `v4`, `v5_simple`, `v5_blend`, `standardized_evaluation_suite`, and `evaluate_sa_relative_vessel_activity` (canonical hash tracked in `deliverables/overleaf_manuscript/OVERLEAF_MANIFEST.json` and `reports/reproducibility_manifest.json`).
- **Notebooks (`notebooks/`):** Updated & executed `01_south_african_acoustic_eda.ipynb`, `02_vessel_noise_modeling.ipynb`, and `03_domain_generalization_v3_v4_and_standards.ipynb` (`v2 -> v3 -> v4 -> v5_simple -> v5_blend`, offline + live actuals, Standards, and Zindi Staff `17,487`-window site activity).
- **Master Verifier (`scripts/reproduce_pipeline.py --mode verify-only`):** `9` self-tests + `5` submission CSV checks PASS.

## Final Competition Checklist

1. **Private LB Selection (`2` checkboxes before 13 Dec 2026, 11:30 UTC):** Select **`a6JJCya4` (`Sub #5`, `0.78747` / `0.5724` AUC)** + **`ovjzdeb8` (`Sub #3`, `0.77524`)** (or `MAKNM3KB` if maximizing pure tilt resilience).
2. **Research Track Re-upload (Optional):** Initial PDF & ZIP submitted (`25 Nov 2026`); updated PDF (`3,291,507` B) & code ZIP now also include live `Sub #5 a6JJCya4` (`0.787467941` / `0.5724`) and can be re-uploaded anytime before deadline.
