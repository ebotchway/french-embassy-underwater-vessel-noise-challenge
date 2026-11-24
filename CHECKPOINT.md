# Project Checkpoint

Updated: 24 November 2026. Status: **SUBMISSIONS #1–#4 (`pBUapzRu`, `1zX9R2QL`, `ovjzdeb8`, `MAKNM3KB`) SCORED LIVE, CANDIDATE #5 (`v5_blend`) VERIFIED & READY FOR UPLOAD, 10-PAGE RESEARCH PDF COMPILED, EXECUTED NOTEBOOKS `01–03`, AND 43-SCRIPT CODE ARCHIVE VERIFIED** (deadline **13 December 2026, 11:30 UTC**).

## Leaderboard & Domain-Generalization Progression

- **Submission #1 — Historical Baseline (`reports/submissions/first_candidate.csv`, ID `pBUapzRu`, commit `aaf4748`):**
  - Public LB `1 - RMSE`: `0.761535762` (`share RMSE`: `0.238464237`), diagnostic `ROC AUC`: `0.517533333`.
- **Submission #2 — Modeling v2 (`submissions/candidate_v2_ensemble.csv`, ID `1zX9R2QL`, SHA-256 `afb088e7356b579f398a8565bf600d163a4aabd1ecfc12b07e6e19002049c85c`):**
  - Public LB `1 - RMSE`: `0.775017430` (`share RMSE`: `0.224982569`), diagnostic `ROC AUC`: `0.548377777` (Rank `508`).
  - Synthetic Validation (`val`, `n=1,500`): `0.8747` (`AUC 0.9725`); Unseen Holdout (`09_Fish_Hoek` + `10_Sea_Point`, `n=541`): `0.8833` (`AUC 0.9365`; unified v4 holdout benchmark: `0.8886` / `AUC 0.9334`).
  - **Diagnosis:** Revealed sensitivity to unmodeled hydrophone frequency response $h(f)$ (spectral tilt/ripple) and sharp South African coastal transients (`vlf_surf`, `reef_active`, `glitch_drop`).
- **Candidate v3 — Hard-Negative + Linear-Excess Intermediate (`submissions/candidate_v3_ensemble.csv`, SHA-256 `883e5dae60cb4826ee3816cbc64681c1faf3d64bf1e2edfe392e0de87237d663`):**
  - 180 features, `10,000` train / `2,000` `val_standard` / `1,500` `val_hard_event` / `541` `holdout` (`41.6%` zero windows), 7-channel SE-ResNet CNN (`CNN2D_v3`) + Hurdle GBDTs.
  - Lifted `val_cal_hard` to `0.8788` (`AUC 0.9634`), `val_standard` to `0.9007` (`AUC 0.9743`), and unseen `holdout` to `0.8971` (`AUC 0.9686`).
  - **Diagnostic discovery:** Feature audit caught `proto_spec_logit` taking `47.0%` of LightGBM gain (`25.8%` mean norm) by memorizing ONC-vs.-SA hydrophone calibration tilt.
- **Submission #3 — Modeling v4 (`submissions/candidate_v4_ensemble.csv`, ID `ovjzdeb8`, SHA-256 `e8cf38a590b8318efd9eadbbc242cb4635d22690f2fd69c2e7c4be2ead0a245e`, Best Live `1-RMSE`):**
  - **Live Public LB (`24 Nov 2026`):** `1 - RMSE`: **`0.775235755`** (`share RMSE`: **`0.224764244`**), diagnostic `ROC AUC`: `0.556555555` (exact integer-fraction test over `45,000 = 150 x 300` public windows).
  - 205 features including `71` exact $h(f)$-invariant `R(t,f)` residuals, five strict physical strata, `65.3%` calibration warping in training (`100%` in `val_cal_hard`), and zero custom `__main__` classes in `models/v4/bundle_v4.joblib`.
  - **`val_cal_hard` (`n=1,500`):** `0.8933` `1-RMSE` (`AUC 0.9800`, `zero_rmse 0.0664`, `Lin CCC 0.9204`, `JOMOPANS L_ex MAE 0.547 dB`, `ECE_10 0.0273`); **Unseen Holdout (`n=541`):** `0.9219` (`AUC 0.9737`, `zero_rmse 0.0329`, `JOMOPANS L_ex MAE 0.308 dB`).
  - **Pathology Diagnosed After Sub #3:** Only `+0.00022` live public lift despite `0.9219` local holdout revealed **simulator overfitting** (`134/205` tabular features and `CNN2D_v4` channel `ch4` remained tilt-sensitive, `Test.csv` scores saturated at `0.8571` mean / `0.9753` median, and marking real SA `strict_hf_ship` as `y = 0` suppressed coastal vessel signatures).
- **Submission #4 — Modeling v5-Simple (`submissions/candidate_v5_simple.csv`, ID `MAKNM3KB`, SHA-256 `d163e749b707d6cbf1fb242aa06bdefd71ae6aeb848882eaae30b0c4a3b0e5ab`, Best Live `ROC AUC`):**
  - **Live Public LB (`24 Nov 2026`):** Diagnostic **`ROC AUC`: `0.563911111`** (**highest live AUC achieved**, `+0.0074` over Sub #3 and `+0.0464` over Sub #1), `1 - RMSE`: `0.771209221` (`share RMSE`: `0.228790778`).
  - `scripts/modeling_v5_simple.py`: `42` purely $h(f)$-invariant features (`R(t,f)` zones + smooth sigmoids + per-band centered `PCEN` + `RPCA`), `0` real `strict_hf_ship` suppression in `y=0`, broadband `-12..+6 dB/dec` CPA warps, `3` regularized learners (`Linear_v5`, shallow `LightGBM_v5`, shallow `CatBoost_v5`). `Test.csv` `+10 dB/dec + ripple` warp diff `2.89e-33` share / `7.77e-16` score; lowest `+8 dB/dec` `zero_rmse` (`0.0787`); restored real SA ship response (`0.5005` vs `0.2357`).
- **Candidate #5 — Modeling v5-Blend (`submissions/candidate_v5_blend.csv`, SHA-256 `b76b2f23de95ddd8e8bf0086e8fb7fe15e38a53b3cedd69d658f222d7ff111ce`, Ready Upload):**
  - `scripts/modeling_v5_blend.py`: Strictly row-independent, zero-tuning **50/50** convex blend pairing `Ensemble_v4` (broadband energy magnitude) and `Ensemble_v5_Simple` (pure $h(f)$-invariant vessel detection; `Test.csv` share corr `0.6864`, RMS diff `0.1672`).
  - Beats **both** parent ensembles under `+8 dB/dec` holdout tilt stress on `1 - RMSE` (**`0.8390`** vs `0.7993` `v4` / `0.8382` `v5_simple`), `logloss` (**`0.3310`** vs `0.3704` / `0.3578`), and `Brier` (**`0.1026`** vs `0.1229` / `0.1113`), while separating real SA `strict_hf_ship` cleanly above ambient (`0.3541` vs `0.2072`). On `Test.csv`: `share mean = 0.2649` (`std = 0.1874`), `score mean = 0.7905`.

## Research Write-up, Notebooks, Standards & Reproducibility Track

- **Primary Research PDF (`deliverables/overleaf_manuscript/research_manuscript_latex.pdf`):** `10` pages (`3,290,317` bytes, `<10 MB`), `0` `Overfull \hbox`, `0` undefined refs/citations, `28/28` APA 7th citations, `23/23` labeled equations/tables/figures (`8` figures, `7` tables, `8` equations), SHA-256 `df6dc1979bb86afe15eb12a08039b9e25f35ada55a369254a67d547c992e5c28`.
- **Overleaf Upload ZIP (`deliverables/overleaf_manuscript/overleaf_research_manuscript.zip`):** `2,679,317` bytes (`11` clean entries), SHA-256 `838085a109e2a0dceb1a83b44be69322730b75df368df31d868ffafab6b7b9fe`.
- **Unified Code Archive (`deliverables/overleaf_manuscript/research_and_modeling_code_final.zip`):** `< 10 MB`, packages `43` verified scripts, all `3` executed notebooks (`01`, `02`, `03`), documentation, and TeX sources; clean-copy extraction passes `py_compile` and `--self-test-only` across `v2`, `v3`, `v4`, `v5_simple`, `v5_blend`, `standardized_evaluation_suite`, and `evaluate_sa_relative_vessel_activity` (canonical hash tracked in `deliverables/overleaf_manuscript/OVERLEAF_MANIFEST.json` and `reports/reproducibility_manifest.json`).
- **Legacy V6 Research Archive (`deliverables/research_submission/`):** Preserved (`5` pages, `research_code_v6.zip` SHA-256 `02431f367aa464a2cebf24e70bac5aeb22775b000f2e4224834e7838d9568eb1`).
- **Notebooks (`notebooks/`):** Updated & executed `01_south_african_acoustic_eda.ipynb`, `02_vessel_noise_modeling.ipynb`, and `03_domain_generalization_v3_v4_and_standards.ipynb` (`v2 -> v3 -> v4 -> v5_simple -> v5_blend` + Standards + Zindi Staff Site Activity).
- **Master Verifier (`scripts/reproduce_pipeline.py --mode verify-only`):** `9` self-tests + `5` submission CSV checks PASS.

## Completed Human Actions & Final Upload Checklist

1. **Completed:** External data disclosure posted ([Discussion #36419](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36419)), organizer map clarification answered (`daniil_zindi`, `24 Nov 2026`) and implemented (`docs/ORGANIZER_MAP_CLARIFICATION.md`), Submissions `#1` (`pBUapzRu`), `#2` (`1zX9R2QL`), `#3` (`ovjzdeb8`), and `#4` (`MAKNM3KB`) uploaded and scored, and `Candidate #5` (`v5_blend`) audited.
2. **Final Upload Actions (Before 13 Dec 2026, 11:30 UTC):**
   - Upload `submissions/candidate_v5_blend.csv` as **Submission #5** on Zindi (combining `#3 ovjzdeb8` and `#4 MAKNM3KB`), and select your two top-performing diverse submissions on Zindi's Private LB checkboxes.
   - Upload `deliverables/overleaf_manuscript/research_manuscript_latex.pdf` (`10` pages) and `deliverables/overleaf_manuscript/research_and_modeling_code_final.zip` (`< 10 MB`) to the Research Track.
