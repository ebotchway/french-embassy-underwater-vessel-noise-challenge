# Project Checkpoint

Updated: 23 November 2026. Status: **SUBMISSION #3 (`Candidate v4`), INTERNATIONAL STANDARDS BENCHMARK, UPDATED RESEARCH PDF, AND REPRODUCIBLE CODE ARCHIVE VERIFIED & READY FOR UPLOAD** (deadline **13 December 2026, 11:30 UTC**).

## Leaderboard & Domain-Generalization Progression

- **Submission #1 — Historical Baseline (`reports/submissions/first_candidate.csv`, ID `pBUapzRu`, commit `aaf4748`):**
  - Public LB `1 - RMSE`: `0.761536` (`share RMSE`: `0.238464`), diagnostic `ROC AUC`: `0.517533`.
- **Submission #2 — Modeling v2 (`submissions/candidate_v2_ensemble.csv`, SHA-256 `afb088e7356b579f398a8565bf600d163a4aabd1ecfc12b07e6e19002049c85c`):**
  - Public LB `1 - RMSE`: `0.77501743` (`share RMSE`: `0.224982569`), diagnostic `ROC AUC`: `0.548377777` (Rank `508`).
  - Synthetic Validation (`val`, `n=1,500`): `0.8747` (`AUC 0.9725`); Unseen Holdout (`09_Fish_Hoek` + `10_Sea_Point`, `n=541`): `0.8833` (`AUC 0.9365`; on unified v4 holdout benchmark: `0.8886` / `AUC 0.9334`).
  - **Diagnosis:** Revealed sensitivity to unmodeled hydrophone frequency response $h(f)$ (spectral tilt/ripple) and sharp South African coastal transients (`vlf_surf`, `reef_active`, `glitch_drop`).
- **Candidate v3 — Hard-Negative + Linear-Excess Intermediate (`submissions/candidate_v3_ensemble.csv`, SHA-256 `883e5dae60cb4826ee3816cbc64681c1faf3d64bf1e2edfe392e0de87237d663`):**
  - 180 features, `10,000` train / `2,000` `val_standard` / `1,500` `val_hard_event` / `541` `holdout` (`41.6%` zero windows), 7-channel SE-ResNet CNN (`CNN2D_v3`) + Hurdle GBDTs.
  - Lifted `val_cal_hard` to `0.8788` (`AUC 0.9634`), `val_standard` to `0.9007` (`AUC 0.9743`), and unseen `holdout` to `0.8971` (`AUC 0.9686`).
  - **Diagnostic discovery:** Feature audit caught `proto_spec_logit` taking `47.0%` of LightGBM gain (`25.8%` mean norm) by memorizing ONC-vs.-SA hydrophone calibration tilt.
- **Submission #3 Candidate — Modeling v4 (`submissions/candidate_v4_ensemble.csv`, SHA-256 `e8cf38a590b8318efd9eadbbc242cb4635d22690f2fd69c2e7c4be2ead0a245e`, Recommended Upload):**
  - 205 features including **71 exact $h(f)$-invariant temporal excess residuals** $R(t,f) = X(t,f) - Q_{0.20, t' \in \Omega(f)}(X(t',f))$ (`+6 dB/dec + 2.5 dB cosine ripple` invariance diff `1.60e-14`), five strict physical strata (`strict_hf_ship`, `vlf_surf`, `reef_active`, `glitch_drop`, `ambient_normal`), `65.3%` calibration warping in training (`100%` in `val_cal_hard`), sharp `2–5 min` VTUAD blocks, and zero custom `__main__` classes in `models/v4/bundle_v4.joblib`.
  - **`val_cal_hard` (`n=1,500`, 100% $h(f)$-warped):** `Ensemble_v4` reaches **`0.8933`** `1-RMSE` (`AUC 0.9800`, `zero_rmse 0.0664`, `Lin CCC 0.9204`, `JOMOPANS L_ex MAE 0.547 dB`, `ECE_10 0.0273`).
  - **`val_standard` (`n=2,000`):** **`0.9137`** `1-RMSE` (`AUC 0.9837`, `zero_rmse 0.0494`).
  - **Unseen-Station Holdout (`09_Fish_Hoek` + `10_Sea_Point`, `n=541`):** **`0.9219`** `1-RMSE` (`RMSE 0.0781`, `AUC 0.9737`, `zero_rmse 0.0329`, `JOMOPANS L_ex MAE 0.308 dB`), cutting unseen-site RMSE by `30.0%` vs `Ensemble_v2`.
  - **Calibration-Tilt Stress (`+8 dB/dec` on unseen holdout):** Preserves **`0.9561` ROC AUC** and **`0.0929` `zero_rmse`** (vs `Ensemble_v2`: `0.8657` AUC / `0.2023` `zero_rmse`).
  - **`Test.csv` Inference (`1,500` rows, strictly zero fitting):** finite in `[0, 1]`, `share_mean = 0.2764`, `share_median = 0.2109`, `share_std = 0.2239`, exact `ID` match.

## Research Write-up, Notebooks, Standards & Reproducibility Track

- **Primary Research PDF (`deliverables/overleaf_manuscript/research_manuscript_latex.pdf`):** `10` pages (`3,227,606` bytes, `<10 MB`), `0` `Overfull \hbox`, `0` undefined refs/citations, `28/28` APA 7th citations, `23/23` labeled equations/tables/figures (`8` figures, `7` tables, `8` equations), SHA-256 `2f0280930b87afd56128ae435b74e72aeeeb32b60ba967b6a89001c25a7f13c4`.
- **Overleaf Upload ZIP (`deliverables/overleaf_manuscript/overleaf_research_manuscript.zip`):** `2,614,708` bytes (`11` clean entries), SHA-256 `41347a45f17f6f5bfc39c184b888c0f890fa321281a5460a44605e9fd3b04a46`.
- **Unified Code Archive (`deliverables/overleaf_manuscript/research_and_modeling_code_final.zip`):** `< 10 MB`, packages `41` verified scripts, all `3` notebooks (`01`, `02`, `03`), documentation, and TeX sources; clean-copy extraction passes `py_compile` and `--self-test-only` across `v2`, `v3`, `v4`, `standardized_evaluation_suite`, and `evaluate_sa_relative_vessel_activity` (canonical hash tracked in `deliverables/overleaf_manuscript/OVERLEAF_MANIFEST.json` and `reports/reproducibility_manifest.json`).
- **Legacy V6 Research Archive (`deliverables/research_submission/`):** Preserved (`5` pages, `research_code_v6.zip` SHA-256 `02431f367aa464a2cebf24e70bac5aeb22775b000f2e4224834e7838d9568eb1`).
- **Notebooks (`notebooks/`):** `01_south_african_acoustic_eda.ipynb`, `02_vessel_noise_modeling.ipynb`, and newly executed `03_domain_generalization_v3_v4_and_standards.ipynb` (`notebooks/` scratch files cleaned).
- **International Standards & Site-Level Activity Suite (`scripts/standardized_evaluation_suite.py`, `scripts/evaluate_sa_relative_vessel_activity.py`, `docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md`):** Evaluates ISO `18405:2017`, IEC `61260-1`, NPL GPG `133`, ADEON, JOMOPANS $L_{\text{excess}}$, Lin's CCC, NSE, PCEN, RPCA, and `17,487` held-out real South African windows across all `8` usable sites + `2` excluded `<15`-min sites (per official Zindi Staff ruling `daniil_zindi`, `24 Nov 2026`).
- **Environment & Storage:** `125` pinned non-AutoML packages in `requirements.txt`; master verifier `scripts/reproduce_pipeline.py --mode verify-only` (`7` self-tests) PASS; raw external audio archives removed after external personal backup verification (`20 GiB` disk free), keeping compact `data/processed/*.parquet` (`696 KB`) + `data/zindi/extracted/` (`92 MB`) for full deterministic reproduction.

## Completed Human Actions & Next Milestones

1. **Completed:** External data disclosure posted on Zindi ([Discussion #36419](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36419)), organizer map clarification answered by Zindi Staff (`daniil_zindi`, `24 Nov 2026`) and implemented (`docs/ORGANIZER_MAP_CLARIFICATION.md`), and Submissions #1 & #2 uploaded and scored.
2. **Next Upload:** Submit `submissions/candidate_v4_ensemble.csv` as **Submission #3** on Zindi (alongside `research_manuscript_latex.pdf` and `research_and_modeling_code_final.zip` before **13 Dec 2026, 11:30 UTC**).
3. **Modeling v5 Contingency (if Sub #3 LB still shows residual upward score bias):** Execute the documented **v5 blueprint** in `docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md` (isolate `score_added_vessel` strictly to `h(f)`-invariant `R_*`, `PCEN`, smooth sigmoid steps, `RPCA`, `ResNorm`/`Freq-MixStyle` without tilt channel `ch4`, and append v5 cleanly in `reproduce_pipeline.py`, Notebook `03`/`04`, manuscript, and docs).
