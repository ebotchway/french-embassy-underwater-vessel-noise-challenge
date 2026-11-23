# Project Checkpoint

Updated: 23 November 2026. Status: **HOLD FOR ZINDI SUBMISSIONS** pending Zindi external-data forum posting, organizer map clarification, and portal upload.

## Leaderboard Track (Candidate v2 Ready)

- **Submission 001 (Historical baseline)** (`reports/submissions/first_candidate.csv`, ID `pBUapzRu`, commit `aaf4748`): public `1 - RMSE` = `0.761536`, diagnostic `ROC AUC` = `0.517533`.
- **Candidate v2 (`submissions/candidate_v2_ensemble.csv`):** Deterministic 5-fold CPU ensemble (`CNN2D` 0.516 + `HistGBM` 0.143 + `CatBoost` 0.115 + `LightGBM` 0.114 + `XGBoost` 0.111) trained strictly inside source-disjoint `train` (`6,000` windows, starter-matched 2s base-10 deci-decade VTUAD `525` + HearMyShip `96`, `+37.5 dB` offset-invariant features, `Test.csv` strictly inference-only).
  - **Validation (`n=1,500`):** `1 - RMSE = 0.8747`, `RMSE = 0.1253`, `ROC AUC = 0.9725` (vs `V1_RF_MaxShift` `0.8270` / `0.9578`).
  - **Unseen-Station Holdout (`09_Fish_Hoek` + `10_Sea_Point`, `n=541`):** `1 - RMSE = 0.8833`, `RMSE = 0.1167`, `ROC AUC = 0.9365` (vs `V1_RF_MaxShift` `0.8390` / `0.8980`).
  - **Test.csv predictions (`1,500` rows):** finite `[0, 1]`, `share_mean = 0.2932`, `share_std = 0.2174`.

## Research Submission Track (V6 Verified)

- `deliverables/research_submission/research_code_v6.zip`: `79,563` bytes (`20` entries, `17` scripts), SHA-256 `02431f367aa464a2cebf24e70bac5aeb22775b000f2e4224834e7838d9568eb1`.
- `deliverables/research_submission/research_manuscript.pdf`: `5` pages (`671,917` bytes), SHA-256 `7d64b4d7c796b8a26802a9653870678e7fd5ae9c2565d7d057a5a28247b9c039`.
- Clean-copy verification (`--verify-only`): byte-identical Figure 4, Exp 007 decision-gate CSVs (`0.09457` vs `0.11424` RMSE), and 6-station descriptive v2 CSVs (`460` training runs, `61.5–90.6` uncalibrated dB).

## Open Human Actions Before Upload

1. Post `docs/EXTERNAL_DATA_DISCLOSURE.md` (VTUAD `v1.3.0` + HearMyShip) on the Zindi forum and log URL + timestamp.
2. Post `docs/ORGANIZER_MAP_CLARIFICATION.md` regarding Figure 4 total-sound site context vs vessel-pressure map.
3. Upload `submissions/candidate_v2_ensemble.csv` (after disclosure), `research_manuscript.pdf`, and `research_code_v6.zip` before **13 Dec 2026, 11:30 UTC**.
