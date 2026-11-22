# Phase 3 research checkpoint — 22 November 2026

## Purpose
Record confirmed organizer requirements, the first leaderboard benchmark, and a reproducible next experiment without rewriting the main learning log and notebook prematurely.

## Submission 001 (observed public result)
- Zindi submission ID: `pBUapzRu`; comment: `Baseline Submission 001`.
- Public score: `0.761535762`; share RMSE displayed: `0.238464237`.
- Diagnostic `Score Added Vessel` ROC AUC displayed: `0.517533333`.
- Rank at the captured screenshot: 565. Rank is time-sensitive and should not be treated as permanent.
- Local candidate: `reports/submissions/first_candidate.csv`; originating Git commit `aaf4748`.
- The candidate's evidence score duplicated its estimated share and was not a separately trained detector.

## Prior synthetic baseline
- Experiment 002: 1,200 training and 300 validation windows; fixed 96-feature Random Forest RMSE `0.112061` on synthetic validation.
- Original Phase 3 audit: same Experiment 002 predictions produced synthetic detection ROC AUC `0.975621` and AP `0.992890`.
- Submitted model was *not* the exact Experiment 002 prediction pipeline. Its per-window-normalized synthetic share RMSE was `0.166477`, with different preprocessing. Comparing the Experiment 002 synthetic AUC directly with Submission 001 public AUC is not a like-for-like model comparison.

## What the official rules and public discussions support
- Leaderboard ranking is based on `1 − RMSE` for `share_added_vessel` only.
- `score_added_vessel` is a diagnostic presence-evidence score evaluated by ROC AUC, not a calibrated probability of natural vessel presence.
- Test data are for inference; no fitting, fine-tuning, pseudo-labeling, or test-derived training examples.
- Permitted external acoustic datasets must be declared with their version, license, source and access details.
- VTUAD's publicly accessible v1.3.2 declaration exists; our previously audited local source was described as **v1.3.0 candidate**, so these must not be conflated.
- Public discussion posts are not equivalent to an organizer ruling when they merely ask questions.

## Current technical audit
- 1,500 test examples; 15 × 25 input cells; 18,350 missing measurements in complete minute profiles; 725 examples affected.
- Original generator and Submission 001 model use window-maximum-relative log-level transformations. Do not interpret the synthetic-to-public score gap as proof this normalization is the cause.
- The current data-to-model alignment audit identified large errors in short events, weak events at St Francis, and some repeated source groups; these are hypotheses for controlled testing, not causal conclusions.

## Immediate next steps
1. Review and post an accurate VTUAD external-data declaration, after confirming the locally used version. **Draft only; not posted automatically.**
2. Run `scripts/audit_submission001_detection.py`, reproducing the exact Submission 001 model and synthetic validation preprocessing. Evaluate ROC AUC and average precision, plus zero/positive sample counts and missing-minute stress variants. This reads **synthetic training/validation only**.
3. Inspect public v1.3.2 Zenodo release metadata and local v1.3.0-candidate source files; compare actual manifests and checksums *before* adopting any release changes.
4. Design new source-separated experiments only after these tests, without relying on leaderboard probe optimization.
5. Update `PROJECT_LEARNING_LOG.md` and `notebooks/01_south_african_acoustic_eda.ipynb` after the reproducibility/like-for-like detection checkpoint.

## Primary source links (checked 22 Nov 2026)
- https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
- https://zindi.world/discussions/35800
- https://zenodo.org/records/23037895
- https://data.oceannetworks.ca/

All claims above distinguish user-observed results from organizer documentation and research hypotheses. No hidden test labels were inspected, and this checkpoint is not evidence of real-world calibration.
