# Experiment 007 — Prespecified source-disjoint calibration and assessment

**Protocol prepared:** 22 November 2026. **Status:** implementation tested on constructed data; full South African/VTUAD run has not been performed in this package.

## Motivation and previous evidence

Experiment 006's contrast-varied model had internal-development RMSE 0.114353 versus 0.133224 for the original-only model, and ROC AUC 0.877137 versus 0.847977. In Experiment 006B, genuine no-addition false-positive rate at 0.5 increased from 0.2333 to 0.3750. The 95% matched-pair bootstrap interval for RMSE reduction was [0.006436, 0.031532], and the false-positive increase interval was [0.075, 0.208333]. These are *historical development* results, not new Experiment 007 predictions.

Experiment 006C found six connected source components among 120 pairs; the largest component contained 112 pairs. Thus independent calibration and assessment cannot be manufactured by re-splitting the existing paired predictions.

## Primary objective and models

A prespecified, group-separated study comparing:

- **Original-only:** original VTUAD source-shaped training for regression and detection.
- **Contrast-varied:** equally sized training with original/halfway/background-shaped source spectra for regression and detection.
- **Hybrid:** contrast-varied regression and original-only detection. This is not a third separately fitted model.

Primary regression outcome: observed-cell added-relative-energy fraction RMSE on the single assessment partition. Primary detection reporting: observed-positive sensitivity and genuine background-only false-positive rate at a threshold chosen using calibration only (plus a fixed 0.5 comparator); ROC AUC and AP are additional descriptive metrics.

## Data and allocation

Only the Experiment 004 **exp004_train** pools are accessible. The three frozen input manifests must match pinned SHA-256 values before any acoustic data are read. Exclude Experiment 004 evaluation, original validation, sealed station holdouts, and the Zindi test. The 007 assessment pool is disjoint from 007 training and calibration, but its underlying source pool has been explored in earlier research and is not a pristine external validation set.

**Before creating mixtures**, assign every background run to a partition per station and every vessel hull to a partition per class. Within strata of at least three groups, approximately 15% of groups go to calibration and 15% to assessment, with all remaining groups in training. One- or two-group strata are train-only. All clips of a hull remain together. Never break groups for station/class balance. Verify pairwise no-overlap of run, clip and hull identifiers.

Defaults: 1,200 synthetic training windows per fitted model, 120 calibration background/source pairs, 120 assessment pairs; five masks in the existing Experiment 006 generator; 1-minute or 3-minute events; zero, 3% and 15% addition conditions with three source shapes; fixed 200 trees and min_samples_leaf=2 for the Random Forests, fixed 96-feature transform and random seeds. Seven scenarios per calibration/assessment pair (one zero and six positive requests). Prescribed station-balanced sampling is *not* station-out-of-domain generalization.

## Operating point and calibration

On **calibration only**, independently choose the threshold for the original detector and varied detector from the fixed grid {0.05, 0.10, …, 1.00}. Choose the feasible threshold with greatest observed-positive sensitivity subject to genuine no-addition FPR <= 0.25; break ties in favor of the **higher** threshold. If no candidate meets the calibration cap, report infeasible; do not silently claim a calibrated policy. The hybrid reuses the original detector. Always assess the prespecified fixed 0.5 threshold too. Assessment FPR is **not required to achieve the calibration-set cap** and may be higher.

Optional Platt mappings are diagnostic only: fit on calibration score/label rows, then compare Brier and log-loss on assessment. They do not establish calibrated probabilities of natural vessel presence and are not automatically used as submission evidence.

## Assessment and uncertainty

Compute RMSE/MAE/bias, ROC AUC/AP, genuine-control FPR, observed-zero FPR, observed-positive sensitivity, and masked-requested-addition counts. Present breakdowns by station, missingness and source intervention. Prespecified comparisons use matched assessment conditions; the source-deletion audit removes a single station, background run, clip or hull. Run-clustered resampling reports descriptive intervals conditional on this partition. It does **not** account fully for shared hull/deployment dependencies; do not claim real-world confidence intervals.

**Decision principle:** Do not change the models, hyperparameters, thresholds, training distribution or feature transformations in response to the assessment results and then continue calling this a single independent test. A negative result is valid. Preserve assessment results, including false-positive failures, for the scientific report.

## Evidence and compliance limitations

The simulator works in relative-energy units with no demonstrated cross-archive pressure calibration. Vessel-associated VTUAD clips contain ambient background and are not isolated vessel-only signals. The spectral adjustments are sensitivity controls, not validated physical vessel templates. Real South African monitoring windows have no independent ground-truth vessel additions in this analysis. All inference about Zindi performance is provisional; no second leaderboard submission is authorized by this protocol.

External VTUAD dataset version, licence and public discussion declaration remain to be verified/published. Working site-level coordinates do not establish hydrophone deployment coordinates; the geographic map must retain its provenance safeguard. The research manuscript must include data periods, station coverage, recorder limitations, relative acoustic analyses, uncertainty, citations, and a clean reproducible code path before the research submission deadline.

## Deliverables and traceability

Script: `scripts/run_experiment007_source_disjoint_v1.py` plus `scripts/verify_experiment007_artifacts_v1.py`.

Results: `reports/baselines/experiment007_source_disjoint_v1/`, with source allocations, provenance, calibration/assessment predictions, chosen thresholds, policy/segment metrics, run-bootstrap uncertainty, source deletion, Platt diagnostics, research note, and SHA-256 manifest.

Results are generated locally only. **No full-results claims should be made until the user executes the script on the local acoustic data.**
