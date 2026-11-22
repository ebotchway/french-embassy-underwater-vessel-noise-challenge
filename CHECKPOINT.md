# Current study status — after Experiment 007 (22 November 2026)

Experiment 007 completed a prespecified training/calibration/assessment split **before** acoustic pairing, using only Experiment 004's training source pool. Source overlap and output hashes passed independent verification. The assessment documents improved synthetic share RMSE for the contrast-varied regressor but competing detector specificity/sensitivity trade-offs; no real-vessel generalization is established. The Experiment 007 assessment is frozen: do not select or retune a threshold using its results. Phase 3 remains preserved in the existing history of this checkpoint.

# Research checkpoint — Experiments 001–006B (22 November 2026)

This file supersedes the earlier Phase 3 **status and next-actions list**, while preserving it verbatim in the historical appendix. All numeric model and audit findings below were supplied from terminal output on the researcher's Mac; the editing environment did not execute against the acoustic archive.

## Current decisions and safeguards

**Current state:** Experiments 001–006B have been performed, with the newest work confined to internal source-separated synthetic development. Submission 001 remains the only observed Zindi submission in this record; its public score is provisional. The separate South African acoustic research manuscript remains a **draft**, not a finished scientific paper. The contrast-varied model is **not approved for promotion or submission** because it generates more genuine-background false alarms at a 0.5 detection threshold.

**Latest checkpoint:** Experiment 006B read-only audit passed and its implementation was committed as `0ad8a9b`. Experiment 006 implementation fix was committed as `1e97005`; research protocol and run guide as `22650d4`. The reported audits and model runs were executed on the researcher's Mac and the numerical results below are transcribed from their terminal output, not rerun against the private acoustic archive in this document-editing environment.

**Next research stage:** Freeze Experiment 006/006B results; design **Experiment 006C** with background-run/hull-grouped disjoint calibration and assessment partitions, investigating specificity, predeclared thresholds, and a possible split regression/detection architecture. Do not choose a threshold on the same examples used to claim independent performance. Do not reopen Experiment 004 frozen evaluation, prior validation, sealed station holdouts, or hidden test data during exploratory selection.

**Unresolved scientific/compliance responsibilities:** Verify the exact locally used VTUAD version and publish a version-accurate public external-data declaration (draft is not proof of posting); obtain authoritative hydrophone/deployment coordinates before geospatial mapping (all ten sites currently have only candidate locations, and Saldanha Bay has two alternatives); verify recorder/acoustic calibration and independent vessel labels/AIS corroboration before attributing acoustic activity to vessels; complete referenced research figures, runnable reproduction and PDF; check the live competition deadlines/rules before final delivery.

---

## 28. Experiment 005 — real-background spectral sensitivity and source-aware uncertainty (22 November 2026)

The **training-only** simulation-realism audit processed **460 sampled South African background runs** (Richards Bay 120, Durban 1, Bird Island 22, St Croix 77, St Francis 120, Saldanha Bay 120). It produced **5,520 scenario rows**, covering original VTUAD relative band-power shapes, halfway shapes and background-shaped controls, each with fixed target shares **3% and 15%** and event durations **1 and 3 minutes**. For matched cases, the background, source selection, duration and requested share were controlled. Interventions are **counterfactual spectral-shape sensitivity checks**, not physically validated source sounds.

The follow-up report `reports/audits/simulation_realism_uncertainty_v1/` analysed **1,840 intervention triplets**, with **460 background runs, 109 vessel hulls and 228 clips overall**; per-scenario hull and clip counts differ. For the median *paired change in maximum minute-to-minute band-shape RMS dB statistic* (original VTUAD minus background-shaped control):

| Requested share | Duration | Median paired contrast | Run-cluster bootstrap 95% interval | Fraction original greater |
|---|---:|---:|---|---:|
| 0.03 | 1 min | +3.4952 | [3.1254, 4.0957] | 0.7913 |
| 0.03 | 3 min | +1.4173 | [0.9385, 1.7663] | 0.6957 |
| 0.15 | 1 min | +6.5732 | [5.8885, 7.2776] | 0.9391 |
| 0.15 | 3 min | +4.9443 | [4.3834, 5.7751] | 0.9043 |

The **3%/3-minute halfway-minus-background** comparison is an important exception: median **0**, run-bootstrap interval **[0, 0.1209]**, and only **49.13%** of observed pairs had a positive difference. Across the original-minus-background and original-minus-halfway analyses, deleting one tested background run, station, clip or hull did not reverse the overall *median* contrast in the reported scenarios. These deletion results do **not** establish independent deployment- or hull-level uncertainty.

**Station heterogeneity:** For the 3%/one-minute original-minus-background comparison, St Francis has median **0** and fraction positive **0.4083**; its 3%/three-minute median is also **0** and fraction positive **0.3167**. Durban's exceptionally large contrasts reflect **one sampled background run** and **one hull per scenario**, hence they cannot characterize Durban generally. The training-only analysis establishes that spectral treatment changes the simulated temporal statistic, **not** that a certain treatment is authentic, or that it improves real-vessel detection or the hidden-test score.

**Files:** `reports/audits/simulation_realism_v1/`, `reports/audits/simulation_realism_uncertainty_v1/`; scripts `audit_simulation_realism_v1.py`, `analyze_experiment005_source_uncertainty_v1.py`.


## 29. Experiment 006 — internal source-separated contrast-augmentation comparison (22 November 2026)

Experiment 006 ran after fixing the script's `pd.Data` → `pd.DataFrame` typo (commit `1e97005`). It built an **internal development allocation solely from `exp004_train`**, with **1,200 training windows** and **120 internal development pairs**. The prediction summary has **840 rows per model** (zero-addition controls and matched spectral-intervention examples). Its terminal reported that Experiment 004 frozen evaluation, the earlier validation, sealed station holdouts and `Test.csv` were not used. The observed internal-development source counts are **21 distinct hulls and 41 clips**, across five development stations. These are internal synthetic metrics, not external/field validation.

| Metric | Original-only RF | Contrast-varied RF | Interpretation |
|---|---:|---:|---|
| Added-share RMSE | 0.133224 | **0.114353** | Reduction 0.018871 (~14.2%) |
| Added-share MAE | 0.074795 | **0.067487** | Reduction 0.007307 |
| Dedicated detector ROC AUC | 0.847977 | **0.877137** | Gain 0.029160 |
| Detector average precision | 0.954545 | **0.966052** | Gain 0.011507 |
| False-positive rate, *all observed-zero* rows, threshold 0.5 | **0.256410** | 0.365385 | Worse by 0.108974 |
| False-positive rate, *genuine no-addition* controls, threshold 0.5 | **0.233333** | 0.375000 | Worse by 0.141667 |

The augmentation improved RMSE for all three spectral-treatment-plus-zero comparisons (original VTUAD **0.108777 → 0.101578**, halfway **0.122376 → 0.102882**, background-shaped **0.134490 → 0.116326**), but ROC AUC on the original VTUAD treatment dropped **0.944395 → 0.925488**. Station RMSE improved in **four of five** internal-development stations; St Francis worsened from **0.087372** to **0.090038**. Richards Bay remained the hardest by RMSE (**0.244579 → 0.209125**) and its genuine background false-positive rate at threshold 0.5 increased **0.291667 → 0.625000**.

The study used prespecified missing-minute masks; some mask categories contain only **10–15 distinct development pairs**, so reported subgroup gains require caution. **Do not use the frozen Experiment 004 evaluation to select between variants repeatedly.**

**Files:** `reports/baselines/experiment006_internal_contrast_v1/` including `development_summary.csv`, `matched_development_predictions.csv`, allocations and provenance; script `run_experiment006_contrast_robustness_v1.py`; research guides under `research_checkpoints/`.


## 30. Experiment 006B — paired uncertainty, specificity and protocol audit (22 November 2026)

The **read-only** audit of Experiment 006's saved predictions passed, with **120 matched development pairs, 21 hulls and 41 clips**, and **1,000 pair-bootstrap replicates**. It did not retrain or read the preserved frozen evaluation, original validation, sealed holdouts or hidden test labels. The following 95% intervals reflect *matched-pair resampling conditional on this internal split*, not independent real-world generalization.

| Metric (contrast-varied advantage except FPR increases) | Point estimate | 95% pair-bootstrap interval |
|---|---:|---|
| RMSE reduction | +0.018871 | [+0.006436, +0.031532] |
| MAE reduction | +0.007307 | [+0.002329, +0.012656] |
| AUC gain | +0.029160 | [+0.016376, +0.041305] |
| Average-precision gain | +0.011507 | [+0.006371, +0.016633] |
| **Genuine no-addition FPR increase at 0.5** | **+0.141667** | **[+0.075000, +0.208333]** |
| All observed-zero FPR increase at 0.5 | +0.108974 | [+0.055556, +0.166667] |

There were **36 requested-positive but observed-zero** rows: under whole-minute masks, a source event can become unobservable and the target/detection label for the **observed cells** is zero. These are distinct from truly no-addition controls; the increased false-positive rate also appears when genuine controls are analysed alone.

**Leave-one-group-out audit:** 144 deletion cases comprised **5 stations, 77 unique background-run IDs, 41 clips and 21 hulls**. All reported RMSE reductions and AUC gains remained positive, and all genuine no-addition FPR increases remained positive after a single-group deletion; the minima were **RMSE reduction 0.011466**, **AUC gain 0.024078** and **FPR increase 0.093750** (across the deletions). This is descriptive source sensitivity, not fully independent grouped inference. In particular, 120 development pairs do not imply 120 unique background run IDs.

**Threshold trade-off (descriptive, not calibrated):** At 0.5 the contrast-varied model has true-positive rate **0.900585** and genuine-background FPR **0.375**, whereas original-only has TPR **0.792398** and genuine-background FPR **0.233333**. At precomputed threshold 0.8 the contrast-varied model has TPR **0.735380**, genuine-background FPR **0.125**; at original-only threshold 0.7, TPR **0.647661**, genuine-background FPR **0.125**. This is **not** evidence of a selected, independently validated operating point; choosing thresholds based on these same observations would be optimistic.

**Decision:** Preserve *both* models and the 006/006B records. The contrast-varied **fraction regressor** remains a candidate; its 0.5-threshold **detector** is not approved for promotion. Design Experiment 006C with disjoint group-based calibration and assessment, predeclared specificity/sensitivity costs, and an optional hybrid (contrast-varied regressor plus original-only detector). Any independent assessment must remain separate from calibration choices. **Do not make a new Zindi submission based solely on these synthetic results.**

**Files:** `reports/audits/experiment006b_readonly_v1/` (`overall_uncertainty.csv`, `threshold_sensitivity.csv`, `leave_one_source_out.csv`, `segment_diagnostics.csv`, `RESEARCH_NOTE.md`); script `audit_experiment006b_readonly_v1.py`, committed as `0ad8a9b`.


## 31. Geospatial provenance, manuscript, and next research gate

`data/zindi/extracted/stations.csv` contains **10 working site-level coordinate candidates**, not verified recorder coordinates. **Nine** entries have a single working site-level candidate requiring deployment assignment; **Saldanha Bay** has **two unresolved candidate coordinate pairs** (`−33.04720, 17.95460` and `−33.08636, 17.89876`). The original dictionary states that candidate site coordinates do not certify hydrophone deployment position. The `map_verified_station_acoustic_excursions_v1.py` self-test passed, but the normal mapping run intentionally stopped with `ValueError: Coordinates or authoritative provenance missing`. This is a **correct scientific safeguard**. Do not bypass it, pretend candidate coordinates are validated hydrophone locations, or describe station-level acoustic excursions as vessel traffic.

The research-manuscript draft and figure collection are at `reports/research_manuscript_v1/`, with the coordinate provenance template at `STATION_COORDINATES_PROVENANCE_TEMPLATE.csv`. The draft still needs verified primary citations, hydrophone/deployment metadata, any available calibration evidence, time-coverage accounting, independent AIS/field vessel evidence for attribution, a careful relative acoustic map when authorized, disclosure of external VTUAD version/license/access and a final reproducible PDF. The submission checkpoint is dated **22 November 2026**; verify current official deadline and rules before final submission (the previously recorded research PDF/code deadline was 13 December 2026 at 11:30 UTC).

**Next technical work:** (i) freeze 006B outputs; (ii) independently audit/check Experiment 006 source and allocation provenance; (iii) design 006C with new, group-separated calibration/assessment, avoiding selection on existing 120 pairs; (iv) prepare research manuscript addendum from the recorded 005–006B results; (v) resolve external-data and monitoring-coordinate provenance. No hidden labels were accessed for the recorded analyses.



---

# Historical appendix — Phase 3 checkpoint, preserved from input

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

## Experiment 007 — Source-disjoint training, calibration and assessment (EXPERIMENT007_SOURCE_DISJOINT_CHECKPOINT_2026_11_22)

**Execution evidence:** The researcher's local execution of `scripts/run_experiment007_source_disjoint_v1.py` reported `EXPERIMENT 007: PASS` and `scripts/verify_experiment007_artifacts_v1.py` reported `OUTPUT INTEGRITY: PASS`, with zero known background-run, hull or clip overlap across the three allocations. Source allocations were performed *before* pairing. Implementation commit: `de051b3`. This is a **new internal synthetic assessment** from `exp004_train` only; its source pools were studied historically, so it is neither a pristine external holdout nor real-vessel validation.

**Assessment:** 120 paired background/source conditions, 840 cases per model, 120 genuine no-addition controls, and 714 observed positives (6 requested-positive cases became observed-zero after masking). Both model families were trained on 1,200 simulated windows; threshold selection used a separate calibration partition. Five stations were assessed; Durban was training-only due to a single eligible background run.

| Model and threshold | Fraction RMSE | Detector AUC | Observed-positive TPR | Genuine no-addition FPR |
|---|---:|---:|---:|---:|
| Original-only, fixed 0.50 | 0.114236 | 0.848328 | 0.7395 | 0.1667 |
| Contrast-varied, fixed 0.50 | 0.094566 | 0.886421 | 0.8768 | 0.3000 |
| Contrast-varied, calibration-selected 0.55 | 0.094566 | 0.886421 | 0.8529 | 0.2333 |
| Hybrid varied regressor + original detector, fixed 0.50 | 0.094566 | 0.848328 | 0.7395 | 0.1667 |

The contrast-varied regressor improved the prespecified internal assessment RMSE. The absolute RMSE reduction against original-only was **0.019669** (~**17.2%**). The contrasted detector's ROC AUC was **0.886421**, versus original-only **0.848328**, but the **fixed 0.50 FPR** increased from **0.1667** to **0.3000**. The *calibration-selected* threshold 0.55 yielded assessment FPR **0.2333** and sensitivity **0.8529**. The 25% FPR limit was imposed only during calibration; assessment need not meet it.

**Hybrid interpretation:** The hybrid is not a third trained model: its share predictions are those of the contrast-varied regressor and its detection scores those of the original-only classifier. Fixed 0.5 hybrid FPR **0.1667** and TPR **0.7395** equal the original detector by construction. The two best candidates reflect a real specificity-versus-sensitivity trade-off, not a single universal winner. ROC AUC is a ranking metric; the synthetic positive prevalence makes AP unsuitable as an estimate of natural event precision.

**Integrity and scientific limits:** The independent output verifier checked the saved results and hashes and found no known source-group overlap. Run-group bootstraps remain conditional on the split and do not make hulls or deployments independent. Relative-energy mixing has no cross-corpus physical sound-pressure calibration; VTUAD candidate vessel clips include ambient sound. Neither model has independent real South African vessel-event ground truth. Station-level position candidates do not establish hydrophone coordinates. The Experiment 004 frozen evaluation, original validation, sealed station holdouts, and hidden Zindi Test labels were not accessed in this experiment.

**Decision:** Freeze Experiment 007 predictions and calibration thresholds. Do not retune using its assessment outputs, choose a new threshold after inspecting results, or submit a second Zindi candidate solely on synthetic evidence. Preserve both the hybrid and calibrated varied-detector configurations as *research policies*, and finish the scientific manuscript, source/version declaration and location/measurement provenance checks. Additional comparison should be independently predeclared before accessing reserved data.

**Files:** `reports/baselines/experiment007_source_disjoint_v1/`; `reports/audits/experiment007_decision_gate_v1/` (if executed); scripts `run_experiment007_source_disjoint_v1.py`, `verify_experiment007_artifacts_v1.py`, `audit_experiment007_decision_gate_v1.py`. Earlier Phase 3 content is already incorporated in `CHECKPOINT.md`; there is no requirement to restore a separate Phase 3 file.

**Read-only decision-gate audit:** Output manifest present; all policy interpretations remain descriptive and no further model selection was performed. Verified gate source SHA256: `f40a18c0cc361ef3f8d4fb20797044dd1c12e421c4e2e4f0649ec3d0211309b4`.

