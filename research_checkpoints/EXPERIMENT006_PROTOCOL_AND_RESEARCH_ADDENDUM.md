# Acoustic research checkpoint — Experiment 005 conclusions and Experiment 006 protocol

**Recorded date:** 22 November 2026  
**Project:** French Embassy Underwater Vessel Noise Challenge  
**Status:** Experiment 005 analysis complete; Experiment 006 code implemented but **actual experimental results pending local execution**.

## Verified findings through Experiment 005

Experiment 005 used **460 training-only South African background recording runs** and source material from the permitted `exp004_train` VTUAD pool. Its temporal-contrast audit generated **5,520 controlled intervention rows**, arranged into **1,840 three-way intervention triplets** (four energy/duration conditions on each run). The later source-aware audit reports 109 distinct hulls and 228 clips overall; counts vary by scenario and should not be represented as independent repetitions.

The source treatments were the original VTUAD spectrum, a halfway spectral-shape blend, and a background-shaped control. They preserve the target added-energy share when constructed, but the latter two are **hypothetical sensitivity controls**, not validated vessel acoustics.

The original-minus-background-shaped *paired maximum temporal-step RMS statistic* had the following observed medians and background-run bootstrap intervals:

| Share | Event duration | Median paired difference | Run-bootstrap interval |
|---|---|---:|---:|
| 3% | 1 minute | 3.4952 | 3.1254–4.0957 |
| 3% | 3 minutes | 1.4173 | 0.9385–1.7663 |
| 15% | 1 minute | 6.5732 | 5.8885–7.2776 |
| 15% | 3 minutes | 4.9443 | 4.3834–5.7751 |

All original-versus-background-shaped comparisons remained positive when deleting any one tested station, recording run, hull or clip, but that sensitivity analysis is descriptive. **St Francis** had zero median original-minus-background-shaped contrast for both weak (3%) conditions, whereas other stations generally had positive values. **Durban** has just one sampled training background run; its unusually large changes must not be generalized. The halfway-versus-background control comparison at 3%/3 minutes had a zero median and an interval including zero.

These observations show that **simulated temporal detectability depends on source spectral treatment and station**. They do not identify the cause of the public leaderboard gap, prove the physical authenticity of either spectral treatment, or calibrate underwater sound pressure.

## Experiment 006 — preregistered internal development protocol

**Question:** Does mixing three source spectral treatments during *training* improve robustness to controlled spectral contrast compared with training exclusively on original VTUAD sources, at equal sample counts and with the same model family?

**Source control:** Use only frozen `exp004_train` manifests whose SHA256 hashes match the Experiment 004 record. Within this already-designated source pool, assign 20% of distinct background recording runs per station and 20% of distinct vessel hulls per class to a new **internal development** partition, using deterministic group hashes. Stations or classes with a single run/hull remain training-only. Do **not** read `exp004_eval`, the existing Experiment 002 validation, Fish Hoek / Sea Point sealed holdouts, or Zindi `Test.csv`.

**Model arms:** Both train the `combined_96` feature extractor with fixed `RandomForestRegressor` and `RandomForestClassifier` (200 trees, minimum leaf size 2, seed recorded). Their training backgrounds, randomly sampled fractions, insertion events, observation masks, and number of examples are identical. `original_only` uses gamma=0 for every positive source insertion. `contrast_varied` prespecifies gamma uniformly among 0, 0.5 and 1. The source-shape operation is the Experiment 005 one and is deliberately *not* claimed to be physical vessel propagation.

**Development conditions:** For each internal-development background/source pair, produce one identical zero-addition control and, for each share 3% and 15%, original/halfway/background-shaped inserted examples. Keep event duration at 1 or 3 minutes and whole-minute missingness as prespecified in the script. Recompute each target over observed cells only; interpolate masked log levels for features only. Do not tune the model settings or choose a new split seed after viewing results.

**Outcomes:** Fraction RMSE/MAE/bias, detection ROC AUC and average precision, and negative-control false-positive rate at threshold 0.5. Report results by spectral intervention (including common zeros for ROC AUC), station, missingness and target-share regime. Preserve per-example source provenance. Repeated rows within matched background/source pairs are *not independent examples*. No p-values or independent real-world performance intervals are implied.

**Decision criteria set before running:** An improvement in pooled RMSE or AUC is **development evidence only**; inspect simultaneously whether the model deteriorates in any source treatment, weak-addition scenario, station or mask. The model is not a candidate for submission merely because it improves on internal development conditions. Source and instrument representativeness, independent measurements, the official data declaration, and a fully reproduced inference pipeline remain separate requirements.

## Scientific report and mapping boundaries

The separate research paper may report measured **relative acoustic station variability**, run-level bootstrap uncertainty, observed spectral-shape differences, and **explicitly simulated** insertion interventions. It must not interpret those as independently verified vessel-presence measurements or calibrated underwater sound-pressure maps. Hydrophone deployment coordinates remain unverified: the local `stations.csv` lists nine working site-level candidates, plus two unresolved Saldanha Bay alternatives. Do not publish the coordinate plot as a hydrophone-location map until authoritative deployment sources are identified and cited.

The exact VTUAD release/version and licence still require verification against the downloaded data card; the competition's public external-data declaration remains an outstanding action until a posting URL or acknowledgment is recorded. The first Zindi public score is provisional, since final evaluation uses a different withheld subset. No later competition score is claimed in this checkpoint.

## Reproducibility references

- Frozen allocations: `research_archives/experiment004_frozen_v1.tar.gz` (SHA256 `fbec6b3685eb73d5592cdd815be2a5cc4d861a5322b154083355fe755f8230c6`).
- Experiment 005: `reports/audits/simulation_realism_v1/paired_temporal_changes.csv`.
- Source uncertainty: `reports/audits/simulation_realism_uncertainty_v1/comparison_summary.csv` and `deletion_summary.csv`.
- Experiment 006 code: `scripts/run_experiment006_contrast_robustness_v1.py`.
- Experiment 006 expected result directory: `reports/baselines/experiment006_internal_contrast_v1/` (**not populated until run**).

**Required next editorial action:** After verifying Experiment 006 results, copy the numeric outcomes and manifest checksums into the main notebook, learning log, and manuscript. Preserve the complete earlier document history and stored notebook outputs.
