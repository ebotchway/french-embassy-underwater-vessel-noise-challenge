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
