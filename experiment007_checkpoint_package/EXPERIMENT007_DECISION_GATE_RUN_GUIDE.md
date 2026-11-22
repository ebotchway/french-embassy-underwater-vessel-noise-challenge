# Experiment 007 — read-only decision gate and research checkpoint

## Scope

This package adds no new model experiment. It audits **existing** Experiment 007 output without training, updating thresholds, or opening test data. It then appends documented findings to the current research notebook, learning log, and `CHECKPOINT.md` from your Mac checkout. Historical Phase 3 text already retained in `CHECKPOINT.md` is left intact. The old separate Phase 3 file is not restored.

Input: `reports/baselines/experiment007_source_disjoint_v1/` (the completed 007 report). The gate verifies saved SHA256 hashes, source group overlap, prediction correspondence, hybrid equivalence, genuine-background and observed-positive labels, calibration-selected threshold provenance, deletion summaries, and source counts. It includes the *already computed* bootstrap results; it does not run new bootstrap optimizations or alter the model.

## Install

After downloading the ZIP into `~/Downloads`, from your Mac's project root with `.venv` active:

```bash
mkdir -p experiment007_checkpoint_package research_checkpoints
unzip -n ~/Downloads/EXPERIMENT007_DECISION_AND_DOCUMENTATION.zip -d experiment007_checkpoint_package
cp experiment007_checkpoint_package/audit_experiment007_decision_gate_v1.py scripts/
cp experiment007_checkpoint_package/append_experiment007_research_checkpoint_v1.py scripts/
cp experiment007_checkpoint_package/EXPERIMENT007_DECISION_GATE_RUN_GUIDE.md research_checkpoints/
```

## Run the read-only audit

```bash
python -m py_compile scripts/audit_experiment007_decision_gate_v1.py scripts/append_experiment007_research_checkpoint_v1.py
python scripts/audit_experiment007_decision_gate_v1.py --self-test-only
python scripts/audit_experiment007_decision_gate_v1.py
cat reports/audits/experiment007_decision_gate_v1/RESEARCH_NOTE.md
```

Output: `reports/audits/experiment007_decision_gate_v1/`. The audit refuses to overwrite any existing output. If rerunning is necessary, use a new `--output` path rather than deleting or silently overwriting the earlier audit.

## Append research documentation

The updater reads `assessment_policy_metrics.csv` and **current local** versions of the notebook, log, and checkpoint. It refuses duplicate application of the unique research marker and creates date-stamped backups before any edits. It appends one markdown cell, leaving all original JSON cells, code content, execution counts, and stored outputs unchanged. It appends a new learning-log section, prepends the current summary to `CHECKPOINT.md`, and creates a separate manuscript addendum without overwriting the manuscript draft.

```bash
python scripts/append_experiment007_research_checkpoint_v1.py
python scripts/append_experiment007_research_checkpoint_v1.py --apply
```

The output addendum is `reports/research_manuscript_v1/EXPERIMENT007_RESULTS_ADDENDUM.md`; backup location is `reports/research_backups/experiment007_checkpoint/`.

Review before committing:

```bash
git status --short
 git diff --stat
 git diff --check
 git diff -- PROJECT_LEARNING_LOG.md CHECKPOINT.md
```

The notebook update can be inspected separately in VS Code. Stage only the intended files:

```bash
git add scripts/audit_experiment007_decision_gate_v1.py scripts/append_experiment007_research_checkpoint_v1.py research_checkpoints/EXPERIMENT007_DECISION_GATE_RUN_GUIDE.md PROJECT_LEARNING_LOG.md CHECKPOINT.md notebooks/01_south_african_acoustic_eda.ipynb
 git diff --cached --check
 git commit -m "Audit Experiment 007 decision gate and document research checkpoint"
```

Note: `reports/` may be excluded under your `.gitignore`. Keep the generated research evidence locally with manifest hashes and, if needed, archive it separately; do not force-add raw competition/external audio files to public Git.

## Research decision and constraints

The user-reported assessment had original-only RMSE 0.114236 versus varied/hybrid 0.094566. At threshold 0.5, original/hybrid no-addition false-positive rate was 16.67% (TPR 73.95%), versus varied detector 30.00% (TPR 87.68%). The varied detector's calibration-selected 0.55 threshold yielded assessment FPR 23.33% and TPR 85.29%. The 25% FPR rule applies to calibration only. The audit does not reselect thresholds or declare a superior external model; it preserves both the hybrid specificity and varied-detector sensitivity options. No new Zindi submission follows from the synthetic assessment alone.

Scientific research blockers remain independent: verifying the VTUAD dataset version and publishing allowed third-party-data disclosure, establishing authoritative hydrophone deployment coordinates before geospatial maps, independent real-vessel or AIS evidence before claiming traffic attribution, acoustic calibration status, and completing the final report/PDF under current organizer requirements.
