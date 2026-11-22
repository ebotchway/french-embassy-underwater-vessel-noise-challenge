# Experiment 007 — Local execution guide

Run from the project root with `.venv` activated. This stage uses existing scripts `run_experiment006_contrast_robustness_v1.py`, `build_experiment004_frozen_mixtures.py`, `build_synthetic_mixtures_v1.py`, `build_synthetic_mixtures_v2.py`, `train_synthetic_baselines_v2.py`, and `audit_simulation_realism_v1.py`. Requires numpy, pandas, scikit-learn, pyarrow and the dependencies already installed for Experiment 006.

## Installation

```bash
mkdir -p experiment007_package
unzip -n ~/Downloads/EXPERIMENT007_IMPLEMENTATION.zip -d experiment007_package
cp experiment007_package/run_experiment007_source_disjoint_v1.py scripts/
cp experiment007_package/verify_experiment007_artifacts_v1.py scripts/
cp experiment007_package/EXPERIMENT007_PREDECLARED_PROTOCOL.md research_checkpoints/
cp experiment007_package/EXPERIMENT007_RUN_GUIDE.md research_checkpoints/
```

If your browser saves the ZIP under a different name or location, use that actual path. Do not use `cp` from Downloads unless the file exists there.

## Self-test and main run

```bash
python -m py_compile scripts/run_experiment007_source_disjoint_v1.py scripts/verify_experiment007_artifacts_v1.py
python scripts/run_experiment007_source_disjoint_v1.py --self-test-only
python scripts/run_experiment007_source_disjoint_v1.py --train 1200 --cal-pairs 120 --assess-pairs 120
```

The first command checks syntax. The self-test runs an in-memory full pipeline on constructed acoustic data. The final command uses **only SHA256-pinned exp004_train** source records from the local archive and writes a new report directory. No earlier report or frozen split is overwritten. Do not repeat the full run against the same output name. If an existing result is already complete, inspect/verify it; if a fresh independent reproducibility run is needed, set `--output` to a new unused path and record the changed purpose explicitly.

## Verify and inspect

```bash
python scripts/verify_experiment007_artifacts_v1.py
cat reports/baselines/experiment007_source_disjoint_v1/RESEARCH_NOTE.md
python - <<'PY'
import pandas as pd
from pathlib import Path
r=Path('reports/baselines/experiment007_source_disjoint_v1')
for name in ['assessment_policy_metrics.csv','assessment_run_bootstrap.csv',
             'assessment_source_deletion.csv','platt_calibration_diagnostic.csv']:
    d=pd.read_csv(r/name)
    print('\n',name,'rows:',len(d),'\n',d.head(20).to_string(index=False))
PY
```

Interpret calibration threshold FPR targets as **calibration-only constraints**, never guaranteed assessment performance. Report any source-stratum/segment with too few controls. Average precision will be optimistic relative to natural prevalence because the synthetic dataset deliberately contains many positives.

## Commit only source code/protocol after local success

```bash
git add scripts/run_experiment007_source_disjoint_v1.py \
  scripts/verify_experiment007_artifacts_v1.py \
  research_checkpoints/EXPERIMENT007_PREDECLARED_PROTOCOL.md \
  research_checkpoints/EXPERIMENT007_RUN_GUIDE.md
git diff --cached --check
git commit -m "Implement prespecified source-disjoint Experiment 007"
```

Do not commit the large/generated `reports/` results if the project intentionally ignores them. Archive and checksum research outputs separately. Do not commit test data, VTUAD audio, location secrets, or personal local paths.

## Next scientific step after results

Interpret assessment results ONCE, preserve all outcomes and update `PROJECT_LEARNING_LOG.md`, `CHECKPOINT.md`, the notebook, and manuscript with actual local results and clearly stated shortcomings. Do not overwrite those files with outdated copies from another environment. Keep the frozen Experiment 004 assessment unused during internal selection. Coordinate verification, independent vessel attribution and external-data declaration are separate outstanding tasks.
