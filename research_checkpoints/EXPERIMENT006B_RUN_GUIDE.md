# Experiment 006B — Read-only protocol, uncertainty, and false-positive audit

## Goal
Audit the existing Experiment 006 *internal development* predictions. No retraining or hyperparameter calibration; no reading of Experiment 004 frozen evaluation, original validation, sealed holdouts, or Zindi Test.csv.

## Install (project root on Mac)

Save `audit_experiment006b_readonly_v1.py` from this conversation into `scripts/`.

```bash
python -m py_compile scripts/audit_experiment006b_readonly_v1.py
python scripts/audit_experiment006b_readonly_v1.py --self-test-only
python scripts/audit_experiment006b_readonly_v1.py
```

The script uses `numpy`, `pandas` and `scikit-learn`, already in your Experiment 006 environment. No tabulate dependency.

Expected output directory:

`reports/audits/experiment006b_readonly_v1/`

If the output already exists, it refuses to overwrite. To repeat, use `--output reports/audits/experiment006b_readonly_v2`. Do not delete historical results.

## Outputs

- `RESEARCH_NOTE.md`: overall paired results and explicit limitations.
- `overall_uncertainty.csv`: paired-background-condition bootstrap interval estimates, conditioned on this *single* internal development allocation and its source reuse.
- `paired_bootstrap_draws.csv`: raw bootstrap estimates for reproducibility.
- `segment_diagnostics.csv`: model-level summaries for station, mask, requested share, intervention, zero controls, and overall.
- `threshold_sensitivity.csv`: descriptive FPR and TPR from fixed detection thresholds 0.1 through 0.9, by station/mask/overall; **not optimized thresholds**.
- `leave_one_source_out.csv`: one-at-a-time omission of station, background run, vessel hull and vessel clip; sensitivity, **not an independent CI**.
- `paired_differences.csv`: same-example squared-error and absolute-error improvements.
- `audit_manifest.json`: input file digests, sample size, method and result metadata.

## Interpretation safeguards

1. Experiment 006 has 120 background/source pairs and 7 conditions per pair, evaluated with both models. The rows are correlated.
2. `requested_share > 0` is not always an *observed* positive after missing-minute masking. Detector labels in Experiment 006 use `observed_share > 0`. The FPR among all observed-zero examples and the FPR on *no-addition controls only* are different diagnostics.
3. The paired bootstrap resamples background/source pairs, not independent deployments or hulls. It provides a conditional sensitivity interval, **not an external generalization confidence interval**.
4. The threshold table is exploratory. Selecting a decision threshold after inspecting it and reporting that same development performance is optimistically biased; use a future untouched calibration/evaluation design.
5. Overall RMSE and ROC AUC refer to development mixtures, not evidence of real-vessel detection nor public/private Zindi performance.
6. The audit validates allocation membership and observed train/dev disjointness, but it cannot verify all original signal generation without reading raw source data. The Experiment 006 output manifest records its own source-frozen hash checks.
7. No promoted model, additional Zindi submission, or frozen evaluation use is justified by this read-only audit alone.

## Review after execution

```bash
cat reports/audits/experiment006b_readonly_v1/RESEARCH_NOTE.md
python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path('reports/audits/experiment006b_readonly_v1')
for name in ('overall_uncertainty.csv','segment_diagnostics.csv','threshold_sensitivity.csv'):
    df=pd.read_csv(p/name)
    print('\n',name)
    if name=='segment_diagnostics.csv':df=df[df.group.isin(['overall','zero','station'])]
    if name=='threshold_sensitivity.csv':df=df[(df.scope=='overall') & df.threshold.isin([.3,.5,.7])]
    print(df.to_string(index=False))
PY
```

## Git

```bash
git add scripts/audit_experiment006b_readonly_v1.py research_checkpoints/EXPERIMENT006B_RUN_GUIDE.md
git commit -m "Add Experiment 006B read-only paired uncertainty and protocol audit"
```

Do not add any accidental `.bak` or raw acoustic source files to the commit.
