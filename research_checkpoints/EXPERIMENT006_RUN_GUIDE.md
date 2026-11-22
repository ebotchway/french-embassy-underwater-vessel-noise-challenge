# Run guide — Experiment 006

Run from the repository root with `.venv` active.

```bash
cp ~/Downloads/run_experiment006_contrast_robustness_v1.py scripts/
python scripts/run_experiment006_contrast_robustness_v1.py --self-test-only
python scripts/run_experiment006_contrast_robustness_v1.py --train 1200 --dev-pairs 120
```

The script uses only the `exp004_train` allocation from the three frozen manifests. It validates SHA256 hashes and writes atomically to a new directory, refusing to overwrite an existing result. No previous evaluation source pool or Zindi Test data is used.

Review:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
root=Path('reports/baselines/experiment006_internal_contrast_v1')
r=pd.read_csv(root/'development_summary.csv')
for group in ['overall','intervention_plus_zeros','requested_share','mask','station']:
    print('\n',group)
    z=r.loc[r.group.eq(group)]
    print(z[['model','scenario','n','pairs','rmse','mae','roc_auc','average_precision','zero_false_positive_rate_at_half']].to_string(index=False))
PY
```

Do not re-run under alternative seeds to obtain more favorable results. The `reports/` results directory may be ignored by Git; preserve its manifest and prediction CSVs alongside the analysis archive. Commit only new code and the reviewed protocol, then consolidate actual findings into research documents at the next checkpoint.

```bash
git add scripts/run_experiment006_contrast_robustness_v1.py
git commit -m "Add source-isolated internal spectral contrast Experiment 006"
```

## Interpretation caveats

This is an *internal development* experiment, not an untouched confirmation test. Shared stations, hardware, recording conditions and historically used training sources remain. All spectral adjustments are counterfactual controls, not authentic vessel passage recordings. Detector AUC is calculated only when both zero and positive targets are present. Do not choose a final submission based only on this experiment.
