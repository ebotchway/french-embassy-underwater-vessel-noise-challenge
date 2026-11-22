# Experiment 006C — Source-connected calibration and hybrid diagnostic

**Status:** Implemented and tested with constructed data; **not run on the Mac acoustic project**.

## Purpose

Use saved Experiment 006 predictions only, with Experiment 006B's checksum, pairing, and leakage checks. No vessel model training, no frozen Experiment 004 evaluation, no sealed station holdouts, and no Zindi test. Strictly separate calibration and assessment pairs by the transitive connected components of shared **background runs, vessel hull IDs, and clip IDs**. A shared source is a dependency even if the matching pair does not directly share the other source.

## Run

From project root with `.venv` active:

```bash
unzip -n ~/Downloads/EXPERIMENT006C_IMPLEMENTATION.zip -d experiment006c_package
cp experiment006c_package/audit_experiment006c_calibration_v1.py scripts/
python -m py_compile scripts/audit_experiment006c_calibration_v1.py
python scripts/audit_experiment006c_calibration_v1.py --self-test-only
python scripts/audit_experiment006c_calibration_v1.py
```

The installed `scripts/audit_experiment006b_readonly_v1.py` must remain alongside this script. By default the script reads `reports/baselines/experiment006_internal_contrast_v1/` and creates `reports/audits/experiment006c_calibration_v1/`. It **refuses to overwrite** existing outputs. No archive or original acoustic source is loaded.

## Prespecified decisions

- Source graph: 120 existing development pairs (observed); group same run/hull/clip into connected components. All nodes of a component remain together.
- Seed 20261122; seek ~30% assessment pairs, at least 12 per partition. Split selection uses **identities and station labels only**, not model errors, scores or outcomes.
- If no feasible split exists, save component counts and explicit feasibility block. **Do not weaken the hull/run/clip restriction** to get a more attractive result.
- Original-only detector, varied detector: on calibration examples only, choose from thresholds `.05,.10,...,1.00` to maximize observed-positive recall subject to **no-addition control FPR <= 0.25**. Tie break: lower FPR then higher threshold. Report both fixed .5 and selected operating points on assessment.
- Hybrid: use **contrast-varied share regression prediction** and **original-only detector score**, both already produced by Experiment 006. Does not involve model retraining. Assess original fixed and calibration-selected thresholds.
- Diagnostic Platt/logistic mapping is fitted on calibration examples only and reports **assessment** Brier/AUC. It is *not* deployed for predictions, and is not a calibration of real-world prevalence.
- Show observed-zero and genuine no-addition controls separately, and station, mask and intervention groups with their denominators.

## Outputs

- `audit_manifest.json` — input hashes, source-component sizes, allocations, selection rules and status.
- `source_component_allocation.csv` — identity-connected allocation of background/source pairs.
- `RESEARCH_NOTE.md` — findings, safeguards, limitations and any infeasibility.
- If feasible: `policy_metrics.csv`, `threshold_calibration_sweep.csv`, `platt_calibration_diagnostic.csv`, `assessment_segments.csv`, `no_addition_scores.csv`.

Inspect:

```bash
cat reports/audits/experiment006c_calibration_v1/RESEARCH_NOTE.md
python - <<'PY'
from pathlib import Path
import pandas as pd
root=Path('reports/audits/experiment006c_calibration_v1')
for name in ('policy_metrics.csv','platt_calibration_diagnostic.csv','assessment_segments.csv'):
    f=root/name
    if f.exists():
        data=pd.read_csv(f)
        print('\n',name,'rows=',len(data))
        print(data.head(25).to_string(index=False))
PY
```

## Interpretation limits

- This **is not** a new independent model benchmark: Experiment 006 development observations were inspected when proposing calibration/hybrid options. Assessment samples are unused **for choosing 006C thresholds**, but not untouched by earlier research decisions.
- Thresholds and prevalence-dependent Brier/AP reflect a controlled synthetic distribution, not actual South African vessel prevalence.
- Calibration/assessment components may be few or strongly station-imbalanced. Never claim uniform station coverage without checking.
- A feasible source split does not establish deployment-level independence or true vessel-free labels.
- Do not compare candidate thresholds repeatedly against the assessment to pick a preferred one. Do not use these outputs to adjust the frozen Experiment 004 evaluation or hidden-test submissions.

## Commit

```bash
git add scripts/audit_experiment006c_calibration_v1.py research_checkpoints/EXPERIMENT006C_RUN_GUIDE.md
git commit -m "Add source-connected Experiment 006C calibration diagnostic"
```

Documentation in notebook and learning log should be updated **after observing the real execution** (successful analysis or feasibility block).
