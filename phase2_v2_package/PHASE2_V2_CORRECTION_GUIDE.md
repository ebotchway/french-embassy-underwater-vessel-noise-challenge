# Phase 2 v2 — Real-acoustic evidence correction

## Why this version exists

The original Phase 2 auditor looked for `reports/audits/real_acoustic_development_v1/station_summary.csv`, but the completed real-acoustic development report actually resides at `reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv`.

Version 2 uses the **actual existing report**. It does not regenerate station statistics or overwrite the original v1 audit. It validates column names, unique nonempty station IDs, positive integer run/recorder counts, finite numerical statistics, and nonnegative variability/range values. If the source fails those checks, the evidence gate is `SOURCE_INVALID`, not `SOURCE_PRESENT`.

Passing those checks is only schema/numerical validation; it does **not** verify source-level provenance, physical units, calibration, hydrophone deployment coordinates, actual vessel passages, or the precise VTUAD release. Nor does it approve a submission.

## Mac commands from the repository root

Copy `audit_research_evidence_phase2_v2.py` to `scripts/`, then:

```bash
python -m py_compile scripts/audit_research_evidence_phase2_v2.py
python scripts/audit_research_evidence_phase2_v2.py --self-test-only
python scripts/audit_research_evidence_phase2_v2.py
```

The default destination is **new**: `reports/audits/research_evidence_phase2_v2/` and will not overwrite the original `research_evidence_phase2_v1/`. If the directory already exists, choose another via `--output reports/audits/research_evidence_phase2_v2_recheck` rather than deleting prior evidence.

Inspect:

```bash
cat reports/audits/research_evidence_phase2_v2/RESEARCH_EVIDENCE_REPORT.md
python - <<'PY'
import pandas as pd
r='reports/audits/research_evidence_phase2_v2'
f=pd.read_csv(f'{r}/file_inventory.csv')
print(f[['label','relative_path','present']].to_string(index=False))
g=pd.read_csv(f'{r}/claim_evidence_gates.csv')
print(g[['gate','status']].to_string(index=False))
PY
```

Expected on the previously reported Mac source inventory: the corrected real station summary path should be present, giving **14/14 named file paths**, if no files were moved. The real-acoustic gate should be `SOURCE_PRESENT_SCHEMA_VALIDATED` if the rows pass the checks. The previously measured development sample comprises six stations and 460 sampled runs; inspect the v2 report for actual values rather than assuming them. Other scientific blocks remain unchanged.

Commit the new script and this guide after reviewing results:

```bash
git add scripts/audit_research_evidence_phase2_v2.py research_checkpoints/PHASE2_V2_CORRECTION_GUIDE.md
git diff --cached --check
git commit -m "Correct Phase 2 real-acoustic evidence path with schema validation"
```

## Scientific interpretation

- A missing **named file** was a path error; it was not proof of missing acoustic observations.
- The corrected file's descriptive station statistics support station-level summaries, not calibrated underwater sound-pressure levels.
- A coordinate-provenance **template** existing on disk is not proof that hydrophone coordinates were verified.
- Keep the v1 audit and the v2 audit as separate records of the correction.
- Preserve the frozen Experiment 004 evaluation, original validation, sealed holdouts, and Zindi test data.
