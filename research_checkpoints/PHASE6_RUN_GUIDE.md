# Phase 6 — Independent scientific submission audit

## Scope

Read-only check of **existing** Phase 5 artifacts: `reports/research_manuscript_v1/submission_review_phase5_v1/RESEARCH_MANUSCRIPT_REVIEW.md`, `.pdf`, `MANUSCRIPT_REVIEW_MANIFEST.json`, the five referenced figures, and selected authoritative local scripts. **Does not generate a new model, re-run training, tune thresholds, alter prior manuscripts, verify hydrophone coordinates or approve a competition submission.**

Latest public competition references (reviewed 22 November 2026):
- https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
- https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35678
- https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35821

The research-writeup instructions describe a defensible vessel-related pressure/activity map, comparison to real South African acoustic patterns, a separately runnable code/notebook file and a **10 MB per-file** maximum; submit by **13 December 2026 11:30 UTC**. Recheck directly before upload.

## Install

Extract the ZIP downloaded from ChatGPT into an unpacked, *ignored* working directory:

```bash
mkdir -p phase6_audit_package research_checkpoints
unzip -n ~/Downloads/RESEARCH_SUBMISSION_PHASE6_AUDIT.zip -d phase6_audit_package
cp phase6_audit_package/audit_final_submission_v1.py scripts/
cp phase6_audit_package/PHASE6_RUN_GUIDE.md phase6_audit_package/PHASE6_SCIENTIFIC_DECISION_PROTOCOL.md research_checkpoints/
```

The ZIP is flat and intentionally contains no Python bytecode or generated dataset copies.

## Validate

```bash
python -m py_compile scripts/audit_final_submission_v1.py
python scripts/audit_final_submission_v1.py --self-test-only
```

To include PDF extraction/structure checking:

```bash
python -m pip install pypdf
```

For PDF visual QA, open the actual Phase 5 PDF or render it locally. The script's text checks are not a substitute for inspecting every page.

## Run against real Phase 5 outputs

```bash
python scripts/audit_final_submission_v1.py
cat reports/audits/final_submission_review_phase6_v1/SCIENTIFIC_AUDIT_REPORT.md
python - <<'PY'
import csv
from pathlib import Path
p=Path('reports/audits/final_submission_review_phase6_v1/FINDINGS.csv')
with p.open(newline='') as f:
    for row in csv.DictReader(f):
        if row['severity']=='BLOCKER' and row['status'] not in ('PASS','VERIFIED'):
            print(row['code'],row['status'],'—',row['required_action'])
PY
```

The audit **refuses to overwrite** its prior output. Do not delete or edit a completed audit merely to achieve a different status. After new evidence or manuscript revisions, use a new explicitly named output path, for example:

```bash
python scripts/audit_final_submission_v1.py --output reports/audits/final_submission_review_phase6_v2
```

The result may display technical `PASS` checks while the overall research submission remains `NOT_CLEARED_AUTOMATICALLY`. This is expected.

## Commit the audit tool

```bash
git add scripts/audit_final_submission_v1.py research_checkpoints/PHASE6_RUN_GUIDE.md research_checkpoints/PHASE6_SCIENTIFIC_DECISION_PROTOCOL.md
git diff --cached --check
git diff --cached --stat
git commit -m 'Add evidence-gated final scientific submission audit'
```

Generated reports live under `reports/` and should not be accidentally committed alongside source code.
