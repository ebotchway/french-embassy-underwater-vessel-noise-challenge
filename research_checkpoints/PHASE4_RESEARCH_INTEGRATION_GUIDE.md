# Phase 4 — evidence-gated research integration

**Purpose.** Assemble a research *working draft* from already verified local Experiment 007 metrics and the actual-data six-station acoustic results. This does not approve or submit the manuscript, train models, create pressure maps, or post the external-data declaration.

## Installation

From the research project root:

```bash
mkdir -p research_phase4_package research_checkpoints
unzip -n ~/Downloads/SUBMISSION_EVIDENCE_PHASE4.zip -d research_phase4_package
cp research_phase4_package/audit_and_assemble_research_v1.py scripts/
cp research_phase4_package/RUN_GUIDE.md research_checkpoints/PHASE4_RESEARCH_INTEGRATION_GUIDE.md
python -m py_compile scripts/audit_and_assemble_research_v1.py
python scripts/audit_and_assemble_research_v1.py --self-test-only
python scripts/audit_and_assemble_research_v1.py
```

The builder stops on missing/mismatched input hashes, non-validated real-acoustic evidence, incomplete Experiment 007 policy metrics or an existing output directory. It never writes to prior results. Default output: `reports/research_manuscript_v1/submission_evidence_phase4_v1/`.

Review:

```bash
cat reports/research_manuscript_v1/submission_evidence_phase4_v1/RESEARCH_INTEGRATION_WORKING.md
python - <<'PY'
import json
from pathlib import Path
p=Path('reports/research_manuscript_v1/submission_evidence_phase4_v1/AUDIT_MANIFEST.json')
x=json.loads(p.read_text())
print('Evidence gate statuses:')
for k,v in x['evidence_gates'].items(): print(k,':',v)
print('Release provenance:',x['vtuad_version_evidence'])
PY
```

The **PDF is a compact integration draft** showing the core evidence claims and five already-verified acoustic figures. The full prior manuscript, acoustic appendix and Experiment 007 addendum are copied as separate Markdown components, retaining their original wording. This package does not falsely declare that all three texts have already been completely copy-edited into one finished article.

Only commit source code and guide after checking diffs; leave automatically generated report data out of Git unless specifically required by your research reproducibility policy.

```bash
git add scripts/audit_and_assemble_research_v1.py research_checkpoints/PHASE4_RESEARCH_INTEGRATION_GUIDE.md
git diff --cached --check
git commit -m "Add evidence-gated Phase 4 research integration"
```

## Why gates remain open

1. **Release:** reconcile local VTUAD files against *publisher-provided* file lists, original download archives and upstream record, not only local SHA ledger / dataset card.
2. **Public declaration:** post an accurate version-qualified external-data declaration to the permitted Zindi discussion and record the actual URL; local text is just a draft. Do not assert that this script posted anything.
3. **Deployment:** obtain sensor ID, deployment date range, CRS, location source and instrument sensitivity before pressure maps.
4. **Vessel attribution:** use independent validated timestamped observations such as suitably licensed AIS/port records with defensible coverage; otherwise label observations as acoustic excursions only.
5. **Submission:** verify official PDF, code size and date requirements immediately before upload. The current package is **NOT** the final submission.

## Sources

- Competition: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
- Research guidance: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36275
- VTUAD record: https://zenodo.org/records/23037895
- ONC archive: https://data.oceannetworks.ca/
