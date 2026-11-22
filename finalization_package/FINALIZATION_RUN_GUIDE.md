# Final research-submission preparation — 22 November 2026

**Scope:** Review provenance and submission readiness; assemble an explicitly *working* research manuscript. Neither script accesses Zindi `Test.csv`, protected Experiment 004 evaluation, sealed holdouts, or genuine natural-vessel labels. No models are trained. No data are uploaded or posted.

## 1. Installation

From the project root with `.venv` activated:

```bash
mkdir -p finalization_package research_checkpoints
unzip -n ~/Downloads/RESEARCH_FINALIZATION_PHASE1.zip -d finalization_package
cp finalization_package/audit_research_submission_readiness_v1.py scripts/
cp finalization_package/build_research_report_draft_v1.py scripts/
cp finalization_package/FINALIZATION_RUN_GUIDE.md research_checkpoints/
cp finalization_package/RESEARCH_SOURCE_REGISTER.md research_checkpoints/
python -m py_compile scripts/audit_research_submission_readiness_v1.py scripts/build_research_report_draft_v1.py
```

## 2. Test then audit actual source/provenance status

```bash
python scripts/audit_research_submission_readiness_v1.py --self-test-only
python scripts/build_research_report_draft_v1.py --self-test-only
python scripts/audit_research_submission_readiness_v1.py
cat reports/audits/research_submission_readiness_v1/READINESS_REPORT.md
```

The audit is intentionally fail-closed for missing/ambiguous sources: a file's presence does not certify pressure calibration or geographic deployment. It lists **candidate** locations separately and blocks labeling them verified.

## 3. Generate report locally

```bash
python -m pip install reportlab
python scripts/build_research_report_draft_v1.py
```

Outputs: `reports/research_manuscript_working_v1/RESEARCH_WORKING_DRAFT.md`, `RESEARCH_WORKING_DRAFT.pdf` (when ReportLab installed), `BUILD_MANIFEST.json`.

This is a **working draft only**. It contains results transcribed from observed 2026-11-22 terminal output; the result tables and figures still require a final evidence-to-code trace and source confirmation. The report builder is deterministic in its experimental text and refuses to overwrite existing output. For a separate reproducibility run, use `--output-dir reports/research_manuscript_working_v2`.

## 4. Human verification gates

- Confirm **actual local VTUAD release** by matching `dataset_card.md`, metadata manifests, WAV SHA-256 ledger and Zenodo record. A candidate version is not a confirmed published release identifier. **Never relabel a local v1.3.0 candidate as v1.3.2** without verified file-level match.
- Verify licensing, attribution and allowed redistribution; publicly disclose dataset/version/access in the competition discussion and record the post URL. The existing draft is not evidence of publication.
- Find the South African acoustic archive's original instrument/deployment records. Candidate site latitude/longitude values are **not** sensor positions; Saldanha Bay has unresolved alternatives.
- Verify units, measurement convention and recorder calibration before calling relative levels calibrated underwater pressure. Otherwise use "archive-reported band levels" and "relative acoustic excursions."
- For *vessel-related* acoustic-activity maps or natural-vessel validation, obtain independent, time-aligned AIS/port/field observations with recording/coverage windows and uncertainties. Do not fabricate ground truth.
- Ensure source independence and avoid further tuning on Experiment 007 assessment or hidden test.
- Reconfirm challenge PDF+code deadline and template requirements before final upload.

## 5. Git hygiene

```bash
git status --short
# Review before staging; do not add raw WAVs, organizer contest data, Test.csv, .venv or generated reports.
git add scripts/audit_research_submission_readiness_v1.py scripts/build_research_report_draft_v1.py research_checkpoints/FINALIZATION_RUN_GUIDE.md research_checkpoints/RESEARCH_SOURCE_REGISTER.md
git diff --cached --check
git commit -m "Add provenance-gated research submission finalization tools"
```

If your previous cleanup is done, **do not restore `PHASE3_CHECKPOINT`**: its material has been consolidated into `CHECKPOINT.md`.

## 6. Deliverable and research scope

The included `RESEARCH_WORKING_DRAFT.pdf` is a preview written from observed local results. It is not the competition-ready PDF: complete your manuscript figures, authentic references, exact data version, method reproduction and evidence-required maps before submission. No script in this package publishes anything to Zindi.
