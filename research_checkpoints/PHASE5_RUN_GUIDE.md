# Phase 5 — Provenance reconciliation and scientific review manuscript

**Research state:** Phase 4 integrated on 22 Nov 2026, Git commit `1208eb1`; Experiment 007 frozen synthetic assessment, no further tuning. Research PDF is NOT submission-cleared. The Phase 5 tools do not alter the notebook, model data, frozen validation, sequestered Experiment 004 evaluation or Test.csv.

## Installation (the ZIP has a flat file layout, without nested package folders)

From your Mac project root with `.venv` active:

```bash
mkdir -p research_phase5_package scripts research_checkpoints
unzip -n ~/Downloads/RESEARCH_SUBMISSION_PHASE5.zip -d research_phase5_package
cp research_phase5_package/audit_submission_provenance_v1.py scripts/
cp research_phase5_package/verify_publisher_assets_v1.py scripts/
cp research_phase5_package/build_scientific_submission_review_v1.py scripts/
cp research_phase5_package/PHASE5_RUN_GUIDE.md research_checkpoints/
cp research_phase5_package/EXTERNAL_DATA_DECLARATION_DRAFT_PHASE5.md research_checkpoints/
cp research_phase5_package/REVIEW_AND_SUBMISSION_GATES.md research_checkpoints/

python -m py_compile scripts/audit_submission_provenance_v1.py scripts/verify_publisher_assets_v1.py scripts/build_scientific_submission_review_v1.py
python scripts/audit_submission_provenance_v1.py --self-test-only
python scripts/verify_publisher_assets_v1.py --self-test-only
python scripts/build_scientific_submission_review_v1.py --self-test-only
```

## Actual evidence audit (reads only metadata and already built research reports)

```bash
python scripts/audit_submission_provenance_v1.py
cat reports/audits/submission_provenance_phase5_v1/EVIDENCE_REPORT.md
```

This generates a SHA manifest, a parsed list of the 525 ledger entries, and a **blank hydrophone-deployment evidence template**. The template is NOT a map or location verification. It records missing sensor/deployment IDs, matching recording periods, authoritative coordinates, CRS, and evidence URL. The original publisher download archive is not presumed present.

## Reconcile publisher archive checksums (optional; do only after obtaining authoritative records)

Fill a copy of `PUBLISHER_ASSETS_TEMPLATE.csv` at `research_checkpoints/PUBLISHER_ASSETS_TO_VERIFY.csv`, with accurate local relative paths and publisher-published MD5/SHA256 sums. Never guess checksums or source URLs. This step can be slow for multi-gigabyte archives.

```bash
cp research_phase5_package/PUBLISHER_ASSETS_TEMPLATE.csv research_checkpoints/PUBLISHER_ASSETS_TO_VERIFY.csv
# Edit the CSV with independently verified publisher asset hashes and real local archive paths.
python scripts/verify_publisher_assets_v1.py \
  --publisher-asset-manifest research_checkpoints/PUBLISHER_ASSETS_TO_VERIFY.csv
```

**Note:** An individual matching asset hash does not certify that all 525 WAVs were extracted from the same release, or authenticate the download provenance of an untrusted list. Attach the publisher source and asset version; inspect the cited publisher record yourself. Do not run the verifier on the untouched blank template.

## Assemble the actual, coherent research review paper

Requires successful Phase 4 assembly and the already verified v2 acoustic figures.

```bash
python scripts/build_scientific_submission_review_v1.py
python scripts/build_scientific_submission_review_v1.py --verify-only
```

Review:

```bash
ls -lh reports/research_manuscript_v1/submission_review_phase5_v1/
cat reports/research_manuscript_v1/submission_review_phase5_v1/RESEARCH_MANUSCRIPT_REVIEW.md
```

The PDF is `reports/research_manuscript_v1/submission_review_phase5_v1/RESEARCH_MANUSCRIPT_REVIEW.pdf`. Tables round presentation values; source CSV retains its original full precision. The five figures are copied byte-identically from Phase 4 with checksum verification.

### If a run fails

- A source missing error means the authoritative Phase 4 or real acoustic output is absent; do **not** invent placeholders.
- An existing output error is intentional. Keep the prior evidence and pick a versioned `--output` destination after carefully reviewing why another run is needed.
- The auditor may correctly classify all source-release/coordinate/vessel/calibration gates as unresolved despite reporting `PASS` for the inventory.
- No automatic public external-data posting, Zindi submission or manuscript approval occurs.

## Commit only source code and research instructions

```bash
git status --short

git add \
  scripts/audit_submission_provenance_v1.py \
  scripts/verify_publisher_assets_v1.py \
  scripts/build_scientific_submission_review_v1.py \
  research_checkpoints/PHASE5_RUN_GUIDE.md \
  research_checkpoints/EXTERNAL_DATA_DECLARATION_DRAFT_PHASE5.md \
  research_checkpoints/REVIEW_AND_SUBMISSION_GATES.md

git diff --cached --check
git diff --cached --stat
# Only when reviewed:
git commit -m "Add Phase 5 provenance verification and scientific manuscript review"
```

Avoid `git add -A` and do not commit extracted `research_phase5_package/`, large WAV archives, raw competition files or generated report artifacts without a deliberate data-release decision.

## External source records to inspect independently

- VTUAD Zenodo record: https://zenodo.org/records/23037895
- Challenge data and external-data policy: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data
- Zindi research submission information: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35821
- Ocean Networks Canada: https://data.oceannetworks.ca/

Research write-up upload and runnable code/notebook are separate, with an official published deadline of **13 December 2026, 11:30 UTC**. Verify the official form and size limits immediately before submission. The public score is not evidence of real-domain vessel detection.
