# Research finalization, Phase 2 — execution guide

**Date:** 2026-11-22. **Scope:** provenance, scientific claims, and source-readiness evidence. This is **not** a claim that the final research PDF or public data declaration is complete.

## Prerequisites

Run from the root of `french-embassy-underwater-vessel-noise-challenge`, with the existing virtual environment activated. The script uses Python standard library only. It does not download external resources, open contest test data, fit models, or alter historical experiment outputs.

## Install

Download `RESEARCH_FINALIZATION_PHASE2.zip`. From the project root:

```bash
mkdir -p research_phase2_package research_checkpoints
unzip -n ~/Downloads/RESEARCH_FINALIZATION_PHASE2.zip -d research_phase2_package
cp research_phase2_package/audit_research_evidence_phase2_v1.py scripts/
cp research_phase2_package/PHASE2_RUN_GUIDE.md research_checkpoints/
cp research_phase2_package/RESEARCH_RELEASE_AND_LOCATION_REVIEW.md research_checkpoints/
cp research_phase2_package/EXTERNAL_DATA_DECLARATION_REVIEW_TEMPLATE.md research_checkpoints/
python -m py_compile scripts/audit_research_evidence_phase2_v1.py
python scripts/audit_research_evidence_phase2_v1.py --self-test-only
```

## Run

First run the Phase 1 readiness auditor **only if it has not already been run**, since it refuses to overwrite outputs:

```bash
python scripts/audit_research_submission_readiness_v1.py
```

Run the new Phase 2 audit:

```bash
python scripts/audit_research_evidence_phase2_v1.py
cat reports/audits/research_evidence_phase2_v1/RESEARCH_EVIDENCE_REPORT.md
```

The Phase 2 script outputs:

- `file_inventory.csv` — expected local research evidence paths and SHA-256 digests.
- `local_vtuad_version_evidence.json` — dataset-card version tokens, **count of checksum ledger entries** (not rehashed WAV verification), and release caveat.
- `unverified_location_registry.csv` — candidate coordinates copied *as candidates* with explicit unverified status.
- `claim_evidence_gates.csv` — blocked/present claims and required next evidence.
- `RESEARCH_EVIDENCE_REPORT.md` — human-readable research decision record.
- `audit_manifest.json` — run provenance and prohibited inputs.

**Never** copy `unverified_location_registry.csv` into the previously built `STATION_COORDINATES_PROVENANCE_TEMPLATE.csv` and mark it verified. The latter requires an authoritative sensor/deployment match.

## Verify genuine source identity

Read the existing local dataset card and checksum manifest; review the upstream publisher's release notes alongside the actual **downloaded archive** and **strict-interval WAV checksum** evidence. In particular, match local archive/metadata file hash, release-specific filenames and the locally applied preprocessing version before publicly declaring that a published version is what you used.

The publisher states v1.3.2 adds supplementary wraparound-splice audit files while the audio, registry, feature store and frozen split remain byte-identical to v1.3.0/v1.3.1. This is a **public release relationship**, not proof of your local source identity. Do not change the existing preprocessing or add new external data during verification.

## Review source locations

Only an original source record connecting station-export ID, hydrophone or recorder ID, measurement time, deployment ID and verified WGS84/other-declared CRS coordinates may authorize a point on a sensor-position map. Station location candidates are suitable leads for research, not authoritative deployment coordinates. Keep Saldanha Bay's two alternatives unresolved until original evidence assigns the correct deployment.

## Commit

After examining the report and checking the change list:

```bash
git add scripts/audit_research_evidence_phase2_v1.py \
    research_checkpoints/PHASE2_RUN_GUIDE.md \
    research_checkpoints/RESEARCH_RELEASE_AND_LOCATION_REVIEW.md \
    research_checkpoints/EXTERNAL_DATA_DECLARATION_REVIEW_TEMPLATE.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add Phase 2 acoustic provenance and evidence-gate audit"
```

Generated `reports/` artifacts are intentionally local; do not commit restricted raw data or audio. A separate `.bak` or extracted package folder does not need tracking.

## Next decisions after the audit

1. Confirm the exact VTUAD release used; retain immutable hashes and publicly declare licensed external data once accurate.
2. Find original matching hydrophone/deployment position records or document that spatial claims remain unsupported.
3. Turn real station acoustic summaries into *non-geographic*, coverage-aware figures first; do not equate level excursions with natural vessel presence.
4. Update the existing manuscript only where verified facts justify the change; prepare PDF and runnable code with source references.
5. Avoid model/threshold tuning on Experiment 007 assessment or hidden Zindi data.

## Common failures

- `FileExistsError`: existing report is protected. Read it; if a new audit is necessary, specify a new `--output` directory.
- `stations.csv missing expected candidate headers`: inspect original metadata format instead of guessing column meanings.
- `SOURCE_MISSING`: locate source evidence or transparently label it absent; do not fabricate it.

**Research write-up deadline currently advertised: 2026-12-13 11:30 UTC. Recheck Zindi immediately before submission.**
