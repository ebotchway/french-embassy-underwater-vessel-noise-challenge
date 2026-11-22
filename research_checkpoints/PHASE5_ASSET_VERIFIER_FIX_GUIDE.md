# Phase 5 — Publisher asset verifier correction

## Cause
The Phase 5 v1 `--self-test-only` failed at `Path.is_relative_to()` because the fixture's root was not normalized with `Path.resolve()` before comparing it with a resolved candidate file path. On macOS `/var/...` may resolve to `/private/var/...`, so the check erroneously reported `Path outside project root`.

## Fix
`verify_publisher_assets_v2.py` canonicalizes the project root before containment checking. It retains validation of relative, non-parent-traversing asset paths, and rejects symlinks resolving outside the root. An additional self-test checks alias roots and symlink escape rejection. A matching SHA-256/MD5 hash establishes agreement with a *supplied* publisher asset checksum, not the provenance or full identity of the dataset version; independently authenticate the publisher checksum source.

## Install from repository root

```bash
mkdir -p phase5_asset_fix_package research_checkpoints
unzip -n ~/Downloads/PHASE5_PUBLISHER_ASSET_VERIFIER_FIX.zip -d phase5_asset_fix_package
cp phase5_asset_fix_package/verify_publisher_assets_v2.py scripts/
cp phase5_asset_fix_package/PHASE5_ASSET_VERIFIER_FIX_GUIDE.md research_checkpoints/
python -m py_compile scripts/verify_publisher_assets_v2.py
python scripts/verify_publisher_assets_v2.py --self-test-only
```

The old v1 file can remain for reproducibility; v2 is the corrected one. Do not run the real publisher-checksum comparison with fabricated checksums: it requires real source hashes entered into the Phase 5 `PUBLISHER_ASSETS_TEMPLATE.csv` and the exact corresponding retained downloaded archive file(s).

For the other two Phase 5 tools, continue as normal:

```bash
python scripts/audit_submission_provenance_v1.py
cat reports/audits/submission_provenance_phase5_v1/EVIDENCE_REPORT.md
python scripts/build_scientific_submission_review_v1.py
python scripts/build_scientific_submission_review_v1.py --verify-only
```

The publisher asset verifier is optional, and its failure on a self-test does not invalidate the already-passed provenance inventory self-test or manuscript self-test.

When a publisher-supplied checksums manifest has been populated and independently reviewed, run:

```bash
python scripts/verify_publisher_assets_v2.py \
  --publisher-asset-manifest research_checkpoints/PUBLISHER_ASSETS_VERIFIED.csv \
  --output reports/audits/publisher_asset_checks_phase5_v2.json
```

This reports *per-file* matches only, not full release authentication. It refuses to overwrite an existing output file.

After testing, commit only the new source and guide:

```bash
git add scripts/verify_publisher_assets_v2.py research_checkpoints/PHASE5_ASSET_VERIFIER_FIX_GUIDE.md
git diff --cached --check
git commit -m "Fix Phase 5 publisher asset self-test path canonicalization"
```
