# Real acoustic results v2 — missing robust-z correction

## Root cause

The Phase 3 v1 builder requires all `station_robust_z` values to be finite, but this derived diagnostic can legitimately be undefined. Its input comes from `reports/research_manuscript_v1/station_relative_anomaly_proxy.csv`. The log only establishes nonfinite values, **not why** they are undefined. Do not fill them with zero or assume that they are acoustic anomalies.

The verifier's missing-manifest error is a consequence of the builder stopping before output generation. It is not an independent source-data failure.

## What v2 does

- Retains strict checks on `window_mean_level` and `station_relative_level_deviation` and requires the original station/run counts to match.
- Treats `station_robust_z` as an optional, derived metric; converts nonfinite entries to explicitly undefined values **for analysis only**, without editing the source CSV.
- Computes any robust-z descriptive statistics only over defined values, with coverage counts and no interval below ten valid runs.
- Records the number of defined and undefined robust-z values in `robust_z_availability.csv`, the results table, the Markdown/PDF appendix and the source-hashed manifest.
- Updates Matplotlib's deprecated `vert=False` to `orientation='horizontal'`.
- Writes to **`reports/research_manuscript_v1/real_acoustic_results_v2/`**, never overwriting the older v1 output or the original data. The v1 script is retained as historical evidence of the failure.
- Supplies a v2-matched independent output verifier and optional documentation updater.

## Install from the project root

```bash
mkdir -p real_acoustic_fix_v2_package research_checkpoints
unzip -n ~/Downloads/REAL_ACOUSTIC_RESULTS_V2_FIX.zip -d real_acoustic_fix_v2_package
cp real_acoustic_fix_v2_package/*.py scripts/
cp real_acoustic_fix_v2_package/REAL_ACOUSTIC_V2_FIX_RUN_GUIDE.md research_checkpoints/
python -m py_compile scripts/build_real_acoustic_results_v2.py scripts/verify_real_acoustic_results_v2.py scripts/append_real_acoustic_checkpoint_v2.py
```

## Test and run

```bash
python scripts/build_real_acoustic_results_v2.py --self-test-only
python scripts/build_real_acoustic_results_v2.py
python scripts/verify_real_acoustic_results_v2.py
cat reports/research_manuscript_v1/real_acoustic_results_v2/REAL_ACOUSTIC_RESULTS_APPENDIX.md
cat reports/research_manuscript_v1/real_acoustic_results_v2/robust_z_availability.csv
```

Inspect the PDF and figures. **Do not run the v1 verifier** against v2 outputs. Never apply the checkpoint updater before v2 build and verification succeed.

## Optional documentation synchronization

```bash
python scripts/append_real_acoustic_checkpoint_v2.py
python scripts/append_real_acoustic_checkpoint_v2.py --apply
```

The update backs up original notebook/log/checkpoint and refuses a duplicate. Do not execute it if you already applied a similar checkpoint without reviewing content, or if you have unresolved edits.

## Git

```bash
git add scripts/build_real_acoustic_results_v2.py scripts/verify_real_acoustic_results_v2.py scripts/append_real_acoustic_checkpoint_v2.py research_checkpoints/REAL_ACOUSTIC_V2_FIX_RUN_GUIDE.md
# Only stage the following if you actually ran the optional documentation update:
# git add notebooks/01_south_african_acoustic_eda.ipynb PROJECT_LEARNING_LOG.md CHECKPOINT.md
git diff --cached --check
git diff --cached --stat
git commit -m "Handle undefined station robust-z diagnostics in acoustic appendix"
```

## Interpretation

Undefined robust z-scores should prompt a separate read-only review of how the station median/MAD or other robust scale was computed. A derived diagnostic being undefined does not invalidate the base acoustic level and deviation statistics when their input checks pass. Never substitute artificial robust z-scores, claim calibrated sound pressure, infer vessel passage, or promote site-level coordinates to verified hydrophone deployment positions.
