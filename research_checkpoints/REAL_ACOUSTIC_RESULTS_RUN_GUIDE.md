# Real-acoustic results and manuscript appendix — Execution guide

**Status:** Code and synthetic-fixture tests completed by the authoring environment. Nothing in this package has been executed against your Mac archive.

## Scientific purpose

Convert previously validated real South African station and run summaries into a reproducible **descriptive** section of the research report, including figure captions, 25-band spectral-shape comparisons, coverage transparency and run-clustered uncertainty. This phase is **not** a new synthetic experiment, detector validation, AIS reconstruction, sound-pressure calibration, or authorization to publish a geographic vessel-activity map.

## Required local inputs

The following paths must already exist. Do not regenerate the raw archive as a prerequisite:

- `reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv` — validated six-station summary with `station`, `sampled_runs`, `recorders`, level and temporal variation fields.
- `reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv` — station/band frequency and peak-referenced spectral distribution.
- `reports/research_manuscript_v1/station_relative_anomaly_proxy.csv` — one row per *distinct* `(station, run_id)`, with relative level statistics.

**Important:** The last file is not automatically an unbiased, vessel-labelled set. The builder deliberately rejects repeated `(station, run_id)` entries or inconsistent run totals; these must be investigated against the actual generator instead of quietly discarding rows.

## Installation (from repo root with `.venv` activated)

```bash
mkdir -p real_acoustic_results_package
unzip -n ~/Downloads/REAL_ACOUSTIC_RESULTS_PHASE3.zip -d real_acoustic_results_package
cp real_acoustic_results_package/*.py scripts/
cp real_acoustic_results_package/REAL_ACOUSTIC_RESULTS_RUN_GUIDE.md research_checkpoints/
python -m pip show numpy pandas matplotlib reportlab
```

If a package is missing, install only that missing requirement into `.venv` using `python -m pip install <package>` and record its installed version. Do not silently upgrade all packages.

## Test and run

```bash
python -m py_compile scripts/build_real_acoustic_results_v1.py \
  scripts/verify_real_acoustic_results_v1.py \
  scripts/append_real_acoustic_checkpoint_v1.py
python scripts/build_real_acoustic_results_v1.py --self-test-only
python scripts/build_real_acoustic_results_v1.py
python scripts/verify_real_acoustic_results_v1.py
```

Output is **new**, refusing overwrite: `reports/research_manuscript_v1/real_acoustic_results_v1/`. Expected files:

- `validated_station_summary.csv`
- `run_bootstrap_intervals.csv`
- `figures/station_run_coverage.png`
- `figures/station_run_levels_intervals.png`
- `figures/station_temporal_variability.png`
- `figures/station_relative_spectral_shapes.png`
- `figures/station_relative_deviations.png`
- `REAL_ACOUSTIC_RESULTS_APPENDIX.md`
- `REAL_ACOUSTIC_RESULTS_APPENDIX.pdf`
- `manifest.json`

The PDF is a **draft appendix**, not the final submitted PDF. Verify the generated figures and PDF locally; do not submit it without original source citations, hydrophone metadata, version declaration and code reproducibility review.

## Optional synchronized documentation update

First review the Markdown appendix and its interpretation. Then:

```bash
python scripts/append_real_acoustic_checkpoint_v1.py
python scripts/append_real_acoustic_checkpoint_v1.py --apply
```

The apply mode backs up the notebook, `PROJECT_LEARNING_LOG.md`, and `CHECKPOINT.md` before appending an idempotent checkpoint. It preserves existing notebook code cell source and output objects. Do not use `--apply` if those files have uncommitted edits you have not reviewed.

## Commit only authoritative sources

```bash
git add scripts/build_real_acoustic_results_v1.py \
  scripts/verify_real_acoustic_results_v1.py \
  scripts/append_real_acoustic_checkpoint_v1.py \
  research_checkpoints/REAL_ACOUSTIC_RESULTS_RUN_GUIDE.md
# If the documentation updater was applied:
git add notebooks/01_south_african_acoustic_eda.ipynb PROJECT_LEARNING_LOG.md CHECKPOINT.md
git diff --cached --check
git diff --cached --stat
git commit -m "Document descriptive South African real-acoustic evidence"
```

The generated report may remain ignored under `reports/`. If a versioned final report copy is needed, copy only the reviewed, source-traceable Markdown appendix into `research_checkpoints/` in a separate commit. Do not commit raw WAVs, extracted contest data, local path secrets, or an unverified declaration.

## Evidence boundary

- 460 run entries across six stations, if the actual inputs match prior audited summaries; Durban has only one sampled train run.
- Bootstrap CIs are conditional on *sampled background run* identities and do not capture deployment-level clustering, environmental dependence, or site population representativeness.
- The printed archive levels do not carry independently verified underwater pressure calibration; relative spectrum is independently peak referenced.
- Station-relative deviations **do not** identify natural vessel events.
- No use of Zindi hidden test, Experiment 004 frozen evaluation, original validation or sealed station holdout.
- **Never** substitute the 10 unverified site coordinates or the two unresolved Saldanha candidate positions for confirmed hydrophone deployments.
- The Phase 2 v2 `14/14` inventory means file presence and certain structural checks only, not final evidence clearance.
