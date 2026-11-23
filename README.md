# Underwater Vessel Noise Challenge

Reproducible acoustic analysis and source-disjoint synthetic modelling for the Zindi **French Embassy Underwater Vessel Noise Challenge**.

## Competition Tracks & Rules

- **Leaderboard track:** Predict `share_added_vessel` and `score_added_vessel` in `[0, 1]` for 1,500 `Test.csv` windows (15 min × 25 third-octave bands, 31.6–7943.3 Hz). Ranking uses `1 - RMSE` on `share_added_vessel`; `score_added_vessel` AUC is diagnostic only. `Test.csv` is strictly inference-only (no fitting, calibration, or pseudo-labelling).
- **Research write-up track:** PDF (4–6 main pages preferred, ≤10 pages, <10 MB) plus separate runnable code (<10 MB). Deadline: **13 December 2026, 11:30 UTC**.
- **External data:** Public corpora allowed under licence if disclosed on the Zindi forum with source links, version, and access steps.

## Repository Layout

- `data/` — competition archive (`data/zindi/`), local VTUAD `v1.3.0` (`data/external/vtuad/`), HearMyShip sample (`data/external/hearmyship/`).
- `scripts/` — audit, split, feature extraction, mixture simulation, baseline, and deliverable build scripts.
- `notebooks/01_south_african_acoustic_eda.ipynb` — executed visual EDA and experiment summary.
- `deliverables/research_submission/` — canonical manuscript (`manuscript.md`, `research_manuscript.pdf`), figures, code archive (`research_code_v6.zip`), and verification JSON.
- `docs/` — external-data disclosure draft, organizer clarification note, submission register, learning log, and frozen protocols.
- `reports/` — generated EDA manifests, baseline outputs, and scientific audits (git-ignored).

## Quickstart

Use `.venv/bin/python` (set `MPLCONFIGDIR=/tmp/mpl` for matplotlib):

```bash
# 1. Verify submission deliverables & run clean-copy tests
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/build_submission_deliverables.py

# 2. Reproduce Experiment 007 decision-gate summaries (read-only)
.venv/bin/python scripts/audit_experiment007_decision_gate_v1.py \
  --input reports/baselines/experiment007_source_disjoint_v1 \
  --output /tmp/exp007_gate_check

# 3. Rebuild the site-context figure from authorized organizer ZIP
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/build_site_context_figure.py \
  --archive data/zindi/raw/South_African_Real_Acoustics.zip \
  --summary deliverables/research_submission/training_only_site_aggregates.csv \
  --output deliverables/research_submission/figures/site_context_candidate_locations.png
```
