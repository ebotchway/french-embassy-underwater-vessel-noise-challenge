# Underwater Vessel Noise Challenge

Reproducible acoustic analysis and source-disjoint synthetic modelling for the Zindi **French Embassy Underwater Vessel Noise Challenge**.

## Competition Tracks & Rules

- **Leaderboard track:** Predict `share_added_vessel` and `score_added_vessel` in `[0, 1]` for `1,500` `Test.csv` windows (`15 min x 25 bands`, `31.6–7943.3 Hz`). Ranking uses `1 - RMSE` on `share_added_vessel`; `score_added_vessel` AUC is diagnostic only. `Test.csv` is strictly inference-only (no fitting, calibration, or pseudo-labelling). Zero AutoML.
- **Research write-up track:** PDF (`4–10` pages, `<10 MB`) plus separate runnable code archive (`<10 MB`). Deadline: **13 December 2026, 11:30 UTC**.
- **External data:** VTUAD `v1.3.0` and HearMyShip are permitted if disclosed on the Zindi forum with source links, versions, and access steps; raw audio is not redistributed.

## Repository Layout

- `data/` — competition archive (`data/zindi/`), local VTUAD `v1.3.0` (`525` WAVs), HearMyShip (`96` WAVs), 2s extracted parquets (`data/processed/`), and leak-free splits (`data/synthetic/modeling_v2/`).
- `notebooks/` — [01_south_african_acoustic_eda.ipynb](notebooks/01_south_african_acoustic_eda.ipynb) (EDA, Finding -> Choice log, Experiments 001–007) and [02_vessel_noise_modeling.ipynb](notebooks/02_vessel_noise_modeling.ipynb) (live deterministic 11-model benchmark & submission sanity).
- `scripts/` — audit, extraction, simulation, feature, deterministic training, and deliverable scripts.
- `deliverables/overleaf_manuscript/` — APA 7th LaTeX manuscript (`research_manuscript_latex.pdf`, `10` pages), Overleaf upload archive (`overleaf_research_manuscript.zip`), and unified verified code archive (`research_and_modeling_code_final.zip`, `31` scripts + `2` notebooks).
- `deliverables/research_submission/` — Legacy V6 descriptive manuscript (`research_manuscript.pdf`, `5` pages) and clean-copy V6 code archive (`research_code_v6.zip`, `17` scripts).
- `submissions/candidate_v2_ensemble.csv` — strictly inference-only `Test.csv` predictions (`1,500` rows).
- `docs/` — external-data disclosure, organizer map clarification, submission register, and protocols.

## Quickstart

Use `.venv/bin/python` (pinned in `requirements.txt`, `126` packages; set `MPLCONFIGDIR=/tmp/mpl` for matplotlib):

```bash
# 1. Verify V6 research deliverables without modifying archives
MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/build_submission_deliverables.py --verify-only

# 2. Verify bit-exact model determinism across all families
.venv/bin/python scripts/modeling_v2_train.py --self-test-only

# 3. Reproduce full deterministic 11-model training + Test.csv inference
.venv/bin/python scripts/modeling_v2_train.py
```
