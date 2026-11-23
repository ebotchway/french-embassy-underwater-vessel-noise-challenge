# Underwater Vessel Noise Challenge

Reproducible coastal underwater acoustic analysis and calibration-invariant synthetic vessel-noise modeling for the Zindi **French Embassy Underwater Vessel Noise Challenge** (prepared by Emmanuel Kwaku Botchway).

## Competition Tracks & Compliance Rules

- **Leaderboard track:** Predict `share_added_vessel` and `score_added_vessel` in `[0, 1]` for `1,500` `Test.csv` windows (`15 min x 25` base-10 deci-decade bands, `31.6–7943.3 Hz`). Ranking uses `1 - RMSE` on `share_added_vessel`; `score_added_vessel` ROC AUC is diagnostic only. `Test.csv` is strictly inference-only (zero fitting, calibration, normalization tuning, or pseudo-labelling). Zero AutoML.
- **Research write-up track:** PDF (`4–10` pages, `< 10 MB`) plus separate runnable code archive (`< 10 MB`). Deadline: **13 December 2026, 11:30 UTC**.
- **External data & non-redistribution:** VTUAD `v1.3.0` (DOI `10.5281/zenodo.22274549`) and HearMyShip (`EU Horizon Europe 101086340`, Marine Pollution Bulletin `213:117903`) are disclosed publicly in [Discussion #36419](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36419) and `docs/EXTERNAL_DATA_DISCLOSURE.md`. Neither South African competition data nor raw external WAVs are redistributed in git or submission code archives.

## Repository Layout

- `data/` — official extracted competition CSVs (`data/zindi/extracted/`), `Test.csv` & `sample_submission.csv` (`data/zindi/raw/`), aligned 2s deci-decade parquets (`data/processed/`: `vtuad_decidecade_2s_minutes.parquet` `646 KB`, `hearmyship_decidecade_2s_clips.parquet` `48 KB`), and deterministic leak-free synthetic splits (`data/synthetic/modeling_{v2,v3,v4}/`).
- `notebooks/` —
  - [`01_south_african_acoustic_eda.ipynb`](notebooks/01_south_african_acoustic_eda.ipynb): South African exploratory telemetry audit, `Finding -> Choice` log, and Experiments `001–007`.
  - [`02_vessel_noise_modeling.ipynb`](notebooks/02_vessel_noise_modeling.ipynb): Aligned 2s deci-decade **Modeling v2** deterministic 11-model benchmark and Submission #2 generation.
  - [`03_domain_generalization_v3_v4_and_standards.ipynb`](notebooks/03_domain_generalization_v3_v4_and_standards.ipynb): Post-Submission #2 LB diagnosis, **Modeling v3 -> v4** calibration invariance proofs (`h(f)`), **International Acoustic Standards Benchmark** (`ISO 18405:2017`, `JOMOPANS`, `PCEN`, `RPCA`), and Submission #3 verification.
- `scripts/` — audit, extraction, v2/v3/v4 simulation, feature engineering (`205` v4 features / `71` `R_*` $h(f)$-invariant), deterministic training, international standards benchmark (`standardized_evaluation_suite.py`), master verifier (`reproduce_pipeline.py`), notebook builders, and LaTeX manuscript builder (`build_overleaf_manuscript.py`).
- `submissions/` — strictly inference-only `Test.csv` predictions for `candidate_v2_ensemble.csv` (Submitted #2, LB `0.77501743`), `candidate_v3_ensemble.csv` (intermediate), and `candidate_v4_ensemble.csv` (**Recommended Submission #3**).
- `deliverables/overleaf_manuscript/` — compiled 9-page APA 7th LaTeX PDF (`research_manuscript_latex.pdf`), Overleaf upload bundle (`overleaf_research_manuscript.zip`), and `< 10 MB` verified unified code archive (`research_and_modeling_code_final.zip`, `40` scripts + `3` notebooks + docs + TeX).
- `deliverables/research_submission/` — preserved legacy V6 descriptive manuscript (`5` pages) and `research_code_v6.zip` (`17` scripts).
- `docs/` — [LITERATURE_STANDARDS_AND_BOUNDARIES.md](docs/LITERATURE_STANDARDS_AND_BOUNDARIES.md), [SUBMISSION_STATUS.md](docs/SUBMISSION_STATUS.md), [PROJECT_LEARNING_LOG.md](docs/PROJECT_LEARNING_LOG.md), [EXTERNAL_DATA_DISCLOSURE.md](docs/EXTERNAL_DATA_DISCLOSURE.md), [ORGANIZER_MAP_CLARIFICATION.md](docs/ORGANIZER_MAP_CLARIFICATION.md), and experimental protocols.

## Deterministic Reproducibility ("Anytime, Anywhere")

Use `.venv/bin/python` (Python `3.13`, `125` pinned non-AutoML packages in `requirements.txt`). Always export the deterministic single-threaded BLAS/OpenMP variables below before executing PyTorch + GBDT workflows:

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl
```

### Fast Verification & Reproduction (From Included Aligned Parquets + Official Zindi Archive)

```bash
# 1. Verify all 6 deterministic self-tests (v2, v3, v4 sim/feat/train, standards suite), submission schemas, and SHA-256 digests
.venv/bin/python scripts/reproduce_pipeline.py --mode verify-only

# 2. Verify Overleaf LaTeX bundle, 9-page PDF, and clean-copy 40-script code ZIP (<10 MB)
.venv/bin/python scripts/build_overleaf_manuscript.py --verify-only

# 3. Verify preserved V6 research deliverables (requires data/zindi/raw/South_African_Real_Acoustics.zip if re-running V6 ZIP test)
.venv/bin/python scripts/build_submission_deliverables.py --verify-only

# 4. Re-export Submission #3 (candidate_v4_ensemble.csv) from saved bundle_v4.joblib + re-run standards suite
.venv/bin/python scripts/reproduce_pipeline.py --mode export-only --version v4

# 5. Full end-to-end deterministic re-simulation + re-training (from data/processed/*.parquet + data/zindi/extracted/)
.venv/bin/python scripts/reproduce_pipeline.py --mode full --version v4
```

### Full Scratch Regeneration From Raw Upstream Audio (Clean Machine)

To regenerate `data/processed/{vtuad_decidecade_2s_minutes,hearmyship_decidecade_2s_clips}.parquet` on a fresh machine without local raw WAV caches:
1. Obtain official competition data from the Zindi competition page into `data/zindi/raw/South_African_Real_Acoustics.zip`, `Test.csv`, and `sample_submission.csv`, then extract via `.venv/bin/python scripts/setup_underwater_project.py`.
2. Download **VTUAD v1.3.0** (`525` WAVs) from Zenodo (`https://doi.org/10.5281/zenodo.22274549`), verify SHA-256 checksums via `.venv/bin/python scripts/verify_publisher_assets_v2.py`, and fetch the stratified **HearMyShip** 96-clip set listed in `docs/metadata/hearmyship_split_v1.csv` via `.venv/bin/python scripts/download_hearmyship.py`.
3. Extract aligned 2s base-10 deci-decades via `.venv/bin/python scripts/extract_external_decidecade.py`, then run `.venv/bin/python scripts/reproduce_pipeline.py --mode full --version all`.
