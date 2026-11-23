# Project Checkpoint

Updated: 23 November 2026. Status: **HOLD** pending Zindi external-data posting, organizer map clarification, and final portal upload.

## Leaderboard Track (Provisional)

- **Submission 001** (`reports/submissions/first_candidate.csv`, ID `pBUapzRu`, commit `aaf4748`): public `1 - RMSE` = `0.761536` (`share_added_vessel` RMSE `0.238464`), diagnostic `score_added_vessel` ROC AUC = `0.517533`.
- No second leaderboard submission has been generated or authorized; `Test.csv` values remain unused for training or calibration.

## Experiment 007 — Frozen Source-Disjoint Assessment

120 paired scenarios (840 cases: 120 non-injected controls, 714 observed positives, 6 masked-to-zero) from `exp004_train` only.

| Policy | Share RMSE | ROC AUC | Synth TPR | Non-Injected FPR |
|:--|--:|--:|--:|--:|
| Original-only, cutoff 0.50 | 0.114236 | 0.848328 | 0.73950 | 0.16667 |
| Contrast-varied, cutoff 0.50 | 0.094566 | 0.886421 | 0.87675 | 0.30000 |
| Contrast-varied, cal cutoff 0.55 | 0.094566 | 0.886421 | 0.85294 | 0.23333 |
| Hybrid (orig det 0.50 + varied reg) | 0.094566 | 0.848328 | 0.73950 | 0.16667 |

- Point RMSE reduction: `0.019670` (~17.2% relative); conditional run-bootstrap 95% CI: `[-0.00041, 0.04230]` (crosses zero).
- Assessment thresholds and predictions are frozen; do not retune against these rows.

## Six-Station Descriptive Summary (Training Runs Only)

460 sampled runs (`120` Richards Bay, `1` Durban, `22` Bird Island, `77` St Croix, `120` St Francis, `120` Saldanha Bay). Source-export dB units (not calibrated dB re 1 µPa).

| Station | Runs | Median Run-Mean | 95% Run-Bootstrap CI | Median Within-Window Var |
|:--|--:|--:|:--|--:|
| Richards Bay | 120 | 63.958 | 62.013–66.346 | 2.518 |
| Durban | 1 | 68.991 | Not estimable | 2.103 |
| Bird Island | 22 | 90.604 | 87.807–91.425 | 2.171 |
| St Croix Island | 77 | 86.420 | 85.249–87.217 | 1.123 |
| St Francis | 120 | 85.875 | 85.101–86.922 | 3.623 |
| Saldanha Bay | 120 | 61.520 | 60.747–62.399 | 1.183 |

## Open Actions Before Submission

1. Post `docs/EXTERNAL_DATA_DISCLOSURE.md` in the Zindi discussion board and record URL + UTC timestamp.
2. Send `docs/ORGANIZER_MAP_CLARIFICATION.md` to organizers regarding Figure 4 (total-sound candidate-site context vs vessel-pressure map).
3. Visually inspect every page of `deliverables/research_submission/research_manuscript.pdf` and upload PDF + `research_code_v6.zip` separately before 13 Dec 2026 11:30 UTC.
