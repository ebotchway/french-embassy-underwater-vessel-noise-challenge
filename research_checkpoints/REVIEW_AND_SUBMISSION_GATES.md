# Phase 5 scientific and submission-review gates

## Evidence already validated in your Mac logs

- 14/14 named research inventory paths present under corrected Phase 2 v2 audit.
- Six station summaries, 460 sampled indexed run rows, 25 spectral bands.
- Five real-acoustic descriptive figures and PDF integrity check passed.
- 459 defined robust z-scores; one missing at single-run Durban; no imputation.
- Experiment 007 frozen internal synthetic 120 assessment pairs; regression RMSE original 0.114236 and varied/hybrid 0.094566; detection AUC original/hybrid 0.848328 and varied 0.886421.
- Original vs varied fixed-0.5 detection FPR 16.67% versus 30.00%; varied calibration-selected FPR 23.33% on assessment, sensitivity 85.29%.
- Bootstrap interval for fixed-threshold RMSE reduction includes 0 (−0.00041, +0.04230).
- These findings are **not** independently validated natural-vessel classifications.

## Pending evidence with no permission to invent

| Gate | Evidence required | Conservative fallback |
|---|---|---|
| Exact acquired VTUAD release | Original archive acquisition receipt, publisher asset checksum and version metadata | State locally processed strict-interval 525 WAV inventory; exact distribution unverified |
| Public external-data declaration | Actual public Zindi discussion URL with verified source/version/licence | Do not assert posted; treat as task to complete |
| Hydrophone georeferenced map | Deployed sensor IDs, accurate recording periods, authoritative coordinates and CRS | Site-level, **non-geographic** descriptive figures only |
| Vessel attribution | Independently labelled events, dated AIS/port observations or validated reference | Acoustic excursions cannot be assigned to vessels |
| Physical pressure map | Instrument sensitivity, reference pressure, processing calibration and unit compatibility | Archive-reported numerical levels and relative spectral shapes only |
| Submission readiness | Full references, clean PDF, runnable code, source attribution, required form, file size limits | Working research draft, no submission approval |

## Manuscript review

- Ensure abstract states internal **synthetic** assessment prominently.
- Preserve the distinction between `share_added_vessel` estimation and `score_added_vessel` detection.
- Include method of relative energy addition, missingness masks, source disjointness and all core limitations.
- Use two decimal places for station level presentation; never round underlying data.
- Keep the source-level uncertainty warning: 460 indexed runs do not guarantee 460 independent deployments.
- Distinguish positive-class prevalence in simulation from natural vessel prevalence.
- Check that figures only depict what their underlying CSV measurements support.
- Verify citations and all technical, reproducibility and licensing references.
- Keep document classified `REVIEW DRAFT`, never automatically `SUBMISSION READY`.

## Submission operations

The official research submission guidance reports PDF and accompanying runnable code/notebook as separate files due 13 December 2026 at 11:30 UTC. Confirm live requirements before uploading. Do not spend competition submissions repeatedly probing leaderboard feedback. No tool here posts or submits anything automatically.
