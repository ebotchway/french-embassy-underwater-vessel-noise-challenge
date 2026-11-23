# Real-acoustic manuscript quality and disclosure gates

Use this alongside the new Markdown/PDF appendix. The purpose is to preserve honest claims and prevent inference inflation.

## Claims supported by the existing training-only reports

- Descriptive archive-reported level summaries for the six sampled development stations, with explicit run counts and recorder count proxies.
- Separately peak-referenced 25-band spectral shape statistics and within-window temporal variability (as defined by the source generator).
- Run-entry-level station-relative deviations, without event attribution.
- Conditional, station-specific uncertainty for the *median of sampled run-window means*, not an absolute SPL comparison.
- Source-disjoint (within the Experiment 007 internal allocation) synthetic prediction comparison, separately labelled as synthetic.

## Claims not supported

- Absolute sound-pressure levels, calibrated inter-station pressure comparisons or acoustic exposure compliance.
- Confirmed hydrophone deployment positions, vessel tracks, vessel activity density, or a geographic vessel noise heatmap.
- Real vessel-event detection precision/recall and Zindi hidden-test generalization.
- Independence across deployments or chronology based on mere run-level separation.

## Before final research submission

1. Match the exact local VTUAD archive release to a publisher asset/checksum record; distinguish v1.3.0 candidate from publisher v1.3.2 supplement statements.
2. Publicly declare licensed external VTUAD source, processing, release and access details; retain a publicly visible discussion link as evidence.
3. Trace every station candidate to its original recorder/hydrophone ID, deployment start/end, exact sensor location, CRS and recording clock.
4. Resolve instrumental calibration and acoustic reference, or restrict the analysis to explicitly uncalibrated relative units.
5. Add independently verified natural vessel evidence for any vessel-attribution or relative-vessel-activity map. If unavailable, report that the required map has **not** been scientifically demonstrated.
6. Confirm official competition submission instructions and current deadline directly before upload.
7. Assemble source citations, code dependency lock, frozen manifests, reproducibility commands, figures and uncertainty interpretation into the final report. Do not claim the final PDF is ready until these gates pass.
