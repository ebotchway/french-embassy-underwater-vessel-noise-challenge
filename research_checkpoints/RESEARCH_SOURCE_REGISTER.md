# Research source register — checkpoint 22 November 2026

| Source | Locator | Status and scientific scope |
|---|---|---|
| Zindi official competition | https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge | External competition; recheck full rules shortly before submission |
| Zindi data page | https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data | Available starter and write-up guidance; verify licensed inputs and output format |
| Zindi research clarification | https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36266 | Research PDF and separate runnable code by 2026-12-13 11:30 UTC, as observed 2026-11-22 |
| VTUAD Zenodo archive | https://zenodo.org/records/23037895 | Public versioned record; exact local dataset release still must be reconciled with downloaded manifests and checksums |
| Ocean Networks Canada | https://data.oceannetworks.ca/ | Upstream source for VTUAD; check source attribution and licence |
| Eastern Cape soundscape research | https://www.mdpi.com/2077-1312/10/6/746 | Scientific context; published deployment locations may differ from this project's exported stations/years; no assumed equivalence |
| Local South African archive candidates | `data/zindi/extracted/stations.csv` | Ten working site-coordinate leads, **not** verified hydrophone positions; Saldanha Bay contains two alternatives |
| Local provenance dictionary | `data/zindi/extracted/DATA_DICTIONARY.md` | Explicitly warns about export-derived recorder IDs and unverified location assignments |
| VTUAD local card | `data/external/vtuad/metadata/dataset_card.md` | Version described in prior local inspection as **v1.3.0 candidate**, not certified published v1.3.2 |
| Existing manuscript | `reports/research_manuscript_v1/SCIENTIFIC_MANUSCRIPT_DRAFT.md` | Synthetic/real descriptive evidence; only a draft |
| Experiment 007 assessment | `reports/baselines/experiment007_source_disjoint_v1/` | 120 internal assessment pairs; synthetic labels; source-disjoint within experiment |
| Experiment 007 decision gate | `reports/audits/experiment007_decision_gate_v1/` | Verified outcome and source deletion; dependence limitations remain |

## Release-identity checklist

1. Get the actual local dataset card and any download timestamp/source file listings.
2. Compare public release asset names, archive hashes (if published) and metadata versions.
3. Compare clip inventory (525), label mapping, checksums, preprocessing and release status. An equal clip count is **not** sufficient to assert identical versions.
4. Describe the actual *used* release accurately and credit upstream ONC contributors.
5. Publish externally required declaration, retain public post URL/date and include it in reproducibility appendix. A draft file is not a public declaration.

## Hydrophone spatial evidence register

Only promote coordinates to verified sensor/deployment coordinates when all are documented: station/export-ID match, sensor identity, recording/deployment dates, longitude/latitude with declared reference system, authoritative source URL/doc/page, instrument location meaning, and uncertainty. Do not equate community location names with sensor coordinates.

**Current status:** no verified match for all monitored deployments; maps requiring exact hydrophone positions or vessel attribution remain blocked.
