# Research evidence register — release and location reconciliation

**Checked:** 22 November 2026. **Status:** source/release investigation, not signed-off scientific provenance.

## VTUAD published archive and local version

Official current record: https://zenodo.org/records/23037895

The publisher's v1.3.2 record describes the audio, registry, feature library and frozen split as byte-identical to v1.3.0/v1.3.1, and records added supplementary audit material, including wrap-around splice information. The upstream source is Ocean Networks Canada; the corpus is AIS-auto-labelled and does not isolate pure vessel-only sound.

Our previous local research described `v1.3.0 candidate` and 525 SHA-256-checked WAVs. This does **not** establish a verified version label. We need the actual downloaded asset filenames and hashes, original local dataset-card version, local extract structure, and reproducible band-power preprocessing record. If the original WAV bytes and associated core files match a specified release, document precisely which artifacts were used and whether supplementary v1.3.2 diagnostics were or were not included.

**Do not relabel local data v1.3.2 solely because it appears on a public discussion board.** If file evidence supports only the common byte-identical core, write that limited conclusion instead.

## External-data sharing rule

Zindi challenge: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge

Research clarification: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36266

The public external-data disclosure should identify the exact used dataset, version/release uncertainty if any, authors, license, publicly accessible URL, download and processing instructions, and no use of hidden test data for fitting or tuning. Record the *actual posted* discussion URL and date after publication. A local draft is not equivalent to publication.

## South African acoustic station coordinates

Local candidate positions: `data/zindi/extracted/stations.csv`; dictionary: `data/zindi/extracted/DATA_DICTIONARY.md`. Ten sites have candidate positions; Saldanha Bay has two competing positions. Neither the candidate array nor a coast-name geocode is a verified instrument location.

Potential scientific context: https://www.mdpi.com/2077-1312/10/6/746 describes some South African acoustic monitoring deployments and instrumentation. It must not be used to assert identity with the exported files without instrument/deployment, site, timestamp and acquisition-system matching.

**Evidence threshold for verified position:** authoritative source with hydrophone/recorder identity, deployed date range, coordinates with CRS, site identifier crosswalk, precision/uncertainty, and clear interpretation of point location. A report figure with station names alone is not enough.

If this cannot be established, the responsible output is a **station-indexed non-geographic comparison** of recorded acoustic statistics with sampling coverage and uncertainty—not a vessel-traffic heatmap or true-pressure map.

## Scientific claims that remain disallowed without additional evidence

- Absolute calibrated underwater sound pressure from the cross-dataset relative mixing procedure.
- Natural vessel activity attribution from unlabeled acoustic changes.
- Spatial interpolation of vessel activity from approximate site coordinates.
- Generalization claims about Zindi hidden/private test from internal synthetic assessments.
- Independent final validation on source groups that have previously been explored or retuned against.

## Minimal reproducibility citation set

1. Zindi challenge URL and the exact accessed rule version/date.
2. VTUAD Zenodo record and authors, local asset filenames and hashes, and actual used licence terms.
3. Ocean Networks Canada provenance through the VTUAD dataset record.
4. South African archive provenance and instrument/export metadata if available.
5. Frozen synthetic protocol and assessment results, with source separation and bootstrap limitations.
6. Research code commit hashes, environment/dependencies, input manifests, and figure-generating commands.
