# Phase 6 — Scientific decision and remediation protocol

**Status from available user-supplied Phase 5 console output and manuscript excerpt (22 November 2026): review draft; not submission-cleared.** The attached package has **not** run against the user's local Phase 5 file. The Phase 6 local audit is required for exact document checks.

## High-priority blockers

1. **Vessel-related spatial map.** The challenge explicitly asks for a research pressure/activity map, tied to real South African observations, with uncertainty. The current manuscript says no qualified georeferenced map can be produced because exact sensor coordinates, vessel attribution and physical calibration are not established. This is a scientific deliverable gap, **not a software failure**. Never substitute site-name coordinates as hydrophone positions or relabel total sound as vessel-caused pressure. Search station exporter metadata and authoritative hydrophone deployment records; independently time-align to acoustic recordings. If evidence cannot be established, seek organizer clarification or submit a clearly bounded descriptive proxy with an explicit acknowledgement that it is **not** the requested pressure map. Neither action guarantees award eligibility.
2. **Real-event vessel specificity.** The 007 detector was tested on source-disjoint *synthetic mixtures*, not confirmed naturally occurring vessel passages. Any comparison to real activity needs date-aligned verified AIS/port records or validated real-event annotations; otherwise keep the claim descriptive.
3. **VTUAD release identity and external-data declaration.** A valid ledger of 525 WAV checksums does **not** establish original download provenance or the publisher record used. Seek retained original archive files/receipts, asset checksums and release-specific data card. Disclose public source/version/license and usage on Zindi; keep permalink and verified posted text. Do not claim that a local declaration draft was publicly posted.
4. **Calibrated sound-pressure units.** Do not label archive numeric summaries `dB re 1 µPa`, physical vessel pressure or absolute energy contributions without calibrated instrument chain, conversion reference, and unit reconciliation. Relative VTUAD features are not physically isolated vessel emissions.
5. **Clean-room reproduction and upload package.** Prepare a separate code/notebook artifact (<=10 MB), environment lock or requirements, clear execution commands, input acquisition and licences, and test it in a clean environment. PDFs and code are separate upload items. Exclude raw third-party recordings and sensitive/unreleased challenge material where redistribution rights are unclear.

## Required final manuscript structure

Abstract (explicitly *synthetic* vs *observed*), challenge task and modelling target, measurement sources and usage licence, station/recorder/time-sampling audit, precise split controls, relative-energy simulation equation and missingness, original/varied/hybrid methods, Exp 007 policy outcomes and uncertainty, genuine background false alarm trade-offs, five **descriptive** acoustic figures with denominators/captions, geography/attribution/calibration limitations, reproducible code, references, and a separate verified map section only if admissible evidence exists.

## Numbers not to conflate

- Six station names; **460 sampled run entries** across **25 challenge frequency bands**. They are not verified distinct instrument deployments.
- 459 defined and one undefined robust-z value; the undefined one is at Durban (one run), not a missing original acoustic level.
- Exp 007: source-disjoint internal **synthetic** assessment, 120 pairs, 840 rows; original RMSE **0.114236** versus varied/hybrid RMSE **0.094566**. Original detector AUC **0.848328**, varied **0.886421**.
- At 0.50, varied detector sensitivity **0.876751**, genuine background FPR **0.300000**; at calibration-selected 0.55, **0.852941** and **0.233333**. Original detector at 0.50: **0.739496**, **0.166667**.
- Bootstrap RMSE reduction 95% interval on one fixed-threshold run-cluster comparison **[-0.00041, 0.04230]** crosses zero. Other comparisons/intervals differ; do not select among intervals to justify certainty.
- Publisher statement regarding VTUAD v1.3.2/v1.3.0 asset equivalence must be cited as a **publisher statement**, not proof the local downloaded archive corresponds to either release.

## Decision rules

- **PASS for file existence or hash integrity** means only that the chosen file/hash is internally consistent. It does not certify the scientific conclusions.
- **PASS on synthetic model assessment** does not imply transfer to naturally occurring vessel noise.
- **Unverified coordinate candidates** are excluded from hydrophone-level maps. A station-level table or plot is not a physical vessel-pressure map.
- **Unknown or unmet submission criteria** remain explicitly blocked. Only verified evidence and a manual final review can change them.
- Do not reopen or optimize on the existing Experiment 004 frozen evaluation, Experiment 007 assessment, hidden Zindi test or public leaderboard probes in an attempt to patch the report.

## Reviewer checklist

1. Render and visually inspect every page of the **actual** review PDF. Check truncation, blank pages, readability of small tables, caption-to-figure mapping, legibility in grayscale, numbering, and consistent units.
2. Reconcile the PDF text against Markdown and the input-hash manifest; verify displayed numerical results and uncertainty with source CSVs.
3. Ensure every nontrivial external statement has a proper citation and that links/licenses are exact.
4. Prepare a legally redistributable **separate** runnable code/notebook package and verify under the 10 MB file limit.
5. Independently verify that the external-data disclosure is publicly posted.
6. Recheck Zindi's research upload form and criteria just before submission.
7. Review any spatial/pressure claims against the *original* sensor deployment record, calibrated measurement method and independent event evidence.

The final approval decision is **manual** and must never be inferred from a script's `PASS` message.
