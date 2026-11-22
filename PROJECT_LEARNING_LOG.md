# Underwater Vessel Noise Challenge — What We Have Learned So Far

**Project:** French Embassy Underwater Vessel Noise Challenge (Zindi)  
**Updated:** 22 November 2026  
**Status:** Data checks and source splits are ready. We have tested one audio feature extractor. We have **not** trained or evaluated a prediction model yet.

## Why we wrote this document

This is our project memory. It explains, in simple language, what we did, what we found, why we made each decision, and what we need to do next. Someone joining the project should be able to read this without knowing all our past conversations.

We want to build a strong and honest competition solution. A good score is important, but the score must come from a method we can explain, repeat, and trust. We also want to avoid using the hidden test data in the wrong way.

## 1. What are we trying to predict?

The challenge gives us underwater sound measurements and asks us to estimate how much of the sound energy was added by a vessel. The main prediction is called `share_added_vessel`. The submission also includes `score_added_vessel`. The main ranking measure we have been working toward is based on root mean squared error (RMSE). **Before submitting, we must check the latest official challenge page and starter files to confirm the exact meaning, valid range, and format of both targets.**

The competition test data has **1,500 examples**. Each example has **375 sound measurements**: 15 minutes of data across 25 frequency bands. The frequency bands run from about 31.6 Hz to 7,943.3 Hz.

The hidden test examples are for prediction only. We do not use their values to fit a model, create labels, choose a scale, tune a threshold, or otherwise learn from them. We will confirm all current challenge rules before making a final submission.

### A few words we use

- **Frequency band:** A small range of sound frequencies. Low bands often contain deeper sounds; high bands contain higher-pitched sounds.
- **Acoustic energy:** A measure of sound power. It is not the same as a sound level written in decibels.
- **Decibel (dB):** A way of expressing a ratio on a logarithmic scale. We must convert compatible levels to linear energy before adding them.
- **Training:** The examples a model learns from.
- **Validation:** Separate examples used to check whether what the model learned works on data it did not see during training.
- **Leakage:** When information from validation or test examples accidentally helps the model during training. Leakage can make a model look better than it really is.
- **Holdout:** Data kept apart for a later, harder test.
- **Manifest:** A file that records which source examples belong to which group.

## 2. What data do we have?

### A. Zindi competition data

The challenge test file contains 1,500 rows with 25 frequency bands measured over 15 one-minute positions. That makes 375 acoustic features per row. We have not used these hidden-test measurements to train or tune anything.

### B. South African underwater sound archive

We checked acoustic data from 10 monitoring stations. Together they contain **1,715,779 one-minute records**. The original files have 62 `TOL_` frequency-band columns. All 25 bands needed for this competition were present and had finite values in the files we inspected.

We checked basic data quality: whether the expected 25 bands were present and finite, whether minute durations were near one minute, and whether the spectrum had enough variation. Some records did not meet every quality rule, so we did not blindly treat all records as equally suitable. We have not yet confirmed the full calibration history or exact acoustic measurement convention of these files.

We built an index of **88,143 non-overlapping, fully observed 15-minute windows** across eight stations. Two other stations, Cape Recife and Riy Banks, did not provide 15 consecutive observed minutes within the runs we found, so we did not force incomplete data into the index.

We also tested retrieval of 16 indexed windows (two per included station). All 16 gave the expected 15 rows and 25 usable frequency-band readings. This is a sample check, not a claim that we manually inspected all 88,143 windows.

The index is at `reports/eda/background_window_index.csv`. Its entries point back to the original Parquet data instead of making large duplicate audio files.

### C. VTUAD vessel recordings

We downloaded and extracted the VTUAD strict-interval WAV audio and its supporting metadata.

- **525 WAV files** in total.
- **509 files** stored in the `vessel/` archive folder.
- **16 files** stored in the `background/` archive folder.
- Every WAV is **mono**, sampled at **32,000 samples per second**, and stored as **16-bit PCM**.
- Vessel-folder files have a median duration of about **3 minutes**; background-folder files have a median duration of about **33 minutes**.
- All **525 files passed SHA-256 checks** against the publisher's per-file checksum list. No files were missing from the metadata-to-WAV mapping.

These checks tell us that the local copies match the publisher's files. They do **not** prove that every recording is correctly labelled, free of background noise, or calibrated as a pressure measurement.

The VTUAD dataset card describes a 525-recording collection rebuilt from Ocean Networks Canada hydrophone and AIS data from 2016–2017. Its licence is listed as **CC BY 4.0**. We should give proper credit when using it and keep a record of any external data included in our solution. The copy we inspected describes itself as **v1.3.0 candidate**.

The labels were made from AIS vessel tracking rules, not by isolating a vessel's sound from everything else. AIS can miss vessels. This is why we call the recordings **candidate vessel-source audio**, not pure vessel-only sound.

### D. HearMyShip

We also found a HearMyShip inventory of **1,149 records** and tested one download. We have **not** finished checking its audio quality, identity information, licence or suitability. For now, it remains a possible independent source for later tests. We are not mixing it into training yet.

## 3. How we chose the South African background windows

The challenge examples cover 15 minutes, so we wanted our real background data to have the same time length and 25-band shape.

We first built an index of 15-minute windows with continuous observations. We did not fill missing minutes with made-up values just to obtain more examples. We also chose non-overlapping windows so the same minute would not appear in many adjacent examples.

### Why we did not use a random split

Underwater sound often changes slowly. Two nearby recordings can sound almost identical because they came from the same place, recorder or conditions. If we randomly placed nearby windows on both sides of our train/validation split, the validation score could be misleading.

Instead, we used **time-based separation**. We also kept **Fish Hoek** and **Sea Point** out of ordinary model development so that we can later test whether a model works at stations it has not trained on.

### The final background split (version 2)

| Group | 15-minute windows | What it is for |
|---|---:|---|
| Training | 67,463 | Learn patterns from background data |
| Validation | 16,946 | Check performance on later background periods |
| Excluded | 3,193 | Preserve time gaps and whole-run separation |
| Station holdout | 541 | Later test on Fish Hoek and Sea Point |
| **Total** | **88,143** | Every indexed window is accounted for |

The file is `reports/eda/background_split_manifest_v2.csv`.

We originally had a small split problem: four continuous recording runs crossed a boundary between an active group and the excluded group. We investigated the exact runs rather than ignoring the failed check. We changed the assignments, including putting Durban's later complete recording run into validation. After that:

- Every indexed continuous recording run belongs to just one split group.
- All six development stations have training and validation windows.
- The train-to-validation time gaps are **9 to about 49 days**, depending on the station. Every checked station exceeded our seven-day minimum.
- The two station holdouts remained separate.
- The total number of windows stayed at 88,143.

**Important limit:** This is a strong first split, but separate recording runs can still share a recorder deployment or environmental conditions. We must not describe the validation data as perfectly independent.

The first version was preserved. We should turn the final background split logic into a version-controlled script so that it can be rebuilt automatically.

## 4. What we learned from the VTUAD metadata

We inspected the dataset card and these tables:

- `metadata.csv`: recording IDs, paths, labels, timestamps, MMSI and other information.
- `registry_anonymized.csv`: anonymized `hull_id` values for vessel identity.
- `clip_acoustic_stats.csv`: summary sound measurements and the publisher's original split.
- `manifest_val.csv`: short overlapping segments cut from recordings.

All three clip-level metadata tables have **525 rows and 525 unique clip IDs**. The validation-segment file has **20,479 segments** drawn from **80 source clips**. Some segments overlap. We will not randomly split such segments, because nearly identical audio could appear on both sides.

### Discovery 1: A folder name is not a sound label

There are **38 recordings labelled `background`**, but only **16 WAVs in the `background/` folder**. The other **22 background-labelled recordings are inside the `vessel/` folder**.

So we decided to use the metadata label, not just the folder name, when choosing candidate vessel audio:

- **487 recordings** have a non-background class label and are eligible to be considered vessel-source recordings.
- **38 background-labelled recordings** are held aside as background references.

We are **not** saying that the 487 clips contain vessel sound alone, or that the 38 references contain absolutely no vessel sound. These are practical labels with known uncertainty.

### Discovery 2: File lengths and metadata lengths do not always agree

Of 525 recordings, **353** matched the metadata duration to within 0.01 seconds. **168** differed by more than one second; **167** differed by more than 60 seconds. The largest observed difference was **300 seconds**.

Some timestamp ranges also do not agree with reported durations. The data card mentions strict-interval re-cutting, but we have not proven the reason for each mismatch.

**Decision:** For audio processing, we use the WAV header and actual decoded samples to determine available audio length. We keep metadata times for research and grouping, but do not use them to decide how many audio samples exist.

### Discovery 3: The publisher's split can repeat the same vessel

The publisher's recording-level split contains **369 train, 80 validation and 76 test clips**. The metadata audit found **57 hull IDs** and **58 MMSIs** present in more than one of those groups.

That may be acceptable for the publisher's own recording-level classification goal. It is not suitable for our intended claim that validation vessels are unseen.

**Decision:** Create our own split based on `hull_id`, so all eligible clips from the same known hull stay on one side. We will still need to check recording dates and related sessions for further overlap.

## 5. The new vessel-source split (version 1)

We created `scripts/build_vtuad_source_splits.py` and ran it successfully. We used a fixed random seed of **42** and split by **whole hull identities**, not individual WAV files.

| Group | Clips | Distinct hulls |
|---|---:|---:|
| Training | 407 | 137 |
| Validation | 80 | 35 |
| Background reference | 38 | Not part of the vessel-source split |
| **Total** | **525** | |

**Known hulls shared between train and validation: zero.** There were no eligible recordings reported under `unknown_identity` in this run. The script passed its checks.

The result is `reports/eda/vtuad_source_split_v1.csv`. The script was committed locally to Git in commit **`ae2ecc3`** (`Add hull-isolated VTUAD vessel source splitting`). The report CSV is a local generated file and is not tracked by Git under the project's current ignore rules.

### Important warning: The split is not balanced across vessel types

We should not ignore this just because the split passed its identity checks. In validation, there are no clips from **dredger, other, passenger ship, pilot vessel, rescue or sailing**. In training, there are **no dredger clips**, even though validation has eight. Other types, including tanker, have only a few examples.

This means version 1 is useful for a first **unseen-hull test**, but it is **not a balanced test of all vessel classes**. A low or high validation error could partly reflect class differences rather than only generalization to new hulls.

**Decision for now:** Preserve v1 as a reproducible baseline. Before treating model results as reliable, design and compare a class-aware, hull-grouped alternative and report each class's coverage. Never split a hull merely to balance the class counts. Rare classes may need a different test plan or may not be measurable on their own.

## 6. What we tested with actual sound processing

We used Python, SoundFile and SciPy to read a real VTUAD WAV file and estimate sound power in the competition's **25 third-octave frequency bands**.

For `vessel/0.wav`, we successfully extracted **5 complete one-minute blocks**, each containing 25 finite, nonnegative band-power values. The measured sum of those 25 bands varied across the five minutes.

We also checked the calculation against a **known 1,000 Hz sine wave** with amplitude 0.1. Its expected mean-square power is 0.005. The calculated dominant band was 1,000 Hz, and the measured band power was essentially 0.005. Relative error was about **1.23 × 10⁻¹⁴**.

**What this proves:** Our first implementation can read a real recording, produce 25 band values and recover the power of a simple test tone.

**What this does not prove:** It does not prove correct results for all broadband sounds, frequencies at band edges, every WAV file, or the exact processing used to produce the South African sound levels. It also does not establish pressure calibration.

A further practical lesson: after restarting a terminal, we must activate our project's Python environment with `source .venv/bin/activate`. Otherwise, `python` or installed packages may not be found.

## 7. Why we cannot directly mix the two datasets yet

The South African archive contains **acoustic levels**, while the WAV extractor currently produces **relative digital signal power**. These are not automatically the same physical units.

Also, VTUAD recordings include ambient ocean noise. If we simply call the whole recording “added vessel noise,” our synthetic labels may be wrong. We need a clear model of what is being added and what is already in a recording.

We will not add numbers written in decibels as though they were ordinary energy numbers. We first need a justified conversion to compatible linear-energy units, then a carefully stated mixing rule. If we cannot verify an absolute physical calibration, we must clearly call the experiment a **relative-energy simulation** instead of claiming it recreates real, calibrated underwater noise.

Many vessel recordings are only two or three minutes long. We must not repeatedly loop one short clip to fill 15 minutes without explaining and testing the unrealistic patterns that creates. A realistic experiment may instead add a shorter event into part of a 15-minute window.

We also need to check the competition's exact definition of `share_added_vessel` and `score_added_vessel` before producing target labels.

## 8. How we are protecting the competition experiment

We have made these choices to keep our results believable:

1. **Keep hidden test data for inference.** Do not use it to train, tune or generate labels.
2. **Use real 15-minute background windows.** Do not invent missing minutes.
3. **Keep nearby background recordings apart in time.** Use our version 2 chronological split and preserve the station holdouts.
4. **Keep known vessel identities apart.** Use a hull-grouped source split instead of randomly splitting overlapping clips.
5. **Do not confuse filenames and labels.** The `vessel/` archive contains 22 recordings labelled background.
6. **Use actual WAV durations.** Some metadata durations do not match.
7. **Check file integrity.** All 525 VTUAD WAV files matched their checksum list.
8. **Call assumptions what they are.** A successful synthetic tone test does not establish acoustic calibration.
9. **Keep source credit and experiment records.** External data, licences, processing settings, random seeds and final code should be documented.
10. **Judge model quality on separate data.** We need overall RMSE, per-station results, source-group checks and class-coverage warnings, not just one attractive number.

## 9. Current project files

These are the paths that have been established during our work. Paths under `reports/` are local generated outputs and are currently excluded from Git.

| Path | Purpose |
|---|---|
| `scripts/audit_underwater_challenge.py` | Earlier dataset audit |
| `scripts/build_background_window_index.py` | Build a list of complete 15-minute background windows |
| `scripts/validate_background_windows.py` | Test whether indexed windows can be retrieved correctly |
| `notebooks/01_south_african_acoustic_eda.ipynb` | Explore South African sound data |
| `reports/eda/background_window_index.csv` | Indexed 15-minute background windows |
| `reports/eda/background_split_manifest.csv` | Original background split; kept for history |
| `reports/eda/background_split_manifest_v2.csv` | Corrected and validated background split |
| `reports/eda/vtuad_metadata_reconciliation.csv` | Match 525 metadata records to 525 WAV files |
| `scripts/build_vtuad_source_splits.py` | Make source splits by hull identity |
| `reports/eda/vtuad_source_split_v1.csv` | First hull-separated source manifest |
| `data/external/vtuad/metadata/dataset_card.md` | Publisher information and limitations |
| `data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt` | Published WAV checksums |

The first 25-band audio extractor and synthetic tone test have so far been run as terminal experiments. **They are not yet saved as a reusable tested project module.**

## 10. What has been completed and what is still open?

| Work | Status |
|---|---|
| Explore South African acoustic archive | Completed initial audit |
| Index complete 15-minute background windows | Completed |
| Retrieve and check example windows | Passed sample check |
| Create background split with time gaps | Version 2 passed |
| Check VTUAD audio headers | Completed |
| Verify VTUAD WAV SHA-256 checksums | All 525 passed |
| Reconcile metadata to WAV files | All 525 mapped |
| Inspect source identity overlap | Completed initial audit |
| Create hull-separated vessel split | Version 1 passed |
| Test real WAV to 25 bands | One file passed |
| Test 1 kHz tone | Passed |
| Check class balance in vessel split | Issue identified; needs improvement |
| Test broadband noise and band edges | Not done |
| Confirm acoustic calibration and mixing rule | Not done |
| Confirm exact target formula and final rules | Not yet rechecked |
| Build reusable vessel feature extractor | Not done |
| Build controlled synthetic training examples | Not done |
| Train and evaluate a baseline model | Not done |
| Produce a Zindi submission | Not done |

## 11. What we should do next, in order

### Next: make the vessel split more useful

Review the current split's missing vessel classes. Create a **class-aware, hull-grouped candidate split** without sharing a hull across train and validation. Compare it against version 1 rather than overwriting version 1. Record how many hulls and clips of each type are in each group. Some rare classes may be impossible to represent well on both sides; report that honestly.

### Then: turn audio tests into code

Write a reusable 25-band extraction script with automatic tests for a known tone, broadband noise, frequencies near the edges of bands and conservation of power over the frequency range being measured. Use the exact band order required by the challenge. Test small samples first. Do not process hundreds of WAV files until the tests pass.

### Then: define a defensible synthetic experiment

Read the official competition target definition carefully. Decide how to represent vessel-added energy when the vessel WAV itself contains ambient noise. Make all assumptions visible. Keep synthetic sources from the validation hulls separate from training sources, and background validation windows separate from training windows.

### Then: train the simplest useful model first

Start with an understandable baseline, compute RMSE on the synthetic validation set, and break down errors by station and vessel type where there is enough data. Only then try stronger models. Keep the station holdouts for a final stress test, not for endless tuning.

### Before any submission

Check the current rules, external-data requirements, target formula, submission column names and allowed methods. Save the configuration, random seeds, software versions, code commit and source acknowledgements so the result can be reproduced.

## 12. Things we must remember

- We have **good data preparation**, not a winning model yet.
- A passing program check tells us the program did what we asked. It does not always tell us our scientific assumptions are right.
- Source independence is more important than an easy random split.
- The current vessel split is hull-separated but **not class-balanced**.
- A background label does not guarantee silence or pure background.
- A vessel label does not mean the waveform contains only vessel sound.
- Correct checksums do not guarantee correct labels or acoustic calibration.
- We must keep the original sources and our generated reports distinct.
- We should keep this file updated whenever we make a major decision, discover a problem, or finish a new experiment.

---

**Short project status:** We have prepared and checked the background windows, verified all VTUAD audio files, found important metadata limitations, created a time-separated background split and a hull-separated vessel split, and confirmed that a basic 25-band extractor works for one real file and a known test tone. Our next job is to improve the source split and build the tested extraction and synthetic-mixture pipeline. We have **not yet trained or scored a prediction model**.
