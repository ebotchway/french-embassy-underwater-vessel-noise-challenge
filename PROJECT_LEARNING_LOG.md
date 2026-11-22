# Underwater Vessel Noise Challenge — What We Have Learned So Far

**Project:** French Embassy Underwater Vessel Noise Challenge (Zindi) 

**Updated:** 22 November 2026 

**Status:** Synthetic Experiments 001 and 002 have been completed. The Experiment 002 96-feature Random Forest reached RMSE 0.112061 and MAE 0.073236 on 300 synthetic validation examples, with known source, duration and station sensitivities. No leaderboard score or verified natural-vessel detection accuracy is available. Preparing an initial, format-checked inference submission and a separate research write-up.

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

We checked acoustic data from 10 monitoring stations. Together they contain **1,715,779 one-minute records**. The original files have 62 `TOL\_` frequency-band columns. All 25 bands needed for this competition were present and had finite values in the files we inspected.

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

We should not ignore this just because the split passed its identity checks. In validation, there are no clips from **other, passenger ship, pilot vessel, rescue or sailing**. In training, there are **no dredger clips**, even though validation has eight. Other types, including tanker, have only a few examples.

This means version 1 is useful for a first **unseen-hull test**, but it is **not a balanced test of all vessel classes**. A low or high validation error could partly reflect class differences rather than only generalization to new hulls.

**Decision for now:** Preserve v1 as a reproducible baseline. Before treating model results as reliable, design and compare a class-aware, hull-grouped alternative and report each class's coverage. Never split a hull merely to balance the class counts. Rare classes may need a different test plan or may not be measurable on their own.

## 6. What we tested with actual sound processing

We used Python, SoundFile and SciPy to read a real VTUAD WAV file and estimate sound power in the competition's **25 third-octave frequency bands**.

For `vessel/0.wav`, we successfully extracted **5 complete one-minute blocks**, each containing 25 finite, nonnegative band-power values. The measured sum of those 25 bands varied across the five minutes.

We also checked the calculation against a **known 1,000 Hz sine wave** with amplitude 0.1. Its expected mean-square power is 0.005. The calculated dominant band was 1,000 Hz, and the measured band power was essentially 0.005. Relative error was about **1.23 × 10⁻¹⁴**.

**What this proves:** Our first implementation can read a real recording, produce 25 band values and recover the power of a simple test tone.

**What this does not prove:** It does not prove correct results for all broadband sounds, frequencies at band edges, every WAV file, or the exact processing used to produce the South African sound levels. It also does not establish pressure calibration.

A further practical lesson: after restarting a terminal, we must activate our project's Python environment with `source .venv/bin/activate`. Otherwise, `python` or installed packages may not be found.

## 7. Why mixing the two datasets requires care

The South African archive contains **acoustic levels**, while the WAV extractor currently produces **relative digital signal power**. These are not automatically the same physical units.

Also, VTUAD recordings include ambient ocean noise. If we simply call the whole recording “added vessel noise,” our synthetic labels may be wrong. We need a clear model of what is being added and what is already in a recording.

We will not add numbers written in decibels as though they were ordinary energy numbers. We first need a justified conversion to compatible linear-energy units, then a carefully stated mixing rule. If we cannot verify an absolute physical calibration, we must clearly call the experiment a **relative-energy simulation** instead of claiming it recreates real, calibrated underwater noise.

Many vessel recordings are only two or three minutes long. We must not repeatedly loop one short clip to fill 15 minutes without explaining and testing the unrealistic patterns that creates. A realistic experiment may instead add a shorter event into part of a 15-minute window.

We used a stated energy-fraction formula for our controlled synthetic experiment and verified its arithmetic. Before any final submission, we must confirm the latest official competition rules, both target fields and the submission format.

## 8. How we are protecting the competition experiment

We have made these choices to keep our results believable:

1\. **Keep hidden test data for inference.** Do not use it to train, tune or generate labels.

2\. **Use real 15-minute background windows.** Do not invent missing minutes.

3\. **Keep nearby background recordings apart in time.** Use our version 2 chronological split and preserve the station holdouts.

4\. **Keep known vessel identities apart.** Use a hull-grouped source split instead of randomly splitting overlapping clips.

5\. **Do not confuse filenames and labels.** The `vessel/` archive contains 22 recordings labelled background.

6\. **Use actual WAV durations.** Some metadata durations do not match.

7\. **Check file integrity.** All 525 VTUAD WAV files matched their checksum list.

8\. **Call assumptions what they are.** A successful synthetic tone test does not establish acoustic calibration.

9\. **Keep source credit and experiment records.** External data, licences, processing settings, random seeds and final code should be documented.

10\. **Judge model quality on separate data.** We need overall RMSE, per-station results, source-group checks and class-coverage warnings, not just one attractive number.

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
| `scripts/extract_vtuad_band_features.py` | Turn eligible WAV clips into relative 25-band minute features |
| `reports/eda/vtuad_minute_band_power_v1.csv` | Complete set of 1,892 minute-level features |

| `reports/eda/vtuad_source_split_v1.csv` | First hull-separated source manifest |

| `data/external/vtuad/metadata/dataset_card.md` | Publisher information and limitations |

| `data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt` | Published WAV checksums |

The 25-band extractor is saved as `scripts/extract_vtuad_band_features.py`. It has passed a 1 kHz tone test, a broadband-noise test, a three-clip smoke test and a complete 487-clip extraction. Frequency-boundary tests and physical calibration are still outstanding.

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

| Create hull-separated vessel split | Versions 1 and class-aware 2 passed |

| Test real WAV to 25 bands | 487 files processed; full feature audit passed |

| Test 1 kHz tone | Passed |

| Check class balance in vessel split | Class-aware version 2 covers nine of eleven classes in both groups |

| Test broadband noise | Passed first white-noise coverage test |
| Test band-edge behaviour | Not done |

| Confirm acoustic calibration and mixing rule | Controlled relative-energy simulation validated; physical calibration remains unverified |

| Confirm exact target formula and final rules | Synthetic energy-fraction arithmetic verified; final official rules and diagnostic field still need confirmation |

| Build reusable vessel feature extractor | Script created and full extraction passed; further tests needed |

| Build controlled synthetic training examples | 200 train and 80 validation examples generated; self-test passed |

| Train and evaluate a baseline model | Three baseline models trained; Random Forest RMSE 0.197845 on synthetic validation |

| Produce a Zindi submission | Not done |

### Completed: Full VTUAD acoustic feature extraction

We successfully converted all 487 eligible vessel recordings into 1,892 one-minute acoustic observations. Each observation contains measurements for the same 25 frequency bands used in the Zindi competition.

Before doing this, we tested our calculation with a known 1,000 Hz sound and with broadband noise. Both tests passed. We also tested three real recordings before processing the complete dataset.

We kept the original recording identifiers, vessel identities and training or validation assignments so that we can trace each measurement back to its source.

These measurements represent relative digital sound power, not calibrated underwater sound-pressure levels. We must account for this difference when combining vessel recordings with South African background measurements.

The full feature dataset passed its structural audit. Since then, we created a class-aware source split, tested simulated mixtures, and trained baseline models. Band-edge processing and physical calibration still need investigation.

### Completed: Full feature audit (22 November 2026)

We checked the extracted feature file, `reports/eda/vtuad_minute_band_power_v1.csv`, without loading the WAV recordings again. **All checks passed.**

| Check | Result |
|---|---:|
| Eligible recordings | 487 |
| One-minute records | 1,892 |
| Training records | 1,560 |
| Validation records | 332 |
| Training hulls | 137 |
| Validation hulls | 35 |
| Hulls in both train and validation | 0 |
| Missing or negative band-power values | 0 |
| Zero-power minutes | 0 |
| Very-low-power minutes below 1e-12 | 0 |

We confirmed that each minute has 25 finite, nonnegative band-power readings, that all minute numbers are unique and consecutive within each clip, and that total power agrees with the sum of the 25 bands. Every record keeps the correct training or validation assignment.

The broadband test measured about **0.5568** of total white-noise power inside our band range, compared with an expected **0.5555**. The 1,000 Hz tone test also passed.

**A warning we must keep:** Some vessel types are missing from validation, while all dredger clips are in validation and none are in training. This can make a model look better or worse for reasons unrelated to general acoustic prediction. We should preserve this first split and create a separate class-aware, hull-isolated comparison.

The median band-power sum is about **3.00e-7** in training and **3.23e-7** in validation. There are some much louder minutes in both groups. These values are **relative digital powers**, not calibrated sound levels, so they must not be added directly to the South African decibel values.

**Decision:** The extraction dataset passed the technical tests. We proceeded with explicitly simulated, relative-energy labels rather than claiming absolute underwater pressure calibration.

## 11. What we should do next, in order

### Next: design a stronger second simulation experiment

We already created the class-aware hull split v2 and trained first models. We should now build a new, more varied relative-energy simulation using continuous target fractions, realistic event lengths and insertion positions, better station coverage and independent validation examples. Preserve all older outputs rather than overwriting them.

### Then: turn audio tests into code

The reusable extraction script is already in place and has processed the 487 eligible clips. Next, test frequencies at band boundaries, check our integration assumptions and verify the 25-band order against the official challenge schema. Preserve the existing output as version 1.

### Then: test how well our results generalize

The first controlled simulation already works. Next, assess the new experiment on fresh examples, evaluate errors by source class and station, check the influence of ambient sound in vessel clips and record all scaling assumptions.

### Then: compare models without overusing one validation set

A constant, Ridge regression and Random Forest have already been tested. The combined feature idea is promising but was chosen after looking at the same 80 validation examples. Preselect it for the next experiment, and assess it on new data before making strong claims. Keep Fish Hoek and Sea Point held out for a later stress test.

### Before any submission

Check the current rules, external-data requirements, target formula, submission column names and allowed methods. Save the configuration, random seeds, software versions, code commit and source acknowledgements so the result can be reproduced.

## 12. Things we must remember

- We have **good data preparation and first synthetic model results**, not a proven competition model yet.

- A passing program check tells us the program did what we asked. It does not always tell us our scientific assumptions are right.

- Source independence is more important than an easy random split.

- The class-aware vessel split v2 keeps hulls separate and represents nine vessel classes in both groups; dredger and pilot vessel each have only one known hull.

- A background label does not guarantee silence or pure background.

- A vessel label does not mean the waveform contains only vessel sound.

- Correct checksums do not guarantee correct labels or acoustic calibration.

- We must keep the original sources and our generated reports distinct.

- We should keep this file updated whenever we make a major decision, discover a problem, or finish a new experiment.


## 13. Better vessel splits and complete acoustic features

After discovering that the first hull-based split did not represent every vessel type well, we created a second split using **whole vessel identities within each class**. We never split recordings from the same known hull across training and validation.

The second split has **410 training clips from 137 hulls**, **77 validation clips from 35 different hulls**, and **38 background-labelled reference clips**. Nine vessel classes appear in both training and validation. Dredger and pilot vessel each have only one known hull, so they remain training-only. There are **zero shared hulls** across our two active groups.

We saved the split as `reports/eda/vtuad_source_split_v2.csv`. Its recorded SHA-256 hash was `3dc2a46b59b1d02d09f5d3685cf007088de75c896ee5384bb8ee5e6d4fd8bf24`.

We did **not** need to repeat the original WAV extraction. We joined the new split to the already calculated acoustic features by `clip_uid` without changing any sound measurements. The new feature report has **1,530 training minutes and 362 validation minutes**, making **1,892 minutes from 487 recordings** in total. The file is `reports/eda/vtuad_minute_band_power_split_v2.csv`; its recorded SHA-256 hash was `9971330ba17fbee8f502c2482efe82488e7feb78bc88574f7961faf0647e2401`.

**Why we did this:** We want to check predictions on vessels the model did not encounter during training, and we want wider class coverage. These checks reduce some forms of leakage; they do not guarantee that different recordings never shared an environmental event.

We committed the feature extractor, v2 source split and split-linking script in Git commit `b5b6a99`. We also updated the research notebook in commit `cbc7d11`.

## 14. Our first real-background synthetic-mixture experiment

We wrote `scripts/build_synthetic_mixtures_v1.py` to combine real South African 15-minute background profiles with shorter, scaled vessel-shaped profiles from VTUAD. We used *relative linear energy* for addition, then saved log-level inputs for modelling. These are controlled simulations, not claims that we reconstructed physically calibrated vessel sound.

The simulation uses this simple rule: **mixture energy = background energy + scaled added-template energy**. The simulated target is the added part divided by the total mixture energy, summed over the 15 minutes and 25 frequency bands.

We checked the arithmetic for desired fractions of **0%, 5%, 20%, 50% and 80%**. Every fraction was recovered to numerical precision. We then ran a **24-training / 12-validation smoke test** and a larger run of **200 training / 80 validation examples**. Both passed the generator's self-test and the resulting arrays had shapes `(200, 15, 25)` and `(80, 15, 25)`.

The larger dataset is saved in `reports/eda/synthetic_v1/`, with `train.npz`, `validation.npz`, and matching provenance CSV files. The files contain an input array called `X` and the simulated fraction `y`. **`X` contains logarithmic sound-level values, which may be negative**; this is normal and does not mean negative physical energy.

We checked that the training and validation examples did not share indexed background run IDs, VTUAD clip IDs, or hull IDs. We also kept Fish Hoek and Sea Point separate for a later station test.

**Limit:** The vessel recordings may include ambient noise, and choosing a target fraction to determine the scale can create easier patterns than naturally recorded vessel contributions. We must never confuse our simulated RMSE with the Zindi hidden-test RMSE.

## 15. The first three models and what they achieved

We trained our first models with **200 simulated training examples** and assessed them on **80 simulated validation examples**. The models received only the 15 × 25 acoustic input values, not the true target or vessel identity.

| Model | Validation RMSE | Validation MAE |
|---|---:|---:|
| Constant prediction (training average) | 0.249648 | 0.217644 |
| Ridge regression | 0.266875 | 0.218580 |
| Random Forest, 200 trees | **0.197845** | **0.159288** |

RMSE means *root mean squared error*. Lower is better. The first Random Forest reduced RMSE by about **20.8%** compared with the constant prediction. Ridge did not beat the constant baseline.

The Random Forest still made important mistakes. When the true simulated added share was **0%**, its average prediction was **19.04%**. At a true **70%**, the average prediction was only **43.47%**. In simple terms, it predicted too much vessel contribution when there was none, and too little when vessel contribution was large.

Errors also differed across stations. Richards Bay had 46 of the 80 validation examples, while St Francis had only two. We therefore cannot treat every per-station error as equally reliable.

The first model reports are under `reports/baselines/synthetic_v1/`. At the time of this update, we have **no official leaderboard result** from these models.

## 16. We checked whether the initial improvement was meaningful

We compared the Random Forest and constant predictions on the same 80 simulated examples. The Random Forest improved RMSE by **0.051803**. A paired bootstrap gave a 95% interval from **0.020786 to 0.083729** for the improvement. In 99.94% of bootstrap resamples, the Random Forest was better.

We also shuffled target assignments *within each station* while keeping predictions unchanged. The median shuffled RMSE was **0.290046**, compared with the original Random Forest RMSE of **0.197845**. The permutation p-value was **0.00020**.

**What we learned:** The model's predictions are meaningfully associated with the correct synthetic targets, beyond simple station membership in this test. **What we did not learn:** These tests cannot prove that the model recognizes real vessel sound in unseen conditions. Related observations can also make bootstrap uncertainty look smaller than it truly is.

## 17. We found that simpler acoustic summaries work much better

We next compared different ways of describing the same 15-minute acoustic window. We used the *same 200 training and 80 validation examples*, so this was an exploratory comparison, not a new independent test.

| Feature description | Number of features | Synthetic validation RMSE |
|---|---:|---:|
| Raw minute-by-band input | 375 | 0.196201 |
| Four overall sound statistics | 4 | 0.196340 |
| Average and variation within each frequency band | 50 | 0.147325 |
| Changes between neighbouring minutes | 42 | 0.208288 |
| **Combined global, spectral and temporal summaries** | **96** | **0.119687** |

**Why the combined representation matters:** It includes the overall sound level, how sound is spread across frequency bands, and how it changes with time. Those pieces provide more useful information together than on their own in this experiment.

The combined representation lowered exploratory RMSE by about **39.5%** compared with our earlier Random Forest result of 0.197845. But we tried several alternatives and then selected the best one using the **same 80 validation examples**. Therefore, the 0.119687 score may be optimistic. We must confirm the advantage on fresh, separate examples.

**Decision:** Use the 96-feature combined representation as one *preselected candidate* for Synthetic Experiment 002. Do not claim that it is a proven winning model.

## 18. What the latest generator inspection taught us

Before changing the simulation, we checked the command-line options of `scripts/build_synthetic_mixtures_v1.py`. It currently supports `--train`, `--validation`, `--seed` and `--self-test`. It does not yet expose controls for event-duration distributions, continuous target distributions or station balancing.

We also opened the provenance CSV files. The training report has **200 rows**, and validation has **80 rows**. Each row records:

- The example number, data partition and background monitoring station.
- The background recording run, window start and original Parquet source.
- The vessel clip ID, hull identity and vessel class when an addition was made.
- How many vessel minutes were used, where we started within the clip and where we inserted them into the 15-minute window.
- The requested share, recovered share and relative scale applied to the template.

A background-only example has no vessel ID, a duration of zero and an added share of zero. This is expected, not missing data requiring a replacement value.

The preview included vessel additions lasting **2, 3, 4 and 14 minutes** and different insertion positions. This demonstrates the structure of those sampled examples; it does not establish the complete event-duration distribution. We should calculate that distribution from the full provenance tables before deciding how to redesign it.

**Why we inspected this:** We want Experiment 002 to reuse the known-good parts of our generator, maintain source separation, and change the simulation rules deliberately rather than guessing how the old generator works.

## 19. Historical plan for Synthetic Experiment 002 (now completed)

We have completed the first simulation and baseline analysis. The next step is **not** to claim a competition score. It is to test whether the improvements survive under a more varied setup.

1. Keep the old Experiment 001 output, its reports and its random seed for comparison; never overwrite them with Experiment 002.
2. Inspect the complete generator code and full provenance distributions before changing target fractions, event lengths, insertion positions or station sampling.
3. Generate more varied **continuous target fractions** and carefully chosen zero-addition cases, while keeping the recorded hull and background-run partitions isolated.
4. Compare the preselected combined acoustic summaries, the original raw features and a simple constant baseline on **new validation examples**. Avoid picking the winning feature representation based on the old 80 examples again.
5. Add controls for different background stations, vessel classes and template scaling, and check sensitivity to acoustic background contamination.
6. Use future station holdouts only for a separate, clearly labelled stress test. Confirm the latest competition rules, licensing and both required submission columns before any submission.

**In plain English:** We now have a working training pipeline and a promising new way to describe underwater sound. We still need to establish whether the improvement is robust and whether our simulation resembles the hidden challenge well enough to be useful.

---

**Short project status as of 22 November 2026:** We completed Experiments 001 and 002; our best Experiment 002 synthetic-validation RMSE is 0.112061 on 300 examples. We now have source- and session-level failure diagnostics, an identified per-window normalization assumption, and a controlled plan for the next research audit. We have not measured hidden-test RMSE or validated detection of natural vessel passages.


## 20. Experiment 002: completed study, not a leaderboard result

We created `scripts/build_synthetic_mixtures_v2.py` and `scripts/train_synthetic_baselines_v2.py` without overwriting Experiment 001. These were committed as Git commit `aa3fb5f`. The updated research notebook was committed in `a531103`.

Our new mixture generator formed **1,200 training and 300 validation windows**, each with **15 minutes × 25 frequency bands**. All six development stations provided 200 training examples and 50 validation examples. The generator sampled continuous target-energy fractions, with 233/63 training/validation zero-addition examples, 340/70 weak examples and 627/167 broader additions. The pipeline reported **zero cross-partition overlap** in known background run IDs, vessel clip IDs and hull IDs. This does not guarantee independence of broader environmental conditions, recording deployments or acquisition equipment.

| Predeclared Experiment 002 model | Synthetic validation RMSE | MAE | Signed bias |
|---|---:|---:|---:|
| Constant training-average prediction | 0.250364 | 0.220723 | -0.003617 |
| Random Forest on all 375 measurements | 0.144499 | 0.105283 | -0.021487 |
| Random Forest on 50 spectral summaries | 0.147717 | 0.102357 | -0.005421 |
| **Random Forest on 96 combined summaries** | **0.112061** | **0.073236** | **-0.004937** |

The combined feature family did best, with a 55.2% relative reduction in RMSE compared with the constant baseline **in this synthetic experiment**. It also had the lowest MAE. The earlier 0.119687 result and this new 0.112061 result use different synthetic distributions, so they should not be described as a controlled before-and-after improvement. These experiments do **not** estimate the hidden leaderboard RMSE.

Our 96 features include four whole-window statistics, 50 band-wise spectral statistics and 42 temporal difference summaries. They do not require labelled test measurements for feature construction.

## 21. The original notebook warned us about several problems: what is confirmed?

The first notebook established that source coverage, recording intervals, recorder/deployment configurations and contiguous run lengths differ between South African stations. We built fully observed 15-minute windows and excluded stations that could not supply them. Later experiments made the importance of these distinctions more visible:

| Earlier observation | Later evidence | Strength of conclusion |
|---|---|---|
| Some stations have unusual recording schedules and frequent gaps | St Francis has exactly 15-minute complete runs under the current indexing rules and weak-event RMSE 0.2331 | **Both patterns measured; causation not proven** |
| Available recording windows are unevenly distributed | Experiment 001's 80 validation examples were dominated by Richards Bay; Experiment 002 explicitly balanced station counts | **Confirmed sampling issue, partly addressed** |
| Different stations and recorder configurations may produce different acoustic patterns | Experiment 002's combined-model station RMSE ranges from 0.062544 to 0.142756 | **Confirmed error heterogeneity; equipment explanation untested** |
| Counts of windows may overstate effective independent samples | Durban's 50 validation examples all came from one indexed run | **Confirmed dependence structure; independent-session diversity remains limited** |
| VTUAD publisher's split reused hull identities | Custom v2 hull split has zero known hull overlap across train and validation | **Identified leakage risk and implemented safeguard** |
| Relative WAV power and South African dB levels are not physically calibrated together | Generator deliberately imposes fractional additions | **Still unresolved physically: exact arithmetic is not acoustic calibration** |

**The central lesson:** Some initial data issues are confirmed and handled, but none of the current model-error correlations proves that recording interruptions, hardware, geography or vessel type caused those errors. Our next experiments must change one factor at a time.

## 22. Detailed Experiment 002 failure analysis

The stronger 96-feature model still struggles in certain situations:

| Subset | Validation examples | RMSE | Average signed error |
|---|---:|---:|---:|
| Zero added vessel energy | 63 | 0.036081 | +0.014969 |
| Weak addition | 70 | 0.115271 | +0.057446 |
| Broader addition | 167 | 0.128444 | -0.038596 |
| St Francis, weak additions | 10 | 0.2331 | +0.2111 |
| One-minute events | 82 | 0.1391 | -0.0356 |
| Two-minute events | 78 | 0.1170 | +0.0067 |
| Three-minute events | 47 | 0.0939 | +0.0043 |

At St Francis, the ten weak examples have mean true added share **0.0375** and mean prediction **0.2486**. This warrants investigating background temporal variation, but our current analyses cannot isolate the cause. Some zero-addition predictions also reached **0.220517**, so a low average zero-group RMSE does not eliminate important false positives.

Across all 300 validation examples, **12 of the 30 largest errors** involved one-minute events (82 such examples overall), while **10 of the 30 largest** involved two-minute events (78 overall). Thus 22/30 of the largest errors came from 1–2-minute additions, compared with 160/300 of all examples. Short-duration event detection should be a priority, without attributing every error to duration alone.

We compared how often individual source groups appeared among the worst errors against how often they appeared overall. Durban's validation recording run appeared **7/30 times**, but supplied **50/300** examples, corresponding to **1.40×** proportional enrichment—not 7 independent failed sessions. Bird Island's displayed run appeared **3/30** versus **18/300** overall (**1.67×**). Hulls H118 (5/30 among 32 exposures), H029 (4/30 among 18) and H101 (3/30 among 12) are research leads, not proven faulty sources. One-example groups with 10× enrichment have too little exposure to support conclusions. The top-30 criterion was selected during exploratory diagnosis, so these are descriptive statistics, not formal significance tests.

Vessel classes in the evaluation are unevenly represented; dredger and pilot vessel have no v2 validation hulls. We will not claim measured generalization to these classes.

## 23. Important newly identified issue: per-window normalization

Inspection of the actual generator code uncovered an assumption we must carry into every new training or inference experiment. In `scripts/build_synthetic_mixtures_v1.py`, the background conversion is effectively:

```python
reference = np.max(background_levels_db)
background_relative_energy = 10 ** ((background_levels_db - reference) / 10)
```

The generator then chooses an added relative-energy template and `alpha` to reproduce a requested energy fraction, and stores the logarithm of the resulting mixture. This is mathematically consistent. **But the normalization removes absolute window level from the starting background representation.** It may produce different feature distributions from the original Zindi test measurements, and may erase useful amplitude information or create a simulation-specific cue. The physical measurement reference and cross-dataset calibration remain unverified.

This is not evidence of incorrect code or a competition-rule violation. It is an **unverified modelling assumption**. It must be assessed with a prespecified normalization comparison on training/validation data, not by choosing an option based on hidden-test results. For inference, any per-window transformation must be fixed and applied consistently to training and test. We must never fit means, calibration factors, target scales, thresholds or models on test features.

## 24. Current official Zindi rules and two different submissions

As checked on **22 November 2026** using the challenge pages:

- Official competition: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge
- Data description: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data
- **Leaderboard submission:** A CSV containing exactly the `ID`, `share_added_vessel`, and `score_added_vessel` columns in the sample submission's order. There are 1,500 test IDs. Both outputs must be finite, continuous values from 0 through 1. The first is the controlled vessel-added energy fraction; the second is an evidence score, not a probability that any natural vessel is present. Only `share_added_vessel` affects the ranking score, **1 − RMSE**. A public/private split uses approximately 30%/70% of test records. The stated limit is **two submissions per day**, **200 overall**.
- **Research write-up:** Separate PDF plus a runnable notebook or code, supported by evidence from **real South African acoustic measurements**, an uncertainty-aware comparison with a baseline, and defensible **relative pressure/activity maps** (absolute pressure only with verified calibration). The published write-up deadline is **13 December 2026 at 11:30 UTC**. Check the live page again before delivery.
- External corpora such as VTUAD/ONC are allowed subject to licence and equal-access disclosure. Publish permitted source URLs, versions, processing and download instructions to the challenge community. Do **not** redistribute audio or contest data without permission.
- The test measurements are **for inference only**. Do not fit, fine-tune, calibrate, construct synthetic training labels, pseudo-label, or perform leaderboard probing with them. Report reproducible code, dependencies, data sources, seeds, source splits, inference runtime and model size. Only public/open-source packages are allowed; AutoML is prohibited by the challenge rules.

**Submission decision:** It is reasonable to make one carefully checked **exploratory leaderboard submission**, with an honest warning that synthetic-validation scores cannot predict leaderboard performance. The notebook and this learning log are NOT the leaderboard CSV. We should not describe the existing EDA notebook as a finished research write-up: real acoustic validation, pressure maps, final figures, source attribution, and a clean reproducibility appendix remain incomplete. If we have not checked the fixed inference representation and CSV integrity, wait to upload the leaderboard candidate until those checks pass.

The `score_added_vessel` field may initially use an **uncalibrated monotone evidence proxy**, explicitly labelled as such, rather than claiming a validated probability. A basic first candidate may produce the same numeric value as the predicted added share for both fields, but the concepts have distinct meanings and a later proper evidence model may improve the diagnostic score.

### Missing measurements in the competition's inference inputs

The official data page explicitly warns that blank cells in `Test.csv` are missing observations (`NaN`), **not silence**, and the target uses the same observed minute–band cells. A read-only schema/missingness inspection of the supplied file found **18,350 missing minute–band cells** among **562,500** input cells, affecting **725 of 1,500 windows**. There were no partially missing minutes in that inspected copy: missing values occurred as entire 25-band minute profiles. The first minute was absent in **624 windows**; the last minute in **100 windows**; a few gaps occurred elsewhere. These are input-validity observations, not target information, and were not used for training, calibration or label inference. The actual test must never be used to fit an imputation model or choose a predictor based on its values.

The draft inference script uses **fixed within-window temporal interpolation by frequency band**, extending the nearest observed minute at missing boundaries, then applies the same window-reference shift to synthetic training/validation/test. It also reports **predeclared synthetic-validation masking checks** (first/last/middle/both endpoints missing) to describe sensitivity without learning from the competition test. This simple imputation is an initial pragmatic choice, not a verified reconstruction of the missing underwater sound or a demonstration that the competition target can be recovered precisely when measurements are absent. A mask-aware model is a future research candidate.

## 25. The next reproducible audit and experiments

The next script, `scripts/audit_acoustic_data_model_alignment.py`, reads **only saved Experiment 002 validation predictions, their source provenance, and the background window index**. It does not touch the competition test data. It produces summaries by station, target regime, event duration, source class, background run, clip and vessel hull, plus a JSON explanation of limitations. It should be run from the project root and saved into `reports/eda/data_model_alignment/`.

After auditing, prepare a controlled Experiment 003: (1) matched tests of duration at fixed background/source and target fraction; (2) a normalization/representation ablation using identical transforms at training and inference; (3) grouped-by-session uncertainty, not falsely independent row-level resampling; (4) changed-source spectral or mixing robustness; and (5) a separate plan for genuine real-acoustic evidence, while leaving Fish Hoek and Sea Point as reserved holdouts until evaluation rules are frozen. Do not change all factors together or repeatedly tune against Experiment 002's already-used validation set.

For a first CSV candidate, `scripts/prepare_zindi_first_submission.py` fits **only synthetic Experiment 002 training examples**, checks the held-out synthetic validation for reporting, and predicts the 1,500 test windows after an identical fixed per-window log-level shift. **This is a new representation experiment, not the source of the earlier 0.112061 result.** It runs in dry-run mode by default and writes a candidate only on explicit request. Review its new synthetic-validation score and the submission's schema before any upload; do not select a representation based on leaderboard feedback.

## 26. Publication quality, attribution and honest authorship

Write the research material in direct, natural language: explain what was measured, what we assumed, what we tested, what failed, and what is still unknown. Preserve figures, source provenance, methods and negative results. **No watermarks, decorative branding or hidden identifying overlays are added to these working files.** This is a plain Markdown log and a standard Jupyter notebook, not an assurance about third-party software or Zindi's own processing. Do not claim academic verification, authorship or physical calibration we cannot document. Keep all external-data licence and acknowledgement notices. Remove personal/local directory paths, access secrets, restricted data or source audio before sharing a public notebook. Organizers may require a polished PDF report; a technical notebook alone is not a substitute.

**Current research position:** Our notebooks identified meaningful data constraints; later simulations reproduce meaningful performance variation, but have not shown their causes. We have a reproducible synthetic baseline, a defensible alignment-audit plan, and a path to a carefully verified first leaderboard candidate. The separate real-data research report still needs evidence and maps.


---

## 27. Phase 3 checkpoint — Submission 001, detection, and Experiment 003 (22 November 2026)

### Public submission is provisional, not the final private ranking

The submitted `reports/submissions/first_candidate.csv` passed schema checks and Zindi accepted and scored it. Submission ID: `pBUapzRu`; public leaderboard score: **0.761535762** (corresponding share RMSE ≈ **0.238464238** under `1 − RMSE`); public detection ROC AUC displayed: **0.517533333**; snapshot rank: **565**. Git baseline: `aaf4748`. The public result is **provisional** and must not be represented as the final private outcome. The public evaluation subset and private evaluation subset are different; this is not evidence of a known fixed-score rescaling. Public feedback must not be used to fit hidden-label targets.

The submission copied the predicted share into `score_added_vessel` as an explicitly **uncalibrated, monotone proxy**; it did not train a dedicated classifier. No final private result is known.

### Like-for-like detection audit (commit `ae6180a`)

The exact submitted-model pipeline was recreated using the synthetic Experiment 002 training and validation partitions. It reproduced synthetic share RMSE **0.1664766801**, MAE **0.1175880686**, detection ROC AUC **0.9533855736**, and average precision **0.9851032280**, on **300 synthetic validation examples** (63 zero, 237 positive). Simple simulated missing-minute conditions yielded ROC AUC approximately **0.9375–0.9531**. These results are not directly comparable to hidden public scores as estimates of a single common data population; strong synthetic detection alongside near-chance public diagnostic discrimination is nevertheless a major warning about simulation validity and generalization.

### Controlled matched robustness (Experiment 003; commits `8364ff9`, `8ebc480`, `e4e8a35`)

An initial **12-pair pilot** was followed by the larger **48-pair development study** across six stations. Each pair generated 13 matched acoustic cases and six observation masks. There were **624 cases per mask per model**, **7,488 model predictions total**, but **only 48 matched pairs**, **36 background runs**, and **17 vessel-source clips**. Durban's eight pairs arose from **one** background run. Model A was the `submitted_max_shift` representation; model B was the original `synthetic_original` representation. Both were trained on already-used synthetic development data. Missing-minute target shares were recalculated **over observed time–frequency cells**, allowing an added event completely removed by the mask to have observed share zero.

| Mask | Submitted RMSE | Original RMSE | Original RMSE advantage | Pair-cluster bootstrap 95% interval | Source-deletion reversal? |
| --- | ---: | ---: | ---: | --- | --- |
| Complete | 0.178687 | 0.127155 | +0.051532 | [+0.029706, +0.074096] | No |
| First minute | 0.179602 | 0.164426 | +0.015177 | [−0.024927, +0.054621] | Yes, run/station |
| Last minute | 0.189053 | 0.145173 | +0.043880 | [+0.015664, +0.072238] | No |
| Middle minute | 0.178411 | 0.128725 | +0.049686 | [+0.027462, +0.072489] | No |
| Both edges | 0.189394 | 0.180296 | +0.009098 | [−0.030646, +0.052150] | Yes, run/station/clip |
| Three middle minutes | 0.173775 | 0.135991 | +0.037784 | [+0.006172, +0.066639] | No |

All six *point estimates* favored the original representation in the 48-pair study, but only four pair-bootstrap intervals excluded zero. Deleting any one station, run, or vessel-source clip did not reverse the original representation's advantage for complete, last, middle, or three-middle missingness. For complete observations, the least favorable leave-one-station-out difference remained **+0.029771**. The first-minute and both-edge comparisons were source-sensitive and should not be described as robust. The 12-pair pilot sometimes favored the submitted representation, reinforcing sample-size caution.

**Statistical limitations:** These are retrospective/development comparisons with source reuse and prior validation inspection; bootstrap clusters at matched-pair level do not make repeated recording runs, hulls or clips independent. Intervals are exploratory, not final confidence statements about competition performance. Changes in the observed zero/positive case mix across missingness scenarios also affect aggregate RMSE; lower RMSE under greater missingness does not necessarily mean a better detector.

### Decisions and outstanding responsibilities

1. **Freeze Submission 001 and Experiments 001–003**; do not overwrite predictions, manifests, model assumptions, scripts or previous notebook outputs.
2. Treat per-window maximum shift as an **unproven representation choice**; preserve both representations for a *newly specified*, source-independent Experiment 004. Do not select a leaderboard candidate from this development comparison alone.
3. Make **background-run, vessel-hull/clip and station/session independence** explicit in Experiment 004 design. Pre-register comparisons and include station holdouts; check feasible sample sizes before claiming validation power.
4. Separate observed-cell share regression (RMSE) from controlled-addition detection (ROC AUC/AP). Include true background-only controls, realistic event duration, and whole-minute masks, with physically defensible uncertainty rather than absolute pressure claims.
5. **External-data declaration remains a draft** (`VTUAD_EXTERNAL_DATA_DECLARATION_DRAFT.md`); verify the exact release and licence and post the source/version/access declaration to Zindi. Do not silently replace v1.3.0 candidate inputs with a reportedly newer release.
6. Continue the **separate real South African acoustic research track**, producing coverage-aware relative acoustic activity maps and uncertainty analyses. The existing work has not established absolute physical calibration or ground-truth natural-vessel labels.
7. Preserve the separate research submission deadline and requirements as organizer-dependent and reconfirm on the challenge page before submission.

**Next formal checkpoint:** Experiment 004 protocol and genuinely held-out source evaluation. These Phase 3 results are recorded from terminal outputs supplied by the researcher, not from executions of the full dataset in this document-generation environment.
