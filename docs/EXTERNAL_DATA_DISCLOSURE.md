# External Data Disclosure (Ready to Post on Zindi)

**Status:** POSTED on Zindi (23 Nov 2026, 10:50 UTC — Discussion #36419).

---

I am using **VTUAD (rebuilt)** — an AIS-auto-labelled vessel-type underwater acoustic dataset reconstructed from Ocean Networks Canada (ONC) hydrophone `ICLISTENAF2523` (Strait of Georgia, 2016–2017) — for synthetic-mixture model development in the French Embassy Underwater Vessel Noise Challenge.

- **Local acquired release:** **v1.3.0** (released 2026-09-23; verified from local `CITATION.cff` and `dataset_card.md`), authors Hao Yuan, Wenbo Wang, Tian Li, Guici Chen, Lingjiang Zeng, and Sixian Liu, licence **CC-BY-4.0**, Concept DOI `https://doi.org/10.5281/zenodo.22274549`.
- **Later Zenodo record (referenced in Zindi starter notebook):** v1.3.2 (`https://doi.org/10.5281/zenodo.23037895`). Upstream ONC portal: `https://data.oceannetworks.ca/`.
- **Local integrity & inventory:** All 525 strict-interval mono 32 kHz WAV files (509 in `vessel/`, 16 in `background/`) passed SHA-256 verification against `strict_interval_wav_SHA256SUMS.txt`. Metadata record 487 non-background clips and 38 `background`-labelled clips (22 stored in `vessel/` have `class_code=0.0`/`background` and were preserved as background/excluded rather than relabelled).
- **Processing:** Audio was converted to complete-minute 25-band relative digital power summaries, absolute digital gain was discarded, and normalized vessel spectra were combined in linear energy with permitted South African training backgrounds to create controlled synthetic additions (`share_added_vessel`). These clips are AIS-associated candidate vessel passages, not pure vessel-only audio or calibrated SPL.
- **Redistribution:** Submitted code archives do **not** redistribute VTUAD WAVs or South African competition files; reviewers must obtain sources from Zenodo/ONC and Zindi under their respective terms.

### HearMyShip Acoustic Dataset (Small Craft & Coastal Vessels)

- **Source:** University of Zagreb FER **HearMyShip** (`https://hearmyship.fer.hr/`), accessed via `https://hearmyship.fer.hr/api/vessels` (`1,149` catalog records).
- **Local sample:** `96` valid 48 kHz WAV clips across `8` vessel categories (`Yacht`: 23, `Ferry`: 23, `Motorboat`: 23, `Tour boat`: 21, `Cargo Ship`: 2, `Fishing Vessel`: 2, `Auxiliary ship`: 1, `Tour Boat - Diving boat`: 1), group-split by vessel passage (`75` train / `21` validation) in [hearmyship_split_v1.csv](file:///Users/qwaku/Codes/QBE/Zindi/french-embassy-underwater-vessel-noise-challenge/data/external/hearmyship/metadata/hearmyship_split_v1.csv).
- **Processing:** Extracted using the competition's 2s non-overlapping Hann periodogram over base-10 deci-decade bands; raw audio is not redistributed.

---

- **Zindi Discussion URL:** [Done](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36419)
- **Posted UTC:** 23 Nov 2026, 10:50 
