# External Data Disclosure (Ready to Post on Zindi)

**Status:** NOT YET POSTED. Post in the Zindi competition discussion board, then record the URL and timestamp below.

---

I am using **VTUAD (rebuilt)** — an AIS-auto-labelled vessel-type underwater acoustic dataset reconstructed from Ocean Networks Canada (ONC) hydrophone `ICLISTENAF2523` (Strait of Georgia, 2016–2017) — for synthetic-mixture model development in the French Embassy Underwater Vessel Noise Challenge.

- **Local acquired release:** **v1.3.0** (released 2026-09-23; verified from local `CITATION.cff` and `dataset_card.md`), authors Hao Yuan, Wenbo Wang, Tian Li, Guici Chen, Lingjiang Zeng, and Sixian Liu, licence **CC-BY-4.0**, Concept DOI `https://doi.org/10.5281/zenodo.22274549`.
- **Later Zenodo record (referenced in Zindi starter notebook):** v1.3.2 (`https://doi.org/10.5281/zenodo.23037895`). Upstream ONC portal: `https://data.oceannetworks.ca/`.
- **Local integrity & inventory:** All 525 strict-interval mono 32 kHz WAV files (509 in `vessel/`, 16 in `background/`) passed SHA-256 verification against `strict_interval_wav_SHA256SUMS.txt`. Metadata record 487 non-background clips and 38 `background`-labelled clips (22 stored in `vessel/` have `class_code=0.0`/`background` and were preserved as background/excluded rather than relabelled).
- **Processing:** Audio was converted to complete-minute 25-band relative digital power summaries, absolute digital gain was discarded, and normalized vessel spectra were combined in linear energy with permitted South African training backgrounds to create controlled synthetic additions (`share_added_vessel`). These clips are AIS-associated candidate vessel passages, not pure vessel-only audio or calibrated SPL.
- **Redistribution:** Submitted code archives do **not** redistribute VTUAD WAVs or South African competition files; reviewers must obtain sources from Zenodo/ONC and Zindi under their respective terms.

---

- **Zindi Discussion URL:** NOT YET RECORDED
- **Posted UTC:** NOT YET RECORDED
