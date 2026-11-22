# Draft only — external acoustic dataset disclosure for Zindi

**DO NOT POST until the locally used release/version and URLs are verified.** This is an editable review template, not proof of public disclosure or organizer approval.

## Suggested discussion title

External-data declaration — VTUAD / Ocean Networks Canada, verified local release [FILL IN]

## Proposed text

We used a publicly accessible underwater acoustic dataset in our development-only experiments for the French Embassy Underwater Vessel Noise Challenge:

- **Dataset:** VTUAD (rebuilt), *An AIS-Auto-Labeled Vessel-Type Underwater Acoustic Dataset Reconstructed from Ocean Networks Canada Archives*.
- **Authors:** Hao Yuan, Wenbo Wang, Tian Li, Guici Chen, Lingjiang Zeng and Sixian Liu (verify against the record at publication).
- **Public source:** https://zenodo.org/records/23037895 and upstream https://data.oceannetworks.ca/.
- **Version actually used:** [VERIFY AGAINST DOWNLOADED FILES; PREVIOUS LOCAL CARD CALLED IT v1.3.0 candidate].
- **Licence applicable to used artifacts:** [VERIFY THE EXACT PUBLISHER/UPSTREAM RECORD AND RELEVANT NOTICE; PRIOR LOCAL CARD REPORTED CC BY 4.0].
- **Assets used:** [EXACT WAV archive names; metadata/registry; checksums and hashes].
- **Processing:** converted eligible recordings to relative 25-band, one-minute digital band-power features; used candidate vessel-associated recordings as source templates in controlled relative-energy synthetic insertions. The waveforms contain ambient sound and are not calibrated as pure vessel pressure.
- **Download/reproduce:** [PUBLIC DOWNLOAD PAGE AND EXACT ACCESS/PROCESSING INSTRUCTIONS; DO NOT REDISTRIBUTE RESTRICTED AUDIO].
- **Leakage safeguard:** Challenge hidden test data were used only for inference; no hidden test labels or test-derived calibration/training were used.

Any publicly posted declaration should be copied into this project along with its direct forum URL, posted date/time, and final text.

## Evidence required before posting

- [ ] Compare local card/version and actual downloaded archive filenames with Zenodo release record.
- [ ] Inspect original local checksum verification log and matching WAV feature manifests.
- [ ] Confirm the license and acknowledgements for every actually used asset.
- [ ] Verify that processing claims match the scripts and run manifests.
- [ ] Confirm the publication is accurate and accessible to other participants.
- [ ] Record the posted discussion URL and date. No publication has been performed by these scripts.
