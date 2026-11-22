# External acoustic data declaration — French Embassy Underwater Vessel Noise Challenge

**Status: DRAFT — review against local dataset card before posting. Not yet published.**

I am declaring the external acoustic dataset used for my first exploratory submission and ongoing research in the French Embassy Underwater Vessel Noise Challenge.

**Source and attribution.** I used the publicly released *VTUAD (rebuilt): An AIS-Auto-Labeled Vessel-Type Underwater Acoustic Dataset Reconstructed from Ocean Networks Canada Archives*, derived from Ocean Networks Canada (ONC) acoustic recordings and AIS information. The local dataset card describes the source package as **v1.3.0 candidate**. The source collection has a versioned Zenodo record at https://zenodo.org/records/23037895 (DOI: https://doi.org/10.5281/zenodo.23037895), and ONC's archive is available at https://data.oceannetworks.ca/. The local card lists **CC BY 4.0**; users should consult the release record and ONC attribution requirements before redistributing any material.

**Subset and checks.** My work used the 32 kHz mono strict-interval WAV recordings. The local extraction includes 525 recordings, checked against the published per-file SHA-256 ledger; 487 have non-background vessel-class labels and 38 are background-labelled references. Labels are based on AIS proximity and are not proof of isolated vessel-only audio. The original publisher's recording split was not used for model validation because some known hull identities appear across partitions.

**Processing.** I converted complete one-minute portions of eligible recordings into relative digital-energy estimates over the challenge's 25 acoustic frequency bands. I grouped recordings by pseudonymous vessel hull to separate modelling training and validation, and generated synthetic 15-minute mixtures using South African background-window data provided for the competition. The resulting synthetic targets are mathematically imposed fractions of relative energy. They do not establish calibrated underwater sound-pressure measurements, confirmed natural vessel presence, or physical equivalence of the two source domains.

**Model use and test restrictions.** My first submission used an openly reproducible Random Forest trained on synthetic examples constructed from permitted development data. The Zindi test inputs were used only for per-window inference and fixed missing-minute handling, not for fitting, pseudo-labelling, calibration, or training-sample generation. The submitted `score_added_vessel` was an **uncalibrated copy of the predicted energy fraction**, not a separately trained detector; the leaderboard's diagnostic AUC should be interpreted accordingly.

**Version clarity.** Some challenge discussions refer to VTUAD **v1.3.2**. I am **not** claiming my earlier extracted data are v1.3.2. I will compare the current public release against my local v1.3.0-candidate records before using any updated files in a new experiment. The existing experiments remain pinned to their audited local inputs.

**Equal access.** The versioned source link above, licensing details, strict-interval WAV files, metadata and checksums are available through the linked public record. My own processing and evaluation methods are described in the challenge write-up and reproducibility materials. I am not redistributing organizer-provided challenge data.

---

**Pre-post checks:** (1) Check the local `data/external/vtuad/metadata/dataset_card.md` and actual download record for the precise version. (2) Verify the release DOI and author/attribution details displayed on Zenodo. (3) Confirm the posted source has equal access and license approval as required by Zindi. (4) Replace any incorrect release description before posting. (5) Record the public discussion URL after posting.


### VTUAD Archive and Metadata Classification Reconciliation

The retained VTUAD strict-interval audio archives contain 525 WAV files: 509 files in the original vessel directory and 16 in the original background directory. Both ZIP archives passed internal integrity checks, and the retained background archive matched the published MD5 checksum.

The processed source manifest contains 525 unique clip identifiers, with all corresponding WAV files present. Metadata classification identifies 487 non-background clips and 38 background-labelled clips. The difference arises because 22 recordings located in the original vessel directory carry a background metadata label.

All 38 metadata-background clips are assigned to the background-reference allocation. The remaining 487 non-background clips are distributed into 410 training and 77 validation clips.

Directory placement, metadata classification and experimental allocation are treated as distinct attributes. No metadata labels were changed to force agreement between the archive directories and processed classifications.

These findings establish consistency of the local file inventory and processing manifest. They do not independently confirm physically isolated vessel signals or establish the provenance of every publisher-side metadata annotation. The retained vessel archive still requires comparison against an authenticated publisher asset checksum.