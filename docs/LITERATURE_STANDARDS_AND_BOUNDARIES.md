# International Underwater Acoustic Standards, Empirical Boundaries, and Modeling Roadmap

**Author:** Emmanuel Kwaku Botchway  
**Date:** 25 November 2026  

---

## 1. Executive Summary

To reconcile local unseen-station actuals (`0.8382–0.9219` `1 - RMSE`) with Zindi's five live submissions (`#1: 0.7615 / 0.5175 AUC`, `#2: 0.7750 / 0.5484 AUC`, `#3: 0.7752 / 0.5566 AUC`, `#4: 0.7712 / 0.5639 AUC`, and `#5 a6JJCya4: 0.7875 / 0.5724 AUC`), I grounded my pipeline in international passive acoustic monitoring (`PAM`) standards and cross-device domain-generalization literature. Mapping this challenge directly onto **ISO 18405:2017**, **IEC 61260-1**, **NPL Good Practice Guide No. 133**, **ADEON**, **JOMOPANS / EU MSFD D11C2**, and **Merchant et al. (2012, 2015)**, and implementing both unsupervised baselines (**PCEN**, **RPCA**) and a 16-metric international evaluation suite (`scripts/standardized_evaluation_suite.py`), quantified every physical boundary affecting single models and explained why pairing `v4` and `v5_simple` improved both live metrics simultaneously.

---

## 2. International Underwater Acoustic Standards Mapped to This Challenge

### 2.1 Terminology, Units, and Grid Geometry (`ISO 18405:2017`, `IEC 61260-1`, `NPL GPG 133`, `ADEON`)
- **ISO 18405:2017 (*Underwater acoustics — Terminology*):** Strictly distinguishes *ambient sound* (all sound except that resulting from a deployed target source under study) from *sound pressure level* (`SPL`, dB re $1\,\mu\text{Pa}^2$). Because the South African (`SA`) recorder exports retain source-export dB conventions without verified end-to-end hydrophone sensitivity curves or ADC scaling (and VTUAD digital spectra discard gain), any supervised model that treats absolute dB levels as calibrated pressure violates ISO 18405 and NPL Good Practice Guide No. 133 (Robinson et al., 2014). My global shift-invariance guarantee ($X \mapsto X + c$ leaves features, tensors, and `float64` `holdout` predictions unchanged to `0.0`) enforces unit-agnostic physical invariance.
- **Deci-decade vs. One-Third Octave Bands (`IEC 61260-1`, `ADEON`):** Following Ainslie et al. (ADEON Underwater Soundscape and Modeling Metadata Standard) and IEC 61260-1 base-10 conventions, the competition's 25 frequency bands span nominal centres $31.6\text{--}7943.3\text{ Hz}$ on an exact base-10 **deci-decade** ratio ($10^{1/10} \approx 1.2589$), which is $0.08\%$ broader than a base-2 one-third octave ($2^{1/3} \approx 1.2599$). Extracting VTUAD and HearMyShip features strictly by integrating 2-second Hann-windowed periodograms across these exact base-10 deci-decade limits eliminates inter-dataset spectral filter mismatch.

### 2.2 Ambient Separation and Excess Level (`JOMOPANS`, `MSFD D11C2`, `Merchant et al., 2012, 2015`)
- Under European Marine Strategy Framework Directive **Descriptor 11 Criterion D11C2** (continuous low-frequency anthropogenic sound) and the **JOMOPANS** North Sea framework, anthropogenic shipping contribution is measured via **ambient separation** into a natural background layer ($B$) and a shipping layer ($V$), yielding the **Excess Level**:
  $$L_{\text{excess}} = 10 \log_{10}\!\left(\frac{B + V}{B}\right) = -10 \log_{10}\!\left(1 - \text{share\_added\_vessel}\right) \quad [\text{dB}]$$
  This exact bijective identity connects the competition target `share_added_vessel` $= E_V / (E_B + E_V)$ directly to the European regulatory JOMOPANS excess level $L_{\text{excess}}$ (e.g., $\text{share}=0.50 \iff L_{\text{excess}}=3.01\text{ dB}$; $\text{share}=0.90 \iff L_{\text{excess}}=10.0\text{ dB}$; $\text{share}=0.99 \iff L_{\text{excess}}=20.0\text{ dB}$, the standard JOMOPANS dominance threshold).
- Accordingly, I report **`jomopans_lex_mae_db`** alongside `1 - RMSE` so regression errors are interpretable in standardized physical decibels of excess shipping noise.
- Furthermore, Merchant et al. (2012, 2015) demonstrated that static threshold rules fail on variable coastal baselines unless adaptive percentile baselines are paired with source discrimination—directly motivating why pure percentile subtraction (`Physics_Robust_q20`, `ROC AUC = 0.4897` on `val_cal_hard`) fails whenever natural backgrounds contain real vessels or biological choruses.

---

## 3. South African Coastal Soundscape Literature Context

Empirical findings from Algoa Bay (**St Croix Island**, **Bird Island**), **St Francis Bay**, **Richards Bay**, and False Bay (Fish Hoek / Sea Point) directly explain the complex background regimes observed in the challenge archive:
1. **Snapping Shrimp (`Alpheus` / `Synalpheus` spp.) & Benthic Reefs:** Shallow-water reefs at `07_St_Francis` (`q_d63 = +3.5 dB`), `04_Bird_Island` (`q_d63 = +3.7 dB`), `03_Cape_Recife` (`+3.8 dB`), and `06_Riy_Banks` (`+3.8 dB`) exhibit broadband impulsive high-frequency snapping-shrimp energy above `2 kHz` (`6.3 kHz > 3.98 kHz`) plus ST600 duty-cycle startup `m01` dropouts (`31.7%` of `Test.csv`).
2. **Dusk/Nocturnal Fish Choruses & Geophony:** St Francis Bay displays localized mid-band biological peaks near `631 Hz`, while exposed coastlines generate strong `31.6–100 Hz` surf/wind surges (`22,311` windows in `vlf_surf`).
3. **Real Commercial Shipping & Offshore Bunkering (`strict_hf_ship`):** Algoa Bay, Durban, Saldanha Bay, and Richards Bay harbor heavy real commercial shipping traffic (`7,286` indexed windows with `hf_rng >= 10 dB`, `mf_rng >= 8 dB`, interior `1 <= argmax_hf <= 13`, and sharp `w6_hf <= 8 min`)—confirming the Zindi starter notebook's warning that `y = 0` backgrounds already contain passing ships.

---

## 4. Technical Boundaries Encountered and Literature Solutions

| # | Concrete Boundary Observed | Empirical Signature | Literature Solution & Implementation Status |
|---|---|---|---|
| **B1** | **Uncalibrated / mismatched hydrophone transfer function $h(f)$** | SA stations roll off at HF (`-12.9 dB/dec` excess slope on real SA ships) vs flat ONC VTUAD (`-5.2 dB/dec`); `Ensemble_v2` degrades to `AUC = 0.8657`, `zero_rmse = 0.2023` under `+8 dB/dec` tilt. | **Implemented in v4:** `h(f)` domain randomization during simulation + per-band `q20` rise `R(t,f) = Xc(t,f) - b20(f)` (`h(f)`-invariant to `1.60e-14`), inspired by **PCEN** (Wang et al., 2017; Lostanlen et al., 2019). **Implemented in v5-Simple:** Restricted all 42 tabular predictors strictly to exact `h(f)`-invariant `R_*` + per-band centered `PCEN` without CNN tilt channel `ch4` (`2.89e-33` `Test.csv` warp drift; optional CNN **ResNorm** / **Freq-MixStyle** per Kim et al., 2021; Zhou et al., 2021 if neural models are revisited). |
| **B2** | **Real SA passing ships & reef bursts in `y = 0`** | Unsupervised **RPCA** (`Candès et al., 2011; Lin et al., 2017`) decomposes sparse transients `S(t,f)` cleanly (`Fig. 3`), but achieves only `0.4996` AUC on `val_cal_hard` because real SA vessels and shrimp bursts are also sparse transients! | **Implemented in v4:** Mined `strict_hf_ship`, `reef_active`, and `glitch_drop` strata strictly into `y = 0` (`100%` hard negatives on `val_cal_hard`). **Implemented in v5-Simple:** Added supervised **RPCA** (`S(t,f)` energy/sparsity/kurtosis) features and excluded `strict_hf_ship` from `y=0`, restoring response on unmixed SA ships (`0.5005` vs `0.2357`). |
| **B3** | **Duty-cycle `m01` missingness & `-15 to -66 dB` dropout glitches** | `41.6%` of `Test.csv` lacks `m01` (`620/1500` exact `0111...` pattern); `3.9%` (`58/1500`) exhibit `> 15 dB` single-minute dropouts (`11.1%` in `07_St_Francis`). | **Implemented in v3/v4/v5-Simple:** Glitch-cleaned quantile masking (`clean_min` / invariant `m_med - m_q20` mask in `v5_simple`), mask-aware CNN pooling, and removal of `mask_m01_missing` from tabular predictors to block station-identity shortcuts. |
| **B4** | **Sharp `2–5 min` block transitions vs. smooth 15-min CPA bells** | `68.5%` of `Test.csv` exhibits `max\|ΔHF\| >= 10 dB` and median `w6_hf = 3.0 min`. | **Implemented in v4:** Added `40%` `short_block_vtuad` synthesis (`2–5 min` interior insertions) + step-ratio/boxcar features (`R_hf_step_ratio`, `R_bar_step_ratio`). |
| **B5** | **Zero-inflated regression calibration & IEEE-754 threshold discretization** | `Ensemble_v2` showed `ECE_10 = 0.1889` (`brier_rel = 0.0538`) on `val_cal_hard`; integer step counts (`>= 8.0`) can flip by `1` on exact 1-decimal `Test.csv` ULP rounding under `+37.5 dB`. | **Implemented in v4 & v5-Simple:** Hurdle/blend temperature scaling (`T = 0.9296` in v4, `T = 0.9063` in v5-Simple), **Lin's CCC**, **NSE**, **ECE**, and **Brier decomposition**; replaced hard `>= 8.0` step counts with smooth sigmoids in **`v5-Simple`** (`2.89e-33` `Test.csv` warp diff). |

---

## 5. Standardized Quantitative Benchmark Results

All metrics below were measured locally via `scripts/standardized_evaluation_suite.py` on strict leakage-free `data/synthetic/modeling_v4/` splits (`reports/standards_benchmark/`):

### 5.1 Standardized Regression, Acoustic Excess-Level (`JOMOPANS`), and Calibration (`val_cal_hard`, $n = 1{,}500$)

| Model | `1 - RMSE` ↑ | Lin's `CCC` ↑ | `NSE (R²)` ↑ | Spearman $\rho$ ↑ | JOMOPANS $L_{\text{ex}}$ MAE (dB) ↓ | `ROC AUC` ↑ | `ECE_10` ↓ | Brier Rel ↓ | Brier Res ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Constant_Mean` | `0.7158` | `0.000` | `-0.001` | `0.000` | `1.738` | `0.5000` | `0.0483` | `0.0023` | `0.0000` |
| `Physics_Robust_q20` | `0.3630` | `0.053` | `-4.028` | `0.050` | `6.583` | `0.4897` | `0.3836` | `0.1650` | `0.0045` |
| `RPCA_Unsupervised` | `0.5413` | `-0.154` | `-1.607` | `-0.211` | `3.265` | `0.4996` | `0.1299` | `0.0221` | `0.0007` |
| `Ensemble_v2` | `0.8364` | `0.811` | `0.668` | `0.753` | `0.886` | `0.9312` | `0.1889` | `0.0538` | `0.1328` |
| `Ensemble_v3` | `0.8788` | `0.897` | `0.818` | `0.837` | `0.631` | `0.9634` | `0.0470` | `0.0075` | `0.1764` |
| **`Ensemble_v4`** | **`0.8933`** | **`0.920`** | **`0.859`** | **`0.864`** | **`0.547`** | **`0.9800`** | **`0.0273`** | **`0.0033`** | **`0.1996`** |
| `Ensemble_v5_Simple` | `0.7843` | `0.608` | `0.423` | `0.680` | `1.352` | `0.8765` | `0.1009` | `0.0141` | `0.1061` |
| `Ensemble_v5_Blend` | `0.8542` | `0.827` | `0.737` | `0.817` | `0.959` | `0.9662` | `0.0877` | `0.0149` | `0.1821` |

### 5.2 Unseen-Station `Holdout` (`09_Fish_Hoek` + `10_Sea_Point`, $n = 541$) and Hydrophone Tilt Stress Test

- **Baseline Unwarped `Holdout` (`0 dB/dec`):** `Ensemble_v4` improves `1 - RMSE` from `0.8886` (`v2`) and `0.8971` (`v3`) to **`0.9219`**, Lin's `CCC` from `0.852` to **`0.924`**, JOMOPANS $L_{\text{ex}}$ MAE from `0.481 dB` to **`0.308 dB`**, and `zero_rmse` from `0.0704` (`v2`) / `0.0758` (`v3`) down to **`0.0329`**. `Ensemble_v5_Simple` trades synthetic fit (`0.8382` `1 - RMSE`, `CCC 0.666`, `ECE_10 0.0432`) for zero tilt leakage, while 50/50 `Ensemble_v5_Blend` delivers `0.8914` (`CCC 0.843`, `AUC 0.9629`, `zero_rmse 0.0507`).
- **Under Extreme `+8 dB/dec` Unseen Hydrophone Tilt (`holdout` stress curve):** `Ensemble_v2` degrades to `ROC AUC = 0.8657` (`zero_rmse = 0.2023`, `logloss = 1.1879`) and `Ensemble_v3` suffers `zero_rmse = 0.2045`; `Ensemble_v4` achieves `ROC AUC = 0.9561` (`zero_rmse = 0.0929`, `logloss = 0.3704`); `Ensemble_v5_Simple` sustains flat horizontal `ROC AUC = 0.9194` across `-6..+10 dB/dec` (`zero_rmse = 0.0787`); and **`Ensemble_v5_Blend` achieves the best `+8 dB/dec` `1 - RMSE` (`0.8390`), lowest `logloss` (`0.3310`), and lowest `Brier` (`0.1026`) at `0.9490` `ROC AUC` (`zero_rmse = 0.0807`)**.

### 5.3 Visual Evidence (`reports/standards_benchmark/figures/`)
1. [`fig1_standards_calibration_and_agreement.png`](../reports/standards_benchmark/figures/fig1_standards_calibration_and_agreement.png): 10-bin reliability calibration curve, Lin's CCC / NSE parity plot, and regime-stratified RMSE across Zero / Low / Moderate / High share buckets.
2. [`fig2_boundary_stress_and_calibration_shift.png`](../reports/standards_benchmark/figures/fig2_boundary_stress_and_calibration_shift.png): Cross-hydrophone tilt stress curve (`-6` to `+10 dB/dec`), per-stratum boundary RMSE across all 5 SA soundscape classes, and JOMOPANS D11C2 $L_{\text{excess}}$ MAE comparison.
3. [`fig3_acoustic_signal_boundaries_decomposition.png`](../reports/standards_benchmark/figures/fig3_acoustic_signal_boundaries_decomposition.png): Visual side-by-side decomposition (`Xc`, `h(f)`-invariant `PCEN`, `RPCA` sparse component `S`) for SA snapping-shrimp reef, real SA vessel passage, dropout glitch, and added VTUAD block.

---

## 6. Modeling v5-Simple (Sub #4) & v5-Blend (Sub #5 Live Breakthrough)

1. **Submission #4 (`MAKNM3KB`, `scripts/modeling_v5_simple.py`):** Excludes CNN tilt channels and global-spectrum predictors completely; uses only 42 exact `h(f)`-invariant `R(t,f)` zone statistics, smooth sigmoid steps, per-band centered `PCEN`, and `RPCA` features across `Linear_v5`, shallow `LightGBM_v5`, and shallow `CatBoost_v5`, without suppressing real SA coastal ships in `y=0`. Offline actuals: `v5_val_100pct_cal 0.8302`, `v4_holdout 0.8382`, flat `+8 dB/dec AUC 0.9194 / zero_rmse 0.0787`, `2.89e-33` `Test.csv` warp drift, and restored real SA `strict_hf_ship` score (`0.5005` vs `0.2357` ambient). On Zindi public LB (`24 Nov 2026`), it lifted diagnostic **`ROC AUC` to `0.563911111`** (`+0.0074` over `v4`) at `0.771209221` `1 - RMSE`.
2. **Submission #5 (`a6JJCya4`, `scripts/modeling_v5_blend.py`, `submissions/candidate_v5_blend.csv`):** Exploits the exact algebraic MSE diversity identity between `Ensemble_v4` (`ovjzdeb8`: best offline `0.9219` / broadband share weighting) and `Ensemble_v5_Simple` (`MAKNM3KB`: best invariant presence detection; `Test.csv` share corr `0.6864`, RMS diff `0.1672`) with zero test-set tuning. Offline, it achieves `val_cal_hard 0.8542`, `holdout 0.8914`, and leads `+8 dB/dec` tilt stress (`1 - RMSE 0.8390`, `logloss 0.3310`, `Brier 0.1026`). Live on Zindi (`25 Nov 2026`), **Submission #5 jumped simultaneously to competition-best `1 - RMSE = 0.787467941` (`RMSE = 0.212532058`, `+0.01223` over `#3`) and `ROC AUC = 0.572400000` (`25,758 / 45,000`)**.

---

## References (APA 7th Edition)

Ainslie, M. A., Miksis-Olds, J. L., Martin, S. B., Heaney, K., de Jong, C. A. F., von Benda-Beckmann, A. M., & Lyons, A. P. (2020). *ADEON underwater soundscape and modeling metadata standard* (Version 1.0). Atlantic Deepwater Ecosystem Observatory Network.

Candès, E. J., Li, X., Ma, Y., & Wright, J. (2011). Robust principal component analysis? *Journal of the ACM*, 58(3), 1–37. https://doi.org/10.1145/1970392.1970395

Farcas, A., Powell, C. F., Brookes, K. L., & Merchant, N. D. (2020). Validated shipping noise maps of the Northeast Atlantic. *Science of the Total Environment*, 735, 139509. https://doi.org/10.1016/j.scitotenv.2020.139509

International Electrotechnical Commission. (2014). *Electroacoustics—Octave-band and fractional-octave-band filters—Part 1: Specifications* (IEC 61260-1:2014).

International Organization for Standardization. (2017). *Underwater acoustics—Terminology* (ISO 18405:2017).

Kim, B., Yang, S., Kim, J., Park, H., Lee, J., & Chang, S. (2021). Domain generalization on efficient acoustic scene classification using residual normalization. *Proceedings of the Detection and Classification of Acoustic Scenes and Events Workshop (DCASE 2021)*, 21–25.

Lin, T.-H., Fang, S.-H., & Tsao, Y. (2017). Improving biodiversity assessment via unsupervised separation of biological sounds from long-duration recordings. *Scientific Reports*, 7, 4547. https://doi.org/10.1038/s41598-017-04790-7

Lostanlen, V., Salamon, J., Cartwright, M., McFee, B., Farnsworth, A., Kelling, S., & Bello, J. P. (2019). Per-channel energy normalization: Why and how. *IEEE Signal Processing Letters*, 26(1), 39–43. https://doi.org/10.1109/LSP.2018.2878620

Merchant, N. D., Blondel, P., Dakin, D. T., & Dorocicz, J. (2012). Averaging underwater noise levels for environmental assessment of shipping. *The Journal of the Acoustical Society of America*, 132(4), EL343–EL349. https://doi.org/10.1121/1.4754429

Merchant, N. D., Fristrup, K. M., Johnson, M. P., Tyack, P. L., Witt, M. J., Blondel, P., & Parks, S. E. (2015). Measuring acoustic habitats. *Methods in Ecology and Evolution*, 6(3), 257–265. https://doi.org/10.1111/2041-210X.12330

Robinson, S. P., Lepper, P. A., & Hazelwood, R. A. (2014). *Good practice guide for underwater noise measurement* (NPL Good Practice Guide No. 133). National Physical Laboratory.

Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017). Trainable frontend for robust and far-field keyword spotting. *2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5670–5674. https://doi.org/10.1109/ICASSP.2017.7953242

Zhou, K., Yang, Y., Qiao, Y., & Xiang, T. (2021). Domain generalization with MixStyle. *International Conference on Learning Representations (ICLR)*.
