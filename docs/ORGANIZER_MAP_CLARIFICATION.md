# Organizer Clarification Request — Spatial Site-Context Map

**Status:** **Answered by Zindi Staff (`daniil_zindi`, 24 Nov 2026, 08:00 UTC) & Implemented.** (Posted: 23 Nov 2026, 10:54 UTC; Community input from `De_DQ`: 24 Nov 2026, 07:18 UTC).

---

**Subject:** French Embassy Underwater Vessel Noise Challenge — Research Write-Up Spatial Map Scope

Hello Zindi & Challenge Organizers,

Our research write-up characterizes 460 training-only recording runs across six South African monitoring sites using 25 third-octave bands. Because `stations.csv` labels coordinates as *working site-level candidates* (with two unresolved candidates for Saldanha Bay) and does not provide deployment dates, physical calibration to dB re 1 µPa, or time-aligned vessel labels, we plotted a **non-interpolated site-context figure (Figure 4)** showing total recorded sound (median run-window mean in source-export units) and sampled run counts at those candidate coordinates, with all caveats stated explicitly.

Could you confirm whether this total-sound site-context figure satisfies the research write-up's spatial-visualization criterion as partial evidence, or whether verified hydrophone deployment metadata, calibration constants, or aligned vessel events can be provided? We prefer stating exact evidentiary limits rather than interpolating or labelling unattributed total sound as vessel pressure.

Thank you.

---

### Official Zindi Staff Ruling (`daniil_zindi`, 24 Nov 2026, 08:00 UTC)

> *"A map of total recorded sound at the candidate sites is useful context and can count as partial evidence. For the vessel-related claim, show how the observed signals support the model or proxy on held-out South African sites or periods. Include a baseline and explain uncertainty.*
>
> *A site-level map is acceptable. Label any uncalibrated estimate as relative acoustic activity and make location uncertainty and measurement coverage clear."*

### Implemented Compliance Upgrade

- **Script & Data Scope:** Created `scripts/evaluate_sa_relative_vessel_activity.py` evaluating strictly **17,487 held-out real South African 15-minute windows** (`0` training rows, `0` `Test.csv` rows) across all `8` usable stations (`6` held-out validation periods + `2` completely unseen holdout sites `09_Fish_Hoek` and `10_Sea_Point`) while explicitly documenting `Cape Recife` and `Riy Banks` (`0` complete 15-minute windows).
- **Relative Acoustic Activity, Baselines & Uncertainty:** Computed the exact $h(f)$-invariant transient vessel-like activity proxy (`strict_hf_ship` rate; verified `0.0` change under $+8\,\mathrm{dB/dec}$ tilt) alongside **Baseline 1 (`median_total_sound_db`)** and **Baseline 2 (`Physics_Robust_q20` vs. `Ensemble_v2` vs. `Ensemble_v4`)** with **1,000-draw calendar-day block bootstrap 95% CIs (seed 42)** and explicit provisional flags for `< 5`-day held-out sites (`09_Fish_Hoek`: `4` days; `10_Sea_Point`: `3` days).
- **Artifact Outputs:** `reports/standards_benchmark/sa_site_relative_vessel_activity.csv`, `reports/standards_benchmark/sa_heldout_signal_support.csv`, `reports/standards_benchmark/sa_site_activity_summary.json`, and upgraded 3-panel **Figure 2** (`reports/standards_benchmark/figures/fig2_site_relative_acoustic_activity_map.png` staged into `deliverables/overleaf_manuscript/research_manuscript_latex.pdf`).
