# Submission Status & Blocker Register

Updated: 23 November 2026. **Decision: HOLD FOR HUMAN ZINDI ACTIONS & FINAL VISUAL REVIEW.**

| Priority | Item | Status | Next Action |
|:--|:--|:--|:--|
| Critical | Vessel-activity spatial map | Open gap | Post `docs/ORGANIZER_MAP_CLARIFICATION.md`; Figure 4 is an honest total-sound context map, not a validated vessel-pressure map. |
| High | Public VTUAD disclosure | Draft ready | Post `docs/EXTERNAL_DATA_DISCLOSURE.md` on Zindi discussion board and log URL + timestamp. |
| High | Research manuscript PDF | Built & verified | Inspect `deliverables/research_submission/research_manuscript.pdf` (`671,920` B, 5 pages, `< 10 MB`). |
| High | Runnable code package | Built & clean-tested | `deliverables/research_submission/research_code_v6.zip` (`79,563` B, 17 scripts, SHA-256 `3916469a19c84a1a96a803f6f44e9ec1be869634a901e4a56b289ea352214f62`). |
| High | Zindi research upload | Not uploaded | Upload PDF + code ZIP separately before **13 Dec 2026, 11:30 UTC**; save receipts. |
| Info | Leaderboard CSV | Provisional | Submission 001 (`pBUapzRu`) scored `0.761536` (`1 - RMSE`). No further probe submissions planned. |

## Verification Scope

- Clean-copy execution validates the 6-station descriptive pipeline, the site-context figure script, and the Experiment 007 decision-gate summary from frozen training manifests and authenticated saved outputs.
- Original Experiment 007 model training, bootstrap draws, and raw Parquet-to-manifest creation are preserved in `scripts/` and `reports/`, not bundled into the <10 MB submission ZIP.
- Physical SPL calibration, hydrophone coordinates, and natural-vessel labels remain unestablished by the available challenge data.
