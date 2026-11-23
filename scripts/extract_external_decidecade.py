#!/usr/bin/env python3
"""Extract 2s base-10 deci-decade band powers matching the Zindi starter notebook."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import periodogram

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from extract_vtuad_band_features import BAND_COLUMNS, resolve_wav  # noqa: E402

EXACT_CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25, dtype=np.float64))
HALF_BAND_FACTOR = 10.0 ** 0.05
SEGMENT_SECONDS = 2
SEGMENTS_PER_MINUTE = 30

DEFAULT_VTUAD_SPLIT = ROOT / "reports/eda/vtuad_source_split_v2.csv"
DEFAULT_VTUAD_OUT = ROOT / "data/processed/vtuad_decidecade_2s_minutes.parquet"
DEFAULT_HMS_SPLIT = ROOT / "data/external/hearmyship/metadata/hearmyship_split_v1.csv"
DEFAULT_HMS_OUT = ROOT / "data/processed/hearmyship_decidecade_2s_clips.parquet"


def starter_minute_band_power(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Exact organizer Cell-11 implementation for verification and minute-level extraction."""
    segment_size = int(2 * sample_rate_hz)
    minutes = samples.size // (30 * segment_size)
    if minutes < 1:
        return np.empty((0, 25), dtype=np.float64)
    segments = samples[: minutes * 30 * segment_size].reshape(-1, segment_size)
    frequency, psd = periodogram(
        segments,
        fs=sample_rate_hz,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    centres = 10.0 ** (1.5 + 0.1 * np.arange(25))
    df = frequency[1] - frequency[0]
    power = np.full((segments.shape[0], 25), np.nan, dtype=np.float64)
    for band, centre in enumerate(centres):
        low = centre / 10 ** 0.05
        high = centre * 10 ** 0.05
        bins = (frequency >= low) & (frequency < high)
        if high <= sample_rate_hz / 2 and bins.any():
            power[:, band] = psd[:, bins].sum(axis=1) * df
    return power.reshape(minutes, 30, 25).mean(axis=1)


def decidecade_2s_segment_powers(samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Compute per-2s-segment base-10 deci-decade band powers using Hann periodogram."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    segment_size = int(SEGMENT_SECONDS * sample_rate_hz)
    n_segments = arr.size // segment_size
    if n_segments < 1:
        raise ValueError(f"Audio shorter than {SEGMENT_SECONDS}s segment ({arr.size} < {segment_size})")

    segments = arr[: n_segments * segment_size].reshape(n_segments, segment_size)
    freq, psd = periodogram(
        segments,
        fs=sample_rate_hz,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    df = float(freq[1] - freq[0])
    powers = np.full((n_segments, len(EXACT_CENTRES)), np.nan, dtype=np.float64)
    for band_idx, centre in enumerate(EXACT_CENTRES):
        low = centre / HALF_BAND_FACTOR
        high = centre * HALF_BAND_FACTOR
        bins = (freq >= low) & (freq < high)
        if high <= sample_rate_hz / 2.0 and bins.any():
            powers[:, band_idx] = psd[:, bins].sum(axis=1) * df
    return powers


def extract_vtuad_dataset(split_csv: Path, output_parquet: Path) -> pd.DataFrame:
    """Extract all complete 60s minutes from all 525 VTUAD WAV files."""
    manifest = pd.read_csv(split_csv)
    rows: list[dict] = []
    frames_per_minute = 32000 * 60

    for row in manifest.itertuples(index=False):
        wav_path = resolve_wav(row.path)
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        with sf.SoundFile(wav_path) as audio:
            if audio.samplerate != 32000 or audio.channels != 1:
                raise ValueError(f"Unexpected VTUAD format in {wav_path}: sr={audio.samplerate}, ch={audio.channels}")
            n_minutes = audio.frames // frames_per_minute
            for m_idx in range(n_minutes):
                block = audio.read(frames=frames_per_minute, dtype="float64", always_2d=False)
                if len(block) != frames_per_minute:
                    raise ValueError(f"Short read in {wav_path} minute {m_idx}")
                seg_powers = decidecade_2s_segment_powers(block, 32000)
                if seg_powers.shape != (SEGMENTS_PER_MINUTE, 25):
                    raise ValueError(f"Unexpected segment shape {seg_powers.shape}")
                mean_power = seg_powers.mean(axis=0)
                if not np.isfinite(mean_power).all() or (mean_power < 0).any():
                    raise ValueError(f"Invalid band powers in {wav_path} minute {m_idx}")
                rec = {
                    "clip_uid": row.clip_uid,
                    "path": row.path,
                    "label": row.label,
                    "directory_category": row.directory_category,
                    "hull_id": row.hull_id,
                    "source_split": row.source_split,
                    "minute_index": int(m_idx),
                    "seconds_used": 60,
                    "sample_rate": 32000,
                    "relative_total_power": float(mean_power.sum()),
                }
                rec.update({col: float(val) for col, val in zip(BAND_COLUMNS, mean_power)})
                rows.append(rec)

    df = pd.DataFrame(rows)
    assert df["clip_uid"].nunique() == 525, f"Expected 525 VTUAD clips, got {df['clip_uid'].nunique()}"
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df


def extract_hearmyship_dataset(split_csv: Path, output_parquet: Path) -> pd.DataFrame:
    """Extract clip-mean 2s base-10 deci-decade band powers for downloaded HearMyShip WAVs."""
    manifest = pd.read_csv(split_csv)
    valid = manifest[manifest["status"].isin(["OK", "EXISTS"])].copy()
    rows: list[dict] = []

    for row in valid.itertuples(index=False):
        wav_path = ROOT / row.local_path
        if not wav_path.is_file():
            continue
        with sf.SoundFile(wav_path) as audio:
            sr = int(audio.samplerate)
            if sr < 18000:
                continue
            signal = audio.read(dtype="float64", always_2d=False)
            clip_dur = audio.frames / sr
        seg_powers = decidecade_2s_segment_powers(signal, sr)
        mean_power = seg_powers.mean(axis=0)
        if not np.isfinite(mean_power).all() or (mean_power < 0).any():
            raise ValueError(f"Invalid band powers in {wav_path}")
        rec = {
            "id": str(row.id),
            "category": str(row.category),
            "vessel_group": str(row.vessel_group),
            "source_split": str(row.source_split),
            "local_path": str(row.local_path),
            "sample_rate": sr,
            "clip_duration_s": float(clip_dur),
            "n_segments": int(seg_powers.shape[0]),
            "seconds_used": int(seg_powers.shape[0] * SEGMENT_SECONDS),
            "relative_total_power": float(mean_power.sum()),
        }
        rec.update({col: float(val) for col, val in zip(BAND_COLUMNS, mean_power)})
        rows.append(rec)

    df = pd.DataFrame(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df


def run_self_test() -> None:
    """Verify exact equivalence to organizer starter_minute_band_power + band isolation."""
    rng = np.random.default_rng(42)
    fs = 32000
    noise = rng.normal(0.0, 0.05, size=fs * 60)
    t = np.arange(fs * 60, dtype=np.float64) / fs
    sig = noise + 0.2 * np.sin(2.0 * np.pi * 1000.0 * t)

    ref = starter_minute_band_power(sig, fs)
    seg = decidecade_2s_segment_powers(sig, fs)
    got = seg.reshape(1, 30, 25).mean(axis=1)
    assert ref.shape == (1, 25) and got.shape == (1, 25)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-15), "Mismatch against organizer Cell-11 implementation"

    pure_tone = 0.1 * np.sin(2.0 * np.pi * 1000.0 * t[: fs * 4])
    tone_p = decidecade_2s_segment_powers(pure_tone, fs).mean(axis=0)
    assert BAND_COLUMNS[int(np.argmax(tone_p))] == "TOL_1000"
    assert abs(tone_p.max() - 0.1**2 / 2.0) / (0.1**2 / 2.0) < 0.02

    # Check stereo handling + arbitrary fs (48 kHz)
    stereo_48k = np.column_stack([rng.normal(0, 0.02, 48000 * 10), rng.normal(0, 0.02, 48000 * 10)])
    p48 = decidecade_2s_segment_powers(stereo_48k, 48000)
    assert p48.shape == (5, 25) and np.isfinite(p48).all() and (p48 >= 0).all()
    print("extract_external_decidecade self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--vtuad-split", type=Path, default=DEFAULT_VTUAD_SPLIT)
    parser.add_argument("--vtuad-out", type=Path, default=DEFAULT_VTUAD_OUT)
    parser.add_argument("--hearmyship-split", type=Path, default=DEFAULT_HMS_SPLIT)
    parser.add_argument("--hearmyship-out", type=Path, default=DEFAULT_HMS_OUT)
    parser.add_argument("--skip-vtuad", action="store_true")
    parser.add_argument("--skip-hearmyship", action="store_true")
    args = parser.parse_args()

    if args.self_test_only:
        run_self_test()
        return

    if not args.skip_vtuad:
        vtuad_df = extract_vtuad_dataset(args.vtuad_split, args.vtuad_out)
        print(
            f"VTUAD extracted: {len(vtuad_df)} minute-rows across {vtuad_df['clip_uid'].nunique()} clips -> {args.vtuad_out}",
            flush=True,
        )

    if not args.skip_hearmyship:
        if not args.hearmyship_split.is_file():
            raise FileNotFoundError(f"Missing HearMyShip split manifest: {args.hearmyship_split}")
        hms_df = extract_hearmyship_dataset(args.hearmyship_split, args.hearmyship_out)
        print(
            f"HearMyShip extracted: {len(hms_df)} clip-rows across {hms_df['category'].nunique()} categories -> {args.hearmyship_out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
