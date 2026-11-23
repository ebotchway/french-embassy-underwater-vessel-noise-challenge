#!/usr/bin/env python3
"""Extract relative third-octave band power from VTUAD WAV files."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

ROOT = Path("data/external/vtuad/extracted")
SPLIT_FILE = Path("reports/eda/vtuad_source_split_v1.csv")
OUTPUT_DIR = Path("reports/eda")

CENTRES = np.array([
    31.6, 39.8, 50.1, 63.1, 79.4,
    100.0, 125.9, 158.5, 199.5, 251.2,
    316.2, 398.1, 501.2, 631.0, 794.3,
    1000.0, 1258.9, 1584.9, 1995.3, 2511.9,
    3162.3, 3981.1, 5011.9, 6309.6, 7943.3
], dtype=np.float64)

BAND_COLUMNS = [f"TOL_{x:g}" for x in CENTRES]
FS = 32000
BLOCK_SECONDS = 60
NPERSEG = 32768
NOVERLAP = 16384


def band_powers(signal, sample_rate=FS):
    """Calculate 25 relative band powers from a waveform."""
    frequencies, psd = welch(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        detrend="constant",
        scaling="density",
    )

    resolution = frequencies[1] - frequencies[0]

    powers = np.empty(len(CENTRES), dtype=np.float64)

    for i, centre in enumerate(CENTRES):
        lower = centre / (2 ** (1 / 6))
        upper = centre * (2 ** (1 / 6))
        mask = (frequencies >= lower) & (frequencies < upper)
        powers[i] = np.sum(psd[mask]) * resolution

    if not np.isfinite(powers).all():
        raise ValueError("Non-finite band power detected")

    if (powers < 0).any():
        raise ValueError("Negative band power detected")

    return powers


def self_test():
    """Check tone accuracy and approximate broadband coverage."""
    rng = np.random.default_rng(42)
    t = np.arange(FS * 10, dtype=np.float64) / FS

    amplitude = 0.1
    tone = amplitude * np.sin(2 * np.pi * 1000 * t)

    powers = band_powers(tone)
    expected = amplitude ** 2 / 2

    assert CENTRES[np.argmax(powers)] == 1000.0
    assert abs(powers.max() - expected) / expected < 0.02

    print("1 kHz tone test: PASS")

    noise = rng.normal(0, 0.05, size=FS * 10)
    noise_powers = band_powers(noise)

    total_waveform_power = np.mean(noise ** 2)

    # White noise has approximately uniform power per Hz.
    # The 25 bands cover roughly 28 Hz to 8.9 kHz.
    lower = CENTRES[0] / (2 ** (1 / 6))
    upper = CENTRES[-1] * (2 ** (1 / 6))
    expected_fraction = (upper - lower) / (FS / 2)

    observed_fraction = noise_powers.sum() / total_waveform_power

    print(f"Broadband expected fraction: {expected_fraction:.4f}")
    print(f"Broadband observed fraction: {observed_fraction:.4f}")

    assert abs(observed_fraction - expected_fraction) < 0.08
    print("Broadband coverage test: PASS")


def resolve_wav(relative_path):
    """Find a file in its verified, original extracted archive."""
    rel = Path(str(relative_path).removeprefix("./"))

    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe relative path: {rel}")

    if rel.parts[0] not in {"vessel", "background"}:
        raise ValueError(f"Unexpected path: {rel}")

    archive = (
        "vtuad_strict_wav_vessel"
        if rel.parts[0] == "vessel"
        else "vtuad_strict_wav_background"
    )

    return (
        ROOT
        / archive
        / "strict_interval_wav"
        / "inclusion_4000_exclusion_6000"
        / rel
    )


def extract_clip(row):
    """Yield one row for each complete recording minute."""
    path = resolve_wav(row.path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with sf.SoundFile(path) as audio:
        if audio.samplerate != FS or audio.channels != 1:
            raise ValueError(f"Unexpected audio format: {path}")

        expected_minutes = audio.frames // (FS * BLOCK_SECONDS)

        for minute in range(expected_minutes):
            signal = audio.read(
                frames=FS * BLOCK_SECONDS,
                dtype="float64",
                always_2d=False,
            )

            if len(signal) != FS * BLOCK_SECONDS:
                raise ValueError(f"Unexpected short read: {path}")

            powers = band_powers(signal)

            result = {
                "clip_uid": row.clip_uid,
                "path": row.path,
                "source_split": row.source_split,
                "hull_id": row.hull_id,
                "label": row.label,
                "minute_index": minute,
                "seconds_used": BLOCK_SECONDS,
                "relative_total_power": float(powers.sum()),
            }

            result.update(
                {column: float(value)
                 for column, value in zip(BAND_COLUMNS, powers)}
            )

            yield result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N clips for smoke testing",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run synthetic signal tests before extraction",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    if args.self_test:
        self_test()

    df = pd.read_csv(
        SPLIT_FILE,
        dtype={"hull_id": "string"},
    )

    active = df[
        df["source_split"].isin(["train", "validation"])
    ].copy()

    assert len(df) == 525
    assert len(active) == 487
    assert active["clip_uid"].is_unique
    assert active["hull_id"].notna().all()
    assert not active["label"].eq("background").any()

    train_hulls = set(
        active.loc[
            active["source_split"].eq("train"), "hull_id"
        ]
    )
    val_hulls = set(
        active.loc[
            active["source_split"].eq("validation"), "hull_id"
        ]
    )
    assert train_hulls.isdisjoint(val_hulls)

    active = active.sort_values("path")

    if args.limit is not None:
        active = active.head(args.limit)

    filename = (
        f"vtuad_band_power_smoke_{args.limit}.csv"
        if args.limit is not None
        else "vtuad_minute_band_power_v1.csv"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / filename
    temporary = output.with_suffix(".tmp")

    columns = [
        "clip_uid", "path", "source_split",
        "hull_id", "label", "minute_index",
        "seconds_used", "relative_total_power",
        *BAND_COLUMNS,
    ]

    num_rows = 0
    num_clips = 0

    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()

            for row in active.itertuples(index=False):
                clip_rows = list(extract_clip(row))

                if not clip_rows:
                    raise ValueError(
                        f"Recording has no full minutes: {row.path}"
                    )

                writer.writerows(clip_rows)
                num_rows += len(clip_rows)
                num_clips += 1

                if num_clips % 25 == 0 or num_clips == len(active):
                    print(
                        f"Processed {num_clips}/{len(active)} clips; "
                        f"{num_rows} minute rows"
                    )

        temporary.replace(output)

    finally:
        if temporary.exists():
            temporary.unlink()

    print("\n=== EXTRACTION COMPLETE ===")
    print("Clips processed:", num_clips)
    print("One-minute records:", num_rows)
    print("Band columns:", len(BAND_COLUMNS))
    print("Saved:", output)
    print("FEATURE EXTRACTION: PASS")


if __name__ == "__main__":
    main()
