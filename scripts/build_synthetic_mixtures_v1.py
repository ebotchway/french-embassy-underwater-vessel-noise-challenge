#!/usr/bin/env python3
"""Build a small reproducible, leakage-aware simulated vessel-energy dataset.

Inputs are real background 15x25 *level* windows and minute-level VTUAD
*relative digital powers*. This is a relative-energy simulation; it does not
establish calibrated underwater sound-pressure or reproduce a hidden generator.

Usage:
    python scripts/build_synthetic_mixtures_v1.py --self-test --train 200 --validation 80
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path('reports/eda')
BACKGROUND = ROOT / 'background_split_manifest_v2.csv'
VESSELS = ROOT / 'vtuad_minute_band_power_split_v2.csv'
OUT = ROOT / 'synthetic_v1'
CENTRES = [
    31.6, 39.8, 50.1, 63.1, 79.4, 100., 125.9, 158.5, 199.5, 251.2,
    316.2, 398.1, 501.2, 631., 794.3, 1000., 1258.9, 1584.9,
    1995.3, 2511.9, 3162.3, 3981.1, 5011.9, 6309.6, 7943.3,
]
BANDS = [f'TOL_{c:g}' for c in CENTRES]
SHAPE = (15, 25)
# Deliberately explicit: target distributions are assumed, not measured.
TARGETS = (0.0, 0.05, 0.15, 0.30, 0.50, 0.70)


def mix_at_share(background: np.ndarray, vessel: np.ndarray, q: float):
    """Mix nonnegative *linear* energies at desired added energy share."""
    if background.shape != SHAPE or vessel.shape != SHAPE:
        raise ValueError('Expected 15 x 25 inputs')
    if not (np.isfinite(background).all() and np.isfinite(vessel).all()):
        raise ValueError('Nonfinite input')
    if np.any(background < 0) or np.any(vessel < 0):
        raise ValueError('Negative energies')
    if not 0 <= q < 1:
        raise ValueError('Share must be in [0, 1)')
    bsum, vsum = float(background.sum()), float(vessel.sum())
    if bsum <= 0:
        raise ValueError('Empty background')
    if q > 0 and vsum <= 0:
        raise ValueError('Empty vessel template')
    alpha = 0.0 if q == 0 else (q / (1 - q)) * (bsum / vsum)
    added = alpha * vessel
    mixed = background + added
    measured = float(added.sum() / mixed.sum())
    if not np.isclose(q, measured, rtol=0, atol=1e-10):
        raise AssertionError(f'Target mismatch: {q}, {measured}')
    return mixed, measured, alpha


def db_to_relative_energy(levels: np.ndarray):
    """Convert level-like values to linear energy, normalized per window.

    Per-window reference subtraction is an explicit assumption for simulation;
    it intentionally does not claim meaningful cross-station absolute power.
    """
    if levels.shape != SHAPE or not np.isfinite(levels).all():
        raise ValueError('Invalid background window')
    ref = float(np.max(levels))
    return np.power(10., (levels - ref) / 10.)


def embed_vessel(rows: np.ndarray, rng: np.random.Generator):
    """Place a contiguous 1-to-15 minute sample in a 15-minute zero template."""
    if rows.ndim != 2 or rows.shape[1] != 25 or not len(rows):
        raise ValueError('Invalid vessel rows')
    if not np.isfinite(rows).all() or np.any(rows < 0):
        raise ValueError('Bad vessel powers')
    duration = min(len(rows), 15)
    src_start = int(rng.integers(0, len(rows) - duration + 1))
    dest_start = int(rng.integers(0, 16 - duration))
    template = np.zeros(SHAPE, dtype=np.float64)
    template[dest_start:dest_start + duration] = rows[src_start:src_start + duration]
    return template, duration, src_start, dest_start


class ParquetRowReader:
    """Read logical Parquet row positions from necessary row groups only."""
    def __init__(self):
        self.files = {}
        self.cache_key = None
        self.cache_data = None

    def read(self, path: str, positions: list[int]):
        if path not in self.files:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(p)
            pf = pq.ParquetFile(p)
            absent = set(BANDS) - set(pf.schema_arrow.names)
            if absent:
                raise ValueError(f'Missing acoustic bands: {sorted(absent)}')
            ends = np.cumsum([pf.metadata.row_group(i).num_rows
                                 for i in range(pf.num_row_groups)])
            self.files[path] = (pf, ends)
        pf, ends = self.files[path]
        if len(positions) != 15:
            raise ValueError('Incorrect window length')
        arr = np.empty(SHAPE, dtype=np.float64)
        for j, pos in enumerate(positions):
            pos = int(pos)
            group = int(np.searchsorted(ends, pos, side='right'))
            if group >= len(ends) or pos < 0:
                raise IndexError(f'Invalid row position {pos}')
            offset = 0 if group == 0 else int(ends[group - 1])
            key = (path, group)
            if key != self.cache_key:
                table = pf.read_row_group(group, columns=BANDS)
                self.cache_data = np.column_stack([
                    table.column(col).to_numpy(zero_copy_only=False)
                    for col in BANDS
                ])
                self.cache_key = key
            arr[j] = self.cache_data[pos - offset]
        return arr


def check_self():
    rng = np.random.default_rng(4)
    B = rng.uniform(.1, 10., SHAPE)
    V, duration, _, _ = embed_vessel(rng.uniform(.1, 1., (3, 25)), rng)
    assert duration == 3 and np.count_nonzero(V.sum(axis=1)) == 3
    for q in TARGETS:
        M, actual, alpha = mix_at_share(B, V, q)
        assert np.isclose(actual, q, atol=1e-12)
        assert np.all(M >= B)
        if q == 0:
            assert np.array_equal(M, B) and alpha == 0
    assert np.allclose(db_to_relative_energy(np.zeros(SHAPE)), 1.)
    print('MIXTURE SELF-TEST: PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--validation', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        check_self()
    if min(args.train, args.validation) < 1:
        parser.error('Training and validation counts must be positive')

    bg = pd.read_csv(BACKGROUND)
    vt = pd.read_csv(VESSELS, dtype={'hull_id': 'string'})
    assert set(BANDS).issubset(vt.columns)
    assert len(bg) == 88143 and len(vt) == 1892
    assert not vt.duplicated(['clip_uid', 'minute_index']).any()
    assert set(vt.source_split.unique()) == {'train', 'validation'}

    split_hulls = [set(vt.loc[vt.source_split.eq(s), 'hull_id'])
                   for s in ('train', 'validation')]
    assert split_hulls[0].isdisjoint(split_hulls[1])

    # Every sample gets one background window; no reused windows within a split.
    rng = np.random.default_rng(args.seed)
    reader = ParquetRowReader()
    OUT.mkdir(parents=True, exist_ok=True)

    for split, n in (('train', args.train), ('validation', args.validation)):
        pool = bg[bg.split.eq(split)].reset_index(drop=True)
        if n > len(pool):
            raise ValueError(f'Requested {n} but only {len(pool)} {split} windows')
        selection = rng.choice(len(pool), size=n, replace=False)

        vf = vt[vt.source_split.eq(split)].copy()
        # A sampling unit is a full recording, not an individual minute.
        clip_groups = {
            k: g.sort_values('minute_index')
            for k, g in vf.groupby('clip_uid', sort=True)
        }
        clip_ids = sorted(clip_groups)
        assert clip_ids
        matrices = {k: g[BANDS].to_numpy(dtype=np.float64)
                    for k, g in clip_groups.items()}
        clip_info = {k: g.iloc[0] for k, g in clip_groups.items()}
        X = np.empty((n, 15, 25), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        manifest_rows = []

        for j, index in enumerate(selection):
            rec = pool.iloc[int(index)]
            positions = json.loads(rec.row_positions_json)
            levels = reader.read(str(rec.source_parquet), positions)
            B = db_to_relative_energy(levels)

            # Uniform target strata (chosen deliberately, not inferred from Zindi).
            q = TARGETS[j % len(TARGETS)]
            clip_uid, duration, start, clip_start = '', 0, -1, -1
            if q == 0:
                M = B.copy()
                measured, alpha = 0.0, 0.0
            else:
                clip_uid = clip_ids[int(rng.integers(len(clip_ids)))]
                V, duration, clip_start, start = embed_vessel(
                    matrices[clip_uid], rng)
                M, measured, alpha = mix_at_share(B, V, q)
            X[j] = (10 * np.log10(np.maximum(M, 1e-30))).astype(np.float32)
            y[j] = measured
            # Do not supply these simulation provenance/target fields to the model.
            info = clip_info.get(clip_uid)
            manifest_rows.append({
                'example': j, 'partition': split,
                'station': rec.station, 'background_run_id': rec.run_id,
                'background_start': rec.start_time,
                'background_source': rec.source_parquet,
                'vessel_clip_uid': clip_uid,
                'vessel_hull_id': '' if info is None else info.hull_id,
                'vessel_class': '' if info is None else info.label,
                'vessel_duration_min': duration,
                'vessel_audio_start_min': clip_start,
                'inserted_start_min': start,
                'requested_share': q,
                'recovered_share': measured,
                'relative_scale': alpha,
            })
            if (j + 1) % 50 == 0 or j + 1 == n:
                print(f'{split}: {j+1}/{n} generated')

        # Forbid any nonfinite acoustic cells, regardless of simulation.
        assert np.isfinite(X).all() and np.isfinite(y).all()
        assert np.all((y >= 0) & (y < 1))
        output_npz = OUT / f'{split}.npz'
        output_csv = OUT / f'{split}_provenance.csv'
        np.savez_compressed(output_npz, X=X, y=y)
        pd.DataFrame(manifest_rows).to_csv(output_csv, index=False)
        print(f'{split}: X={X.shape}, y={y.shape}')
        print(f'Saved {output_npz} SHA256={hashlib.sha256(output_npz.read_bytes()).hexdigest()}')

    print('SIMULATED MIXTURES V1: PASS')
    print('WARNING: Synthetic validation is NOT evidence of Zindi hidden-test performance.')


if __name__ == '__main__':
    main()
