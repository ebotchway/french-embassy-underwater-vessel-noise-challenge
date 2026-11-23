#!/usr/bin/env python3
"""Synthetic Experiment 002: class-aware, station-balanced relative-energy mixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_synthetic_mixtures_v1 import (BANDS, SHAPE, ParquetRowReader,
                                         db_to_relative_energy, mix_at_share)

BG_DEFAULT = Path('reports/eda/background_split_manifest_v2.csv')
VT_DEFAULT = Path('reports/eda/vtuad_minute_band_power_split_v2.csv')
OUT_DEFAULT = Path('reports/eda/synthetic_v2')


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample_fraction(rng, zero_probability):
    """Prespecified target mix: zero, weak continuous, and broad continuous."""
    u = rng.random()
    if u < zero_probability:
        return 0.0, 'zero'
    if u < zero_probability + (1.0 - zero_probability) * 0.35:
        return float(rng.uniform(0.005, 0.10)), 'weak'
    return float(rng.uniform(0.10, 0.80)), 'broad'


def sample_background(pool, n, rng):
    """Near-even sampling by station without reusing an indexed window."""
    grouped = {station: block.index.to_numpy(dtype=np.int64)
               for station, block in pool.groupby('station', sort=True)}
    if not grouped or n > len(pool):
        raise ValueError('Insufficient available background windows')
    stations = sorted(grouped)
    chosen = {s: [] for s in stations}
    remaining = {s: rng.permutation(grouped[s]).tolist() for s in stations}
    out = []
    while len(out) < n:
        viable = [s for s in stations if remaining[s]]
        if not viable:
            raise ValueError('Exhausted background windows')
        minimum = min(len(chosen[s]) for s in viable)
        candidates = [s for s in viable if len(chosen[s]) == minimum]
        station = candidates[int(rng.integers(len(candidates)))]
        index = remaining[station].pop()
        chosen[station].append(index)
        out.append(index)
    assert len(set(out)) == n
    return out


def select_vessel_source(class_map, rng):
    # Choose class first, then clip: less tug dominance than choosing clips uniformly.
    vessel_class = sorted(class_map)[int(rng.integers(len(class_map)))]
    clips = class_map[vessel_class]
    clip_uid = clips[int(rng.integers(len(clips)))]
    return vessel_class, clip_uid


def embed_event(rows, rng):
    """Use real adjacent available minutes without looping or padding with audio."""
    if rows.ndim != 2 or rows.shape[1] != 25 or len(rows) < 1:
        raise ValueError('Invalid vessel recording features')
    max_len = min(15, len(rows))
    duration = int(rng.integers(1, max_len + 1))
    source_offset = int(rng.integers(0, len(rows) - duration + 1))
    insertion = int(rng.integers(0, 16 - duration))
    template = np.zeros(SHAPE, dtype=np.float64)
    template[insertion:insertion + duration] = rows[source_offset:source_offset + duration]
    return template, duration, source_offset, insertion


def self_test():
    r = np.random.default_rng(8)
    B = r.uniform(.1, 10, SHAPE)
    rows = r.uniform(.01, .2, (6, 25))
    seen = set()
    for _ in range(300):
        q, kind = sample_fraction(r, .20)
        seen.add(kind)
        V, duration, source_start, insertion = embed_event(rows, r)
        assert 1 <= duration <= 6 and source_start >= 0
        assert insertion + duration <= 15
        assert np.count_nonzero(V.sum(axis=1)) == duration
        M, observed, _ = mix_at_share(B, V, q)
        assert np.isclose(q, observed, atol=1e-11)
        assert np.all(M >= B)
    assert seen == {'zero', 'weak', 'broad'}
    print('EXPERIMENT 002 SELF-TEST: PASS')


def run(args):
    if min(args.train, args.validation) < 1 or not 0 <= args.zero_probability < 1:
        raise ValueError('Counts must be positive; --zero-probability must lie in [0,1)')
    if args.output.resolve() == Path('reports/eda/synthetic_v1').resolve():
        raise ValueError('Refusing to overwrite Experiment 001')
    if args.self_test:
        self_test()

    bg = pd.read_csv(args.background)
    vt = pd.read_csv(args.vessels, dtype={'hull_id': 'string'})
    required_bg = {'split', 'station', 'run_id', 'start_time', 'source_parquet', 'row_positions_json'}
    required_vt = {'source_split', 'clip_uid', 'hull_id', 'label', 'minute_index', *BANDS}
    if not required_bg <= set(bg.columns) or not required_vt <= set(vt.columns):
        raise ValueError('Missing input manifest columns')
    if not vt.source_split.isin(['train', 'validation']).all():
        raise ValueError('Vessel features must use v2 train/validation assignments')
    if vt.duplicated(['clip_uid', 'minute_index']).any():
        raise ValueError('Duplicate vessel minute indexes')
    if not np.isfinite(vt[BANDS].to_numpy(float)).all() or (vt[BANDS].to_numpy(float) < 0).any():
        raise ValueError('Invalid vessel band powers')
    train_hulls = set(vt.loc[vt.source_split.eq('train'), 'hull_id'])
    val_hulls = set(vt.loc[vt.source_split.eq('validation'), 'hull_id'])
    assert train_hulls.isdisjoint(val_hulls)
    run_train = set(bg.loc[bg.split.eq('train'), 'run_id'])
    run_val = set(bg.loc[bg.split.eq('validation'), 'run_id'])
    assert run_train.isdisjoint(run_val)

    # Split seeds are independent so adjusting train count does not change validation.
    seeds = {'train': args.seed, 'validation': args.seed + 1000003}
    reader = ParquetRowReader()
    args.output.mkdir(parents=True, exist_ok=True)
    info = {'experiment': '002', 'seed': args.seed,
            'train_seed': seeds['train'], 'validation_seed': seeds['validation'],
            'zero_probability': args.zero_probability,
            'positive_target': '35% U(0.005,0.10), 65% U(0.10,0.80)',
            'class_sampling': 'uniform across classes, then uniform across clips within class',
            'station_sampling': 'near-equal station counts, windows unique per partition',
            'representation': '10*log10 of per-window normalized relative energy',
            'calibration': 'NOT calibrated sound pressure, vessel recordings may contain ambient noise',
            'input_hashes': {'background': digest(args.background), 'vessels': digest(args.vessels)}}

    provenance_by_split = {}
    for split, n in [('train', args.train), ('validation', args.validation)]:
        rng = np.random.default_rng(seeds[split])
        pool = bg.loc[bg.split.eq(split)].reset_index(drop=True)
        selections = sample_background(pool, n, rng)
        selected = vt.loc[vt.source_split.eq(split)].copy()
        clip_groups = {uid: g.sort_values('minute_index')
                       for uid, g in selected.groupby('clip_uid', sort=True)}
        clip_arrays = {uid: g[BANDS].to_numpy(np.float64) for uid, g in clip_groups.items()}
        clip_info = {uid: g.iloc[0] for uid, g in clip_groups.items()}
        class_map = {}
        for uid, g in clip_groups.items():
            row = g.iloc[0]
            if g['hull_id'].nunique() != 1 or g['label'].nunique() != 1:
                raise ValueError(f'Inconsistent clip metadata: {uid}')
            if not np.array_equal(g.minute_index.to_numpy(), np.arange(len(g))):
                raise ValueError(f'Nonconsecutive vessel minutes: {uid}')
            class_map.setdefault(row.label, []).append(uid)
        X = np.empty((n, *SHAPE), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        records = []
        for j, index in enumerate(selections):
            rec = pool.iloc[index]
            B = db_to_relative_energy(reader.read(str(rec.source_parquet), json.loads(rec.row_positions_json)))
            q, regime = sample_fraction(rng, args.zero_probability)
            uid, vessel_class, duration, src_start, insertion = '', '', 0, -1, -1
            if q == 0:
                M, actual, alpha = B.copy(), 0.0, 0.0
            else:
                vessel_class, uid = select_vessel_source(class_map, rng)
                V, duration, src_start, insertion = embed_event(clip_arrays[uid], rng)
                M, actual, alpha = mix_at_share(B, V, q)
            X[j] = (10 * np.log10(np.maximum(M, 1e-30))).astype(np.float32)
            y[j] = actual
            source = clip_info.get(uid)
            records.append({
                'example': j, 'partition': split, 'station': rec.station,
                'background_run_id': rec.run_id, 'background_start': rec.start_time,
                'background_source': rec.source_parquet,
                'vessel_clip_uid': uid,
                'vessel_hull_id': '' if source is None else source.hull_id,
                'vessel_class': vessel_class,
                'vessel_duration_min': duration,
                'vessel_audio_start_min': src_start,
                'inserted_start_min': insertion,
                'requested_share': q, 'recovered_share': actual,
                'relative_scale': alpha, 'target_regime': regime,
            })
            if (j + 1) % 200 == 0 or j + 1 == n:
                print(f'{split}: {j+1}/{n} examples')
        assert np.isfinite(X).all() and np.isfinite(y).all()
        assert np.all((y >= 0) & (y < 1))
        df = pd.DataFrame(records)
        assert np.allclose(df.requested_share, y, rtol=0, atol=1e-6)
        provenance_by_split[split] = df
        # Create output files only after each partition's checks have passed.
        np.savez_compressed(args.output / f'{split}.npz', X=X, y=y)
        df.to_csv(args.output / f'{split}_provenance.csv', index=False)
        print(f'{split}: X={X.shape}, y={y.shape}; stations={df.station.nunique()}')
        print('Station counts:', df.station.value_counts().sort_index().to_dict())
        print('Target regimes:', df.target_regime.value_counts().to_dict())
        info[split] = {'examples': n,
                       'npz_sha256': digest(args.output / f'{split}.npz'),
                       'provenance_sha256': digest(args.output / f'{split}_provenance.csv')}

    a, b = provenance_by_split['train'], provenance_by_split['validation']
    for key in ['background_run_id', 'vessel_clip_uid', 'vessel_hull_id']:
        x = set(a[key].dropna().astype(str)) - {''}
        z = set(b[key].dropna().astype(str)) - {''}
        if x & z:
            raise ValueError(f'Cross-partition overlap in {key}')
    (args.output / 'experiment_config.json').write_text(json.dumps(info, indent=2) + '\n')
    print('EXPERIMENT 002 GENERATION: PASS')
    print('WARNING: Validation is simulated; this is not a Zindi competition score.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=1200)
    parser.add_argument('--validation', type=int, default=300)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--zero-probability', type=float, default=.20)
    parser.add_argument('--background', type=Path, default=BG_DEFAULT)
    parser.add_argument('--vessels', type=Path, default=VT_DEFAULT)
    parser.add_argument('--output', type=Path, default=OUT_DEFAULT)
    parser.add_argument('--self-test', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
