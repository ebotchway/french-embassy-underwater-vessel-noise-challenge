#!/usr/bin/env python3
"""Phase 3 matched counterfactual stress study (development only).

Run from the repository root. Import the existing v1/v2 scripts without modifying them.
Fit two prespecified Random Forest models on *synthetic_v2/train only*:
  1. submitted preprocessing: per-window max shift
  2. original synthetic preprocessing: no additional shift
Use *validation-only* background runs and hull-isolated VTUAD sources to
construct matched zero/positive examples with event durations 1-4 minutes.
For missing-minute tests, recompute the target on the observed cells, rather
than reusing the complete-window target. Test values/labels are NEVER used.
The source validation pool was previously inspected in Experiment 002;
this is a *development robustness audit*, not an independent holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from build_synthetic_mixtures_v1 import BANDS, SHAPE, ParquetRowReader, db_to_relative_energy, mix_at_share
from build_synthetic_mixtures_v2 import sample_background
from prepare_zindi_first_submission import normalize, impute_missing_minutes
from train_synthetic_baselines_v2 import feature_families
from train_synthetic_baselines_v1 import load_partition, audit_overlap


def checksum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def masks() -> dict[str, list[int]]:
    # Prespecified; these masks do not depend on actual challenge test values.
    return {'complete': [], 'first': [0], 'last': [14],
            'middle': [7], 'both_edges': [0, 14], 'three_middle': [6, 7, 8]}


def observed_fraction(B, V, alpha, observed):
    """Observed-cell label, with every band of each observed minute counted."""
    num = float((alpha * V)[observed].sum(dtype=np.float64))
    den = float((B + alpha * V)[observed].sum(dtype=np.float64))
    if den <= 0 or not np.isfinite(den):
        raise ValueError('Invalid energy denominator')
    return num / den


def tests():
    r = np.random.default_rng(19)
    B = r.uniform(0.1, 5, SHAPE)
    V = np.zeros(SHAPE)
    V[7:10, :] = r.uniform(0.01, .2, (3, 25))
    for q in [0, .03, .15, .5]:
        M, recovered, alpha = mix_at_share(B, V, q)
        assert abs(recovered - q) < 1e-10
        assert abs(observed_fraction(B, V, alpha, np.ones(SHAPE, bool)) - q) < 1e-10
        mask = np.ones(SHAPE, bool)
        mask[7:10] = False
        assert abs(observed_fraction(B, V, alpha, mask)) < 1e-12
        assert np.all(M >= B)
    print('OBSERVED-CELL TARGET & MIXING SELF-TEST: PASS')


def select_pairs(bg, vt, per_station, rng):
    pool = bg.loc[bg['split'].eq('validation')].reset_index(drop=True)
    stations = sorted(pool.station.unique())
    if len(stations) != 6:
        raise ValueError(f'Expected 6 development validation stations, got {stations}')
    chosen = sample_background(pool, per_station * len(stations), rng)
    counts = pool.iloc[chosen].station.value_counts()
    if not all(int(counts.get(s, 0)) == per_station for s in stations):
        raise ValueError('Failed balanced station selection')
    # Entire 4-minute sequences are essential to matched duration comparisons.
    sources = []
    for uid, g in vt.loc[vt.source_split.eq('validation')].groupby('clip_uid'):
        g = g.sort_values('minute_index')
        if len(g) < 4:
            continue
        if not np.array_equal(g.minute_index.to_numpy(), np.arange(len(g))):
            raise ValueError(f'Non-contiguous minute indexes in {uid}')
        if g.hull_id.nunique() != 1 or g.label.nunique() != 1:
            raise ValueError(f'Inconsistent source labels in {uid}')
        sources.append((uid, str(g.hull_id.iloc[0]), str(g.label.iloc[0]), g[BANDS].to_numpy(float)))
    if not sources:
        raise ValueError('No validation source clips have >=4 continuous full minutes')
    return pool, chosen, sources, stations


def build(args, rng):
    bg = pd.read_csv(args.background)
    vt = pd.read_csv(args.vessels, dtype={'hull_id': 'string'})
    if not {'split', 'station', 'run_id', 'source_parquet', 'row_positions_json'} <= set(bg.columns):
        raise ValueError('Missing background columns')
    if not {'source_split', 'clip_uid', 'hull_id', 'label', 'minute_index', *BANDS} <= set(vt.columns):
        raise ValueError('Missing vessel source columns')
    pool, choices, sources, stations = select_pairs(bg, vt, args.per_station, rng)
    source_classes = sorted({x[2] for x in sources})
    source_by_class = {c: [x for x in sources if x[2] == c] for c in source_classes}
    rows = []
    X = []
    Y = []
    group_ids = []
    reader = ParquetRowReader()
    for k, idx in enumerate(choices):
        r = pool.iloc[idx]
        B = db_to_relative_energy(reader.read(str(r.source_parquet), json.loads(r.row_positions_json)))
        if B.shape != SHAPE or not np.isfinite(B).all() or (B <= 0).any():
            raise ValueError('Invalid background energy')
        c = source_classes[int(rng.integers(len(source_classes)))]
        uid, hull, vclass, source = source_by_class[c][int(rng.integers(len(source_by_class[c])))]
        offset = int(rng.integers(len(source) - 3))
        # Same source 4-minute segment and start for every duration and share.
        insert = int(rng.integers(0, 12))
        group = f'pair_{k:04d}'
        scenario = [('zero', 0, 0.)]
        scenario += [('positive', d, q) for d in (1, 2, 3, 4)
                     for q in (.03, .15, .5)]
        for kind, duration, target in scenario:
            V = np.zeros(SHAPE, dtype=np.float64)
            if duration:
                V[insert:insert+duration] = source[offset:offset+duration]
                M, actual, alpha = mix_at_share(B, V, target)
            else:
                M, actual, alpha = B.copy(), 0., 0.
            if abs(actual - target) > 1e-9:
                raise ValueError('Target not recovered')
            X.append(10 * np.log10(np.maximum(M, 1e-30)))
            Y.append((B, V, alpha))
            group_ids.append(group)
            rows.append({'example': len(rows), 'pair': group, 'station': str(r.station),
                         'background_run_id': str(r.run_id), 'source_clip_uid': uid,
                         'vessel_hull_id': hull, 'vessel_class': vclass,
                         'source_start_min': offset, 'inserted_start_min': insert,
                         'event_duration': duration, 'requested_share': target,
                         'actual_full_share': actual, 'kind': kind})
        print(f'Matched background/source pair {k+1}/{len(choices)}', flush=True)
    return np.stack(X), Y, pd.DataFrame(rows), stations, sources


def fit_models(synthetic, seed):
    Xtr, ytr, ptr, _ = load_partition(synthetic, 'train')
    Xva, yva, pva, _ = load_partition(synthetic, 'validation')
    audit_overlap(ptr, pva)
    models = {}
    for name, trans in [('submitted_max_shift', normalize),
                        ('synthetic_original', lambda x: np.asarray(x, float))]:
        features = feature_families(trans(Xtr))['combined_96']
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                   random_state=seed, n_jobs=-1)
        rf.fit(features, ytr)
        val_pred = np.clip(rf.predict(feature_families(trans(Xva))['combined_96']), 0, 1)
        val_rmse = float(np.sqrt(mean_squared_error(yva, val_pred)))
        models[name] = (rf, trans, val_rmse)
    print('Development validation RMSE (already-used Experiment 002 set):',
          {k: round(v[2], 6) for k,v in models.items()})
    return models


def evaluate(args):
    tests()
    rng = np.random.default_rng(args.seed)
    models = fit_models(args.synthetic, args.model_seed)
    X, energy, meta, stations, source_pool = build(args, rng)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for mask_name, missing_min in masks().items():
        observed = np.ones(SHAPE, dtype=bool)
        observed[missing_min, :] = False
        # Correct target is computed over OBSERVED cells, not the original target.
        y_obs = np.asarray([observed_fraction(B, V, a, observed) for B, V, a in energy])
        corrupted = X.copy()
        if missing_min:
            corrupted[:, missing_min, :] = np.nan
            X_used, _ = impute_missing_minutes(corrupted)
        else:
            X_used = corrupted
        for name, (model, transform, _) in models.items():
            feat = feature_families(transform(X_used))['combined_96']
            pred = np.clip(model.predict(feat), 0, 1)
            for i in range(len(X)):
                result_rows.append({'example': i, 'mask': mask_name, 'model': name,
                                    'observed_share': y_obs[i], 'prediction': float(pred[i])})
    predictions = pd.DataFrame(result_rows).merge(meta, on='example', validate='many_to_one')
    if not np.isfinite(predictions[['observed_share', 'prediction']].to_numpy()).all():
        raise ValueError('Nonfinite predictions')
    predictions.to_csv(out/'paired_predictions.csv', index=False)
    summaries = []
    for (model, mask_name), df in predictions.groupby(['model','mask']):
        y = df.observed_share.to_numpy(float)
        p = df.prediction.to_numpy(float)
        present = (y > 1e-12).astype(int)
        # A positive controlled event may have been fully masked: then observed share=0.
        auc = float(roc_auc_score(present, p)) if len(np.unique(present)) == 2 else None
        summaries.append({'model': model, 'mask': mask_name, 'n': len(df),
                          'unique_pairs': df.pair.nunique(),
                          'rmse': float(np.sqrt(mean_squared_error(y, p))),
                          'mae': float(mean_absolute_error(y,p)),
                          'bias': float(np.mean(p-y)), 'roc_auc': auc,
                          'observed_zero': int((present == 0).sum()),
                          'observed_positive': int((present == 1).sum())})
    summary = pd.DataFrame(summaries).sort_values(['mask','model'])
    summary.to_csv(out/'summary.csv', index=False)
    config = {'experiment': '003-matched-development-stress', 'seed': args.seed,
              'model_seed': args.model_seed, 'pairs_per_station': args.per_station,
              'validation_sources_with_4plus_minutes': len(source_pool),
              'validation_stations': stations,
              'shares': [0,.03,.15,.5], 'durations': [1,2,3,4],
              'missing_masks': masks(),
              'target_policy': 'added energy / total mixture energy over OBSERVED cells',
              'limitations': [
                  'Source validation partition already examined in Experiment 002: not an independent final test',
                  'Four-minute source eligibility restricts the vessel population',
                  'VTUAD relative power is not physically calibrated to South African levels',
                  'All predictions are generated from controlled synthetic mixtures, not actual competition labels',
                  'Missing-minute interpolation is a prediction policy; target is computed BEFORE imputation'],
              'input_sha256': {str(p): checksum(p) for p in [args.background, args.vessels,
                  args.synthetic/'train.npz', args.synthetic/'validation.npz']}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('\nMATCHED ROBUSTNESS RESULTS (development synthetic data only)')
    print(summary.round(6).to_string(index=False))
    print('\nReport directory:', out)
    print('NO hidden-test measurements or labels used; NO leaderboard submission generated.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background', type=Path, default=Path('reports/eda/background_split_manifest_v2.csv'))
    p.add_argument('--vessels', type=Path, default=Path('reports/eda/vtuad_minute_band_power_split_v2.csv'))
    p.add_argument('--synthetic', type=Path, default=Path('reports/eda/synthetic_v2'))
    p.add_argument('--output', type=Path, default=Path('reports/robustness/phase3_matched_v1'))
    p.add_argument('--seed', type=int, default=314159)
    p.add_argument('--model-seed', type=int, default=42)
    p.add_argument('--per-station', type=int, default=4)
    p.add_argument('--self-test-only', action='store_true')
    args=p.parse_args()
    if not 1 <= args.per_station <= 100:
        p.error('Choose --per-station between 1 and 100')
    if args.self_test_only:
        tests()
    else:
        evaluate(args)

if __name__=='__main__':
    main()
