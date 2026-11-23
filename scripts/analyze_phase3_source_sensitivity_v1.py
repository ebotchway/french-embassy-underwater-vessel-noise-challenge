#!/usr/bin/env python3
"""Read-only source-dependence sensitivity for Experiment 003."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('submitted_max_shift', 'synthetic_original')
MASKS = ('complete', 'first', 'last', 'middle', 'both_edges', 'three_middle')
COLUMNS = {'pair', 'mask', 'model', 'observed_share', 'prediction', 'station',
           'example', 'background_run_id', 'source_clip_uid'}


def rmse_advantage(frame):
    metrics = {}
    for model in MODELS:
        x = frame.loc[frame.model.eq(model)]
        if len(x) == 0:
            raise ValueError('Missing model in group')
        metrics[model] = float(np.sqrt(np.mean((x.prediction.to_numpy(float) -
                                                x.observed_share.to_numpy(float)) ** 2)))
    return metrics[MODELS[0]] - metrics[MODELS[1]], metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('reports/robustness/phase3_matched_8pairs/paired_predictions.csv'))
    p.add_argument('--output', type=Path, default=Path('reports/robustness/phase3_matched_8pairs/source_sensitivity'))
    args = p.parse_args()
    if args.input.resolve().is_relative_to(args.output.resolve()):
        raise ValueError('Input cannot be inside output directory')
    df = pd.read_csv(args.input, dtype={'background_run_id': 'string', 'source_clip_uid': 'string'})
    if missing := COLUMNS - set(df):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if set(df.model) != set(MODELS) or set(df['mask']) != set(MASKS):
        raise ValueError('Unexpected models or masks')
    if df.duplicated(['pair', 'mask', 'model', 'example']).any():
        raise ValueError('Duplicate prediction records')
    if not np.isfinite(df[['observed_share', 'prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite targets/predictions')
    if df.groupby(['pair', 'mask', 'model']).size().nunique() != 1:
        raise ValueError('Unequal number of scenarios per group')
    # Enforce pairing, stable source metadata, and target agreement across models.
    keys = ['pair', 'mask', 'example']
    wide = df.pivot(index=keys, columns='model', values='observed_share')
    if wide.isna().any().any() or not np.allclose(wide[MODELS[0]], wide[MODELS[1]], atol=1e-12, rtol=0):
        raise ValueError('Nonmatching examples or targets')
    metadata = ['station', 'background_run_id', 'source_clip_uid']
    for col in metadata:
        if df.groupby('pair')[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f'Inconsistent pair metadata: {col}')
    df = df.copy()
    df['background_run_key'] = df.station.astype(str) + '::' + df.background_run_id.fillna('UNKNOWN').astype(str)
    descriptions = []
    deletions = []
    source_stats = []
    for mask in MASKS:
        d = df.loc[df['mask'].eq(mask)]
        base, met = rmse_advantage(d)
        descriptions.append({'mask': mask, 'rows': len(d), 'pairs': d.pair.nunique(),
                             'background_runs': d.background_run_key.nunique(),
                             'source_clips': d.source_clip_uid.nunique(),
                             'baseline_advantage_original': base,
                             'rmse_submitted': met[MODELS[0]], 'rmse_original': met[MODELS[1]]})
        for group_col, group_name in [('station', 'station'),
                                      ('background_run_key', 'background_run'),
                                      ('source_clip_uid', 'vessel_source_clip')]:
            for source, group in d.groupby(group_col, dropna=False):
                keep = d.loc[~d.pair.isin(group.pair.unique())]
                if keep.pair.nunique() < 2:
                    continue
                advantage, _ = rmse_advantage(keep)
                deletions.append({'mask': mask, 'leave_out_type': group_name,
                                  'leave_out_id': str(source),
                                  'removed_pairs': int(group.pair.nunique()),
                                  'remaining_pairs': int(keep.pair.nunique()),
                                  'advantage_after_deletion': advantage,
                                  'delta_vs_full': advantage-base,
                                  'reverses_original_advantage': bool((base > 0) != (advantage > 0))})
                source_stats.append((mask, group_name, len(group), group.pair.nunique()))
    summary = pd.DataFrame(descriptions)
    deletion = pd.DataFrame(deletions)
    results = []
    for (mask, kind), group in deletion.groupby(['mask', 'leave_out_type']):
        results.append({'mask': mask, 'leave_out_type': kind,
                        'source_groups': len(group),
                        'min_advantage_after_deletion': float(group.advantage_after_deletion.min()),
                        'max_advantage_after_deletion': float(group.advantage_after_deletion.max()),
                        'reversal_count': int(group.reverses_original_advantage.sum()),
                        'max_removed_pairs': int(group.removed_pairs.max())})
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'baseline_by_mask.csv', index=False)
    deletion.to_csv(out/'leave_one_source_out.csv', index=False)
    pd.DataFrame(results).to_csv(out/'source_sensitivity_summary.csv', index=False)
    manifest = {'design': 'Retrospective leave-one-group-out deletion on already-seen validation source pool',
                'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'independent_confirmation': False,
                'caveats': ['Deleting one station/run/clip does not create an independent holdout',
                            'Background runs and source clips form crossed dependence; this is not a formal multiway cluster bootstrap',
                            'Do not tune using hidden test labels or provisional leaderboard',
                            'No statistical significance can be claimed from these deletion bounds']}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('SOURCE-DEPENDENCE SENSITIVITY: PASS')
    print('Pairs:', df.pair.nunique(), 'Predictions:', len(df))
    print('\nBASELINE')
    print(summary[['mask', 'pairs', 'background_runs', 'source_clips', 'baseline_advantage_original']].round(6).to_string(index=False))
    print('\nDELETION SENSITIVITY')
    print(pd.DataFrame(results).round(6).to_string(index=False))
    print('\nOutputs:', out)
    print('LIMIT: Descriptive source deletion only, not independent source-level validation.')

if __name__ == '__main__':
    main()
