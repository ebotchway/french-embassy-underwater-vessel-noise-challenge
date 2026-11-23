#!/usr/bin/env python3
"""Experiment 002: fixed, predeclared acoustic summaries vs reference baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_synthetic_baselines_v1 import load_partition, audit_overlap, checksum


def feature_families(X):
    """Frozen before seeing Experiment 002 validation outcomes."""
    flat = X.reshape(len(X), -1)
    global_ = np.column_stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2)),
                               X.min(axis=(1, 2)), X.max(axis=(1, 2))])
    spectral = np.column_stack([X.mean(axis=1), X.std(axis=1)])
    difference = np.diff(X, axis=1)
    temporal = np.column_stack([difference.mean(axis=2), difference.std(axis=2),
                                np.abs(difference).mean(axis=2)])
    combined = np.column_stack([global_, spectral, temporal])
    assert flat.shape[1] == 375 and spectral.shape[1] == 50 and combined.shape[1] == 96
    return {'raw_375': flat, 'spectral_50': spectral, 'combined_96': combined}


def score(y, p):
    return {'rmse': float(np.sqrt(mean_squared_error(y, p))),
            'mae': float(mean_absolute_error(y, p)),
            'bias': float(np.mean(p-y)),
            'prediction_min': float(p.min()), 'prediction_max': float(p.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('reports/eda/synthetic_v2'))
    p.add_argument('--output', type=Path, default=Path('reports/baselines/synthetic_v2'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    Xtr, ytr, ptr, hashes_tr = load_partition(args.input, 'train')
    Xva, yva, pva, hashes_va = load_partition(args.input, 'validation')
    audit_overlap(ptr, pva)
    if len(Xtr) < 30 or len(Xva) < 15:
        raise ValueError('At least 30 train / 15 validation examples required')
    if not {'target_regime', 'inserted_start_min'} <= set(ptr.columns) & set(pva.columns):
        raise ValueError('Not an Experiment 002 provenance manifest')
    if 'experiment_config.json' not in {x.name for x in args.input.iterdir()}:
        raise ValueError('Experiment configuration missing')
    config = json.loads((args.input / 'experiment_config.json').read_text())
    if config.get('experiment') != '002':
        raise ValueError('Expected Experiment 002 configuration')

    print('EXPERIMENT 002 INTEGRITY: PASS')
    print('Training:', Xtr.shape, 'Validation:', Xva.shape)
    print('Cross-partition run, vessel clip and hull overlap: 0')
    ftr, fva = feature_families(Xtr), feature_families(Xva)
    # Models are fixed BEFORE reading validation performance; no tuning loop.
    models = [
        ('constant_train_mean', 'raw_375', DummyRegressor(strategy='mean')),
        ('rf_raw_375', 'raw_375', RandomForestRegressor(n_estimators=200,
             min_samples_leaf=2, random_state=args.seed, n_jobs=-1)),
        ('rf_spectral_50', 'spectral_50', RandomForestRegressor(n_estimators=200,
             min_samples_leaf=2, random_state=args.seed, n_jobs=-1)),
        ('rf_combined_96', 'combined_96', RandomForestRegressor(n_estimators=200,
             min_samples_leaf=2, random_state=args.seed, n_jobs=-1)),
    ]
    predictions = pva[['example', 'station', 'vessel_class', 'vessel_duration_min',
                       'requested_share', 'target_regime']].copy()
    predictions['actual_share'] = yva
    metrics = []
    for name, features, model in models:
        model.fit(ftr[features], ytr)
        pred = np.clip(model.predict(fva[features]), 0, 1)
        predictions[name] = pred
        s = score(yva, pred)
        metrics.append({'model': name, **s})
        print(f'{name:22s} RMSE={s["rmse"]:.6f} MAE={s["mae"]:.6f}')

    rows = []
    for name in ['rf_raw_375', 'rf_spectral_50', 'rf_combined_96']:
        for grouping in ['station', 'target_regime']:
            for key, group in predictions.groupby(grouping, dropna=False):
                metric = score(group.actual_share.to_numpy(), group[name].to_numpy())
                rows.append({'model': name, 'group_type': grouping, 'group': str(key),
                             'n': len(group), **metric})
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output / 'metrics.csv', index=False)
    predictions.to_csv(args.output / 'validation_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(args.output / 'group_metrics.csv', index=False)
    run_info = {
        'experiment': '002', 'type': 'synthetic relative-energy simulation',
        'disclaimer': 'Not calibrated acoustic pressure and not a Zindi hidden-test score',
        'n_train': len(ytr), 'n_validation': len(yva), 'seed': args.seed,
        'predeclared_candidates': [m[0] for m in models],
        'model_selection_note': 'Compare prespecified models once; validation is not used for hyperparameter search',
        'inputs_sha256': {'train': hashes_tr, 'validation': hashes_va},
        'generator_config': config,
    }
    (args.output / 'run_info.json').write_text(json.dumps(run_info, indent=2) + '\n')
    print('\nRESULTS:')
    print(pd.DataFrame(metrics).sort_values('rmse').to_string(index=False))
    print('Reports:', args.output)
    print('EXPERIMENT 002 BASELINES: PASS (synthetic validation only)')


if __name__ == '__main__':
    main()
