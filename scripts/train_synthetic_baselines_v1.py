#!/usr/bin/env python3
"""Audit and evaluate simple baselines on simulated 15x25 acoustics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED_SHAPE = (15, 25)
PROVENANCE_REQUIRED = {
    'example', 'partition', 'station', 'background_run_id',
    'background_start', 'background_source', 'vessel_clip_uid',
    'vessel_hull_id', 'vessel_class', 'vessel_duration_min',
    'requested_share', 'recovered_share', 'relative_scale',
}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_partition(folder: Path, split: str):
    npz_path = folder / f'{split}.npz'
    provenance_path = folder / f'{split}_provenance.csv'
    with np.load(npz_path, allow_pickle=False) as z:
        if set(z.files) != {'X', 'y'}:
            raise ValueError(f'{npz_path}: expected arrays X and y, got {z.files}')
        X = np.asarray(z['X'], dtype=np.float64)
        y = np.asarray(z['y'], dtype=np.float64)
    p = pd.read_csv(provenance_path, keep_default_na=False)
    absent = PROVENANCE_REQUIRED - set(p.columns)
    if absent:
        raise ValueError(f'{provenance_path}: missing columns {sorted(absent)}')
    if X.ndim != 3 or X.shape[1:] != EXPECTED_SHAPE:
        raise ValueError(f'{split}: X shape must be (N,15,25), got {X.shape}')
    n = len(X)
    if n < 2 or y.shape != (n,) or len(p) != n:
        raise ValueError(f'{split}: inconsistent or insufficient example counts')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f'{split}: nonfinite X/y')
    if not np.all((y >= 0) & (y < 1)):
        raise ValueError(f'{split}: target outside [0,1)')
    if not p['partition'].eq(split).all():
        raise ValueError(f'{split}: incorrect provenance partition')
    example_ids = pd.to_numeric(p['example'], errors='raise').to_numpy()
    if not np.array_equal(example_ids, np.arange(n)):
        raise ValueError(f'{split}: provenance does not align with array rows')
    for column in ('requested_share', 'recovered_share'):
        q = pd.to_numeric(p[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(q).all() or not np.allclose(y, q, atol=1e-6, rtol=0):
            raise ValueError(f'{split}: targets disagree with {column}')
    background_key = p['background_source'].astype(str) + '|' + p['background_start'].astype(str)
    if background_key.duplicated().any():
        raise ValueError(f'{split}: repeated background windows within partition')
    vessel_id = p['vessel_hull_id'].astype(str).str.strip()
    vessel_clip = p['vessel_clip_uid'].astype(str).str.strip()
    no_addition = np.isclose(y, 0.0, atol=1e-8)
    if np.any(no_addition & vessel_clip.ne('').to_numpy()):
        raise ValueError(f'{split}: zero addition has a vessel source')
    if np.any((~no_addition) & (vessel_id.eq('') | vessel_clip.eq('')).to_numpy()):
        raise ValueError(f'{split}: nonzero addition missing vessel identity')
    if np.any(no_addition & (pd.to_numeric(p['vessel_duration_min']) != 0).to_numpy()):
        raise ValueError(f'{split}: zero addition has a nonzero vessel duration')
    if np.any((~no_addition) & (pd.to_numeric(p['vessel_duration_min']) <= 0).to_numpy()):
        raise ValueError(f'{split}: nonzero addition has nonpositive vessel duration')
    # Nonnegative is NOT required: X is log10 relative energy and may be negative.
    return X, y, p, { 'npz': checksum(npz_path), 'provenance': checksum(provenance_path) }


def audit_overlap(a: pd.DataFrame, b: pd.DataFrame) -> None:
    for column in ('background_run_id', 'vessel_hull_id', 'vessel_clip_uid'):
        x = set(a[column].astype(str).str.strip()) - {''}
        y = set(b[column].astype(str).str.strip()) - {''}
        intersect = x & y
        if intersect:
            raise ValueError(f'Cross-partition leakage: {column}, {len(intersect)} overlapping values')
    # Also assert exact source-time windows are not reused between sets.
    key = lambda p: set(zip(p['background_source'].astype(str), p['background_start'].astype(str)))
    if key(a) & key(b):
        raise ValueError('Cross-partition leakage: exact background window reused')


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        'rmse': float(np.sqrt(mean_squared_error(y, pred))),
        'mae': float(mean_absolute_error(y, pred)),
        'bias': float(np.mean(pred - y)),
        'prediction_min': float(np.min(pred)),
        'prediction_max': float(np.max(pred)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('reports/eda/synthetic_v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/baselines/synthetic_v1'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    Xtr, ytr, ptr, hash_tr = load_partition(args.input, 'train')
    Xva, yva, pva, hash_va = load_partition(args.input, 'validation')
    audit_overlap(ptr, pva)

    print('SYNTHETIC DATA INTEGRITY: PASS')
    print(f'Train: {Xtr.shape}, Validation: {Xva.shape}')
    print(f'Target train: min={ytr.min():.4f} mean={ytr.mean():.4f} max={ytr.max():.4f}')
    print(f'Target valid: min={yva.min():.4f} mean={yva.mean():.4f} max={yva.max():.4f}')
    print('Cross-partition background run / vessel clip / hull overlap: 0')

    # IMPORTANT: models never receive provenance, q, hull IDs, or station IDs.
    tr = Xtr.reshape(len(Xtr), -1)
    va = Xva.reshape(len(Xva), -1)
    models = {
        'constant_train_mean': DummyRegressor(strategy='mean'),
        'ridge_alpha_10': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'random_forest_200': RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, max_features=0.75,
            random_state=args.seed, n_jobs=-1,
        ),
    }
    records = []
    predictions = pva[['example', 'station', 'vessel_class', 'vessel_duration_min', 'requested_share']].copy()
    predictions['actual_share'] = yva
    for name, model in models.items():
        model.fit(tr, ytr)
        raw = np.asarray(model.predict(va), dtype=float)
        pred = np.clip(raw, 0.0, 1.0)
        metric = scores(yva, pred)
        records.append({'model': name, **metric})
        predictions[name] = pred
        print(f'{name:22s} RMSE={metric["rmse"]:.6f} MAE={metric["mae"]:.6f} '
              f'prediction_range=[{metric["prediction_min"]:.4f}, {metric["prediction_max"]:.4f}]')

    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError('Nonfinite outputs')
    metrics = pd.DataFrame(records).sort_values('rmse', kind='stable')
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    predictions.to_csv(args.output / 'validation_predictions.csv', index=False)
    # Report per simulated fraction and station for the best *listed* model;
    # do not use this small validation set for a hyperparameter search.
    best = str(metrics.iloc[0]['model'])
    def grouped_errors(column):
        result = []
        for label, group in predictions.groupby(column, dropna=False):
            err = group[best].to_numpy() - group['actual_share'].to_numpy()
            result.append({column: label, 'n': len(group),
                           'rmse': float(np.sqrt(np.mean(err**2)))})
        return pd.DataFrame(result)

    grouped_errors('requested_share').to_csv(args.output / 'best_model_by_target.csv', index=False)
    grouped_errors('station').to_csv(args.output / 'best_model_by_station.csv', index=False)
    report = {
        'description': 'Relative-energy simulation only; not a Zindi hidden-test score',
        'n_train': len(Xtr), 'n_validation': len(Xva),
        'seed': args.seed, 'input_hashes': {'train': hash_tr, 'validation': hash_va},
        'feature_shape': [15, 25],
        'primary_baseline': 'constant_train_mean',
        'best_of_three_on_this_validation': best,
        'note': ('Models use only 375 flattened acoustic values. Predictions clipped to [0,1]. '
                 'Mixture target values were assigned by a simulator, and the vessel reference '
                 'contains ambient sound. Results are not calibrated to real measurements.'),
    }
    (args.output / 'run_info.json').write_text(json.dumps(report, indent=2) + '\n')
    print('\nSUMMARY (synthetic validation only):')
    print(metrics.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'\nSaved reports to {args.output}')
    print('CAUTION: Do not interpret these scores as competition performance.')


if __name__ == '__main__':
    main()
