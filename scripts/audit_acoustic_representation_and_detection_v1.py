#!/usr/bin/env python3
"""Read-only Phase 3 audit: official band grid, representation, and synthetic detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error

BAND_CENTRES = np.array([
    31.6, 39.8, 50.1, 63.1, 79.4, 100.0, 125.9, 158.5, 199.5,
    251.2, 316.2, 398.1, 501.2, 631.0, 794.3, 1000.0, 1258.9,
    1584.9, 1995.3, 2511.9, 3162.3, 3981.1, 5011.9, 6309.6, 7943.3
], dtype=float)
PATTERN = re.compile(r'^m(\d{2})_TOL_([\d.]+)$')


def digest(path: Path) -> str:
    obj = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            obj.update(block)
    return obj.hexdigest()


def band_audit(cols: list[str]) -> pd.DataFrame:
    if len(cols) != 375 or len(set(cols)) != 375:
        raise ValueError(f'Expected 375 unique acoustic feature columns; got {len(cols)}')
    rows = []
    expected = [f'm{m:02d}_TOL_{c:g}' for m in range(1, 16) for c in BAND_CENTRES]
    # Exact names/order checked so the audit catches any accidental reshape/permutation.
    if cols != expected:
        first = next(((i, x, y) for i, (x, y) in enumerate(zip(cols, expected)) if x != y), None)
        raise ValueError(f'Test column order differs from expected minute-major grid: {first}')
    for j, name in enumerate(cols):
        match = PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f'Unparseable column: {name}')
        minute, center = int(match.group(1)), float(match.group(2))
        expected_center = float(BAND_CENTRES[j % 25])
        if minute != j // 25 + 1 or not math.isclose(center, expected_center, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'Unexpected minute/band: {name}')
        # Base-10 1/10-decade band spacing, half-width 0.05 decade.
        half = 10 ** 0.05
        rows.append(dict(column=name, minute=minute, center_hz=center,
                         nominal_lower_edge_hz=center / half,
                         nominal_upper_edge_hz=center * half))
    ratios = BAND_CENTRES[1:] / BAND_CENTRES[:-1]
    if not np.allclose(ratios, 10 ** 0.1, rtol=0.003, atol=0):
        raise ValueError('Band centers do not approximate base-10 decidecade spacing')
    return pd.DataFrame(rows)


def inspect_source_policy(generator: Path, inference: Path) -> dict:
    result = {}
    for label, path in [('generator', generator), ('inference', inference)]:
        if not path.is_file():
            result[label] = {'found': False, 'path': str(path)}
            continue
        src = path.read_text(encoding='utf-8')
        result[label] = {'found': True, 'path': str(path), 'sha256': digest(path),
                         'has_window_max_shift': bool(
                             ('np.max(levels)' in src and 'levels - ref' in src)
                             if label == 'generator' else
                             ('X.max(axis=(1,2)' in src or 'X.max(axis=(1, 2)' in src)),
                         'has_relative_mix': ('mix_at_share' in src if label == 'generator' else None)}
    return result


def mathematical_checks() -> dict:
    # These are tests of the formula, not tests of physical calibration or domain transfer.
    rng = np.random.default_rng(9182)
    levels = rng.normal(loc=91.0, scale=13.0, size=(15, 25))
    def transform(x):
        return np.power(10.0, (x - x.max()) / 10.0)
    B = transform(levels)
    V = np.zeros((15, 25), dtype=float)
    V[7:10] = rng.lognormal(mean=-7.0, sigma=0.8, size=(3, 25))
    shift_err = float(np.max(np.abs(B - transform(levels + 32.0))))
    # The generator intentionally makes backgrounds invariant to any uniform dB offset.
    if shift_err > 1e-12:
        raise AssertionError('Per-window shift invariance does not hold')
    errs = {}
    for q in [0.0, 0.005, 0.05, 0.3, 0.8]:
        alpha = 0.0 if q == 0.0 else (q / (1.0 - q)) * B.sum() / V.sum()
        energy_added = alpha * V
        measured = float(energy_added.sum() / (B + energy_added).sum())
        errs[str(q)] = abs(q - measured)
    if max(errs.values()) > 1e-12:
        raise AssertionError('Synthetic fraction recovery failed')
    return {
        'max_background_energy_difference_after_global_plus32dB': shift_err,
        'fraction_recovery_absolute_errors': errs,
        'interpretation': 'Correct algebra and deliberate invariance to absolute log-level offset; NOT acoustic calibration.',
    }


def missingness_mask_only(test_path: Path, cols: list[str]):
    # Read Test.csv only to audit the *location* of missing observations; never
    # use a finite acoustic value to fit/choose a transform or change a model.
    df = pd.read_csv(test_path, usecols=cols)
    X = df[cols].to_numpy(dtype=float).reshape(-1, 15, 25)
    if np.isinf(X).any():
        raise ValueError('Infinite test feature(s)')
    present = np.isfinite(X)
    minute_observed = present.sum(axis=2)
    partial = (minute_observed > 0) & (minute_observed < 25)
    mask = ~present
    missing_by_minute = mask.sum(axis=(0, 2)).astype(int)
    complete = np.all(present, axis=(1, 2))
    results = {
        'rows': int(X.shape[0]),
        'n_features_per_row': 375,
        'missing_cells': int(mask.sum()),
        'rows_with_any_missing_cells': int((~complete).sum()),
        'rows_with_all_cells_missing': int((~present.any(axis=(1, 2))).sum()),
        'partially_observed_minute_profiles': int(partial.sum()),
        'missing_full_minute_profiles': int((minute_observed == 0).sum()),
        'missing_cells_by_minute': missing_by_minute.tolist(),
        'rows_missing_first_minute': int((minute_observed[:, 0] == 0).sum()),
        'rows_missing_last_minute': int((minute_observed[:, -1] == 0).sum()),
        'policy': 'Mask counts only. No analysis of test acoustic values, fitting, or calibration.',
    }
    return results, pd.DataFrame({'minute': np.arange(1, 16), 'missing_cells': missing_by_minute,
                                   'rows_full_minute_missing': (minute_observed == 0).sum(axis=0).astype(int)})


def validation_detection(pred_path: Path, provenance_path: Path):
    predictions = pd.read_csv(pred_path)
    provenance = pd.read_csv(provenance_path)
    needed = {'example', 'actual_share', 'rf_combined_96', 'station', 'target_regime'}
    if not needed.issubset(predictions.columns):
        raise ValueError(f'Validation predictions missing {sorted(needed - set(predictions.columns))}')
    if not {'example', 'vessel_hull_id', 'background_run_id'}.issubset(provenance.columns):
        raise ValueError('Missing provenance source IDs')
    if predictions.example.duplicated().any() or provenance.example.duplicated().any():
        raise ValueError('Duplicated example IDs')
    merged = predictions.merge(provenance[['example', 'vessel_hull_id', 'background_run_id']],
                               on='example', validate='one_to_one', how='left', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Missing source provenance for validation predictions')
    y = merged.actual_share.to_numpy(float)
    p = merged.rf_combined_96.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.all((y >= 0) & (y <= 1)):
        raise ValueError('Non-finite or out-of-range synthetic validation values')
    detector_label = (y > 0).astype(int)
    if detector_label.min() == detector_label.max():
        raise ValueError('Must have both zero and positive simulated additions')
    # This is ROC AUC on SYNTHETIC validation, not ROC AUC on Zindi hidden labels.
    overall_auc = float(roc_auc_score(detector_label, p))
    ap = float(average_precision_score(detector_label, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    groups = []
    for station, block in merged.groupby('station', sort=True):
        yi = (block.actual_share.to_numpy(float) > 0).astype(int)
        pi = block.rf_combined_96.to_numpy(float)
        groups.append({'station': station, 'n': len(block), 'zero': int((yi == 0).sum()),
                       'positive': int((yi == 1).sum()),
                       'synthetic_roc_auc': float(roc_auc_score(yi, pi)) if len(np.unique(yi)) == 2 else np.nan,
                       'synthetic_rmse': float(np.sqrt(mean_squared_error(block.actual_share, pi))),
                       'background_runs': int(block.background_run_id.nunique()),
                       'vessel_hulls': int(block.vessel_hull_id.nunique())})
    return {'n': len(y), 'positive_n': int(detector_label.sum()),
            'zero_n': int((detector_label == 0).sum()), 'share_rmse': rmse,
            'share_as_uncalibrated_detection_score_roc_auc': overall_auc,
            'share_as_uncalibrated_detection_score_average_precision': ap,
            'caution': 'Synthetic validation only, from the same simulator; not a hidden-test AUC or an independent model.'}, pd.DataFrame(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, default=Path('data/zindi/raw/Test.csv'))
    parser.add_argument('--predictions', type=Path, default=Path('reports/baselines/synthetic_v2/validation_predictions.csv'))
    parser.add_argument('--provenance', type=Path, default=Path('reports/eda/synthetic_v2/validation_provenance.csv'))
    parser.add_argument('--generator', type=Path, default=Path('scripts/build_synthetic_mixtures_v1.py'))
    parser.add_argument('--inference', type=Path, default=Path('scripts/prepare_zindi_first_submission.py'))
    parser.add_argument('--out', type=Path, default=Path('reports/audits/phase3_representation_detection_v1'))
    args = parser.parse_args()
    if not args.test.is_file():
        raise FileNotFoundError(f'Competition Test.csv not found: {args.test}')
    if not args.predictions.is_file() or not args.provenance.is_file():
        raise FileNotFoundError('Experiment 002 validation prediction/provenance files not found')

    header = pd.read_csv(args.test, nrows=0).columns.tolist()
    if header[0] != 'ID':
        raise ValueError('First competition column must be ID')
    bands = band_audit(header[1:])
    mask_stats, missing_by_minute = missingness_mask_only(args.test, header[1:])
    source = inspect_source_policy(args.generator, args.inference)
    math_checks = mathematical_checks()
    synthetic, by_station = validation_detection(args.predictions, args.provenance)

    args.out.mkdir(parents=True, exist_ok=True)
    bands.loc[bands.minute == 1].to_csv(args.out / 'band_centres_and_nominal_edges.csv', index=False)
    missing_by_minute.to_csv(args.out / 'test_missingness_mask_by_minute.csv', index=False)
    by_station.to_csv(args.out / 'synthetic_detection_by_station.csv', index=False)
    summary = {
        'audit_version': 'phase3_representation_detection_v1',
        'data_access': 'Only Test.csv column names and missingness locations; all detection labels from synthetic validation',
        'test_schema_sha256': hashlib.sha256(','.join(header).encode()).hexdigest(),
        'input_hashes': {'test': digest(args.test), 'predictions': digest(args.predictions),
                         'provenance': digest(args.provenance)},
        'bands': {'n': 25, 'minutes': 15, 'spacing': 'nominal base-10 decidecade; centres rounded',
                  'nominal_lower_upper_edges': 'centre * 10**(-0.05), centre * 10**(+0.05)',
                  'not_a_claim_of_exact_instrument_filter_edges': True},
        'test_missingness_mask': mask_stats,
        'source_policy': source,
        'mathematical_checks': math_checks,
        'synthetic_detection': synthetic,
        'open_questions': [
            'Does VTUAD WAV band extraction use exactly compatible band-edge and PSD integration conventions?',
            'How should missing test minute profiles enter both feature computation and observed-cell target fraction?',
            'Are training synthetic event scales and source-reference spectra transferable to challenge test construction?',
            'Does window maximum normalization remove informative physical sound-level differences?',
            'How do performance and AUC change under independently constructed session/hull grouped simulations?',
        ],
        'no_model_fitted': True,
        'no_submission_written': True,
        'causal_claims_made': False,
    }
    (args.out / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('PHASE 3 ACOUSTIC REPRESENTATION & DETECTION AUDIT: PASS')
    print(f'Bands: {summary["bands"]["n"]}; minute positions: {summary["bands"]["minutes"]}; exact column order: PASS')
    print('Test mask only: missing cells =', mask_stats['missing_cells'],
          '; affected rows =', mask_stats['rows_with_any_missing_cells'],
          '; partially observed minute profiles =', mask_stats['partially_observed_minute_profiles'])
    print('Source policy:', source)
    print('Synthetic validation only: n={n}, ROC_AUC={share_as_uncalibrated_detection_score_roc_auc:.6f}, '
          'AP={share_as_uncalibrated_detection_score_average_precision:.6f}, '
          'share_RMSE={share_rmse:.6f}'.format(**synthetic))
    print('\nSYNTHETIC DETECTION BY STATION:')
    print(by_station.round(5).to_string(index=False))
    print('\nAll output reports:', args.out)
    print('CAUTION: No hidden test labels used. No real-world validity or physical calibration established.')


if __name__ == '__main__':
    main()
