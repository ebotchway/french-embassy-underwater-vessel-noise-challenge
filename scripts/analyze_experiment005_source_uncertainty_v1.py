#!/usr/bin/env python3
"""Experiment 005: paired, source-aware descriptive uncertainty audit.

Reads ONLY the already-created training-only simulation-realism paired output.
Does not access the competition Test.csv, frozen eval, or sealed holdouts.
Does not refit, alter source data, or create a leaderboard submission.

Usage:
  python scripts/analyze_experiment005_source_uncertainty_v1.py --self-test-only
  python scripts/analyze_experiment005_source_uncertainty_v1.py

A cluster bootstrap over background RUNS is descriptive: vessel hulls/clips can
repeat across runs, and stations/sessions may still be dependent. Leave-one-
source-out checks are provided; none of this establishes a real-vessel label.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path('reports/audits/simulation_realism_v1/paired_temporal_changes.csv')
OUTPUT = Path('reports/audits/simulation_realism_uncertainty_v1')
ARMS = ('original_vtuad', 'halfway_shape', 'background_shaped_control')
PAIR_KEYS = ['station', 'background_run_id', 'requested_share', 'duration_min']
METRIC = 'change_in_max_step_rms_db'
REQUIRED = set(PAIR_KEYS + ['vessel_clip_uid', 'vessel_hull_id', 'spectral_intervention',
                            METRIC, 'natural_max_step_rms_db', 'injected_max_step_rms_db'])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    if df.empty or df[list(REQUIRED)].isna().any().any():
        raise ValueError('Empty data or missing required values')
    if set(df.spectral_intervention) != set(ARMS):
        raise ValueError('Unexpected or missing intervention arms')
    if not np.isfinite(df[[METRIC, 'natural_max_step_rms_db', 'injected_max_step_rms_db',
                            'requested_share', 'duration_min']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite numeric values')
    if df.duplicated(PAIR_KEYS + ['spectral_intervention']).any():
        raise ValueError('Duplicate paired interventions')
    g = df.groupby(PAIR_KEYS, dropna=False, sort=False)
    if not g['spectral_intervention'].nunique().eq(3).all():
        raise ValueError('Unmatched intervention set for at least one background run/scenario')
    for source in ['vessel_clip_uid', 'vessel_hull_id']:
        if not g[source].nunique().eq(1).all():
            raise ValueError(f'Unmatched source within intervention triplet: {source}')
    wide = df.pivot(index=PAIR_KEYS, columns='spectral_intervention', values=METRIC).reset_index()
    meta = df.loc[df.spectral_intervention == 'original_vtuad',
                  PAIR_KEYS + ['vessel_clip_uid', 'vessel_hull_id', 'natural_max_step_rms_db']]
    wide = wide.merge(meta, on=PAIR_KEYS, validate='one_to_one')
    wide['delta_original_vs_background'] = wide['original_vtuad'] - wide['background_shaped_control']
    wide['delta_original_vs_halfway'] = wide['original_vtuad'] - wide['halfway_shape']
    wide['delta_halfway_vs_background'] = wide['halfway_shape'] - wide['background_shaped_control']
    # Every background run should be exposed at the same scenarios; fail closed.
    scenarios = set(zip(wide.requested_share, wide.duration_min))
    per_run = wide.groupby('background_run_id', sort=False).apply(
        lambda x: set(zip(x.requested_share, x.duration_min)), include_groups=False)
    if not per_run.map(lambda x: x == scenarios).all():
        raise ValueError('Different scenario coverage across background runs')
    if wide.groupby('background_run_id').station.nunique().gt(1).any():
        raise ValueError('Background run ID appears across multiple stations')
    return wide


def clustered_ci(x, n_boot, seed):
    """Resample independent *run IDs*; one matched difference per run/scenario."""
    z = np.asarray(x, dtype=float)
    if len(z) < 2:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(z), size=(n_boot, len(z)))
    stats = np.median(z[samples], axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(lo), float(hi), float(np.mean(stats > 0))


def summarize(wide, bootstrap, seed):
    rows = []
    for (share, duration), frame in wide.groupby(['requested_share', 'duration_min'], sort=True):
        for metric in ['delta_original_vs_background','delta_original_vs_halfway',
                       'delta_halfway_vs_background']:
            # one row per run in every scenario
            if frame.background_run_id.duplicated().any():
                raise ValueError('One run repeated within a scenario')
            vals = frame[metric].to_numpy(float)
            lo, hi, positive = clustered_ci(vals, bootstrap, seed + len(rows))
            rows.append({'requested_share': float(share), 'duration_min': int(duration),
                         'comparison': metric, 'background_runs':len(frame),
                         'vessel_hulls':int(frame.vessel_hull_id.nunique()),
                         'vessel_clips':int(frame.vessel_clip_uid.nunique()),
                         'median_paired_difference':float(np.median(vals)),
                         'mean_paired_difference':float(np.mean(vals)),
                         'positive_fraction':float(np.mean(vals > 0)),
                         'run_cluster_bootstrap_median_low':lo,
                         'run_cluster_bootstrap_median_high':hi,
                         'run_cluster_bootstrap_fraction_positive':positive})
    return pd.DataFrame(rows)


def deletion_analysis(wide):
    out = []
    for (share, duration), frame in wide.groupby(['requested_share', 'duration_min'], sort=True):
        for metric in ['delta_original_vs_background', 'delta_original_vs_halfway']:
            full = float(np.median(frame[metric]))
            for type_, field in [('station','station'), ('background_run','background_run_id'),
                                 ('vessel_hull','vessel_hull_id'), ('vessel_clip','vessel_clip_uid')]:
                for group, removed in frame.groupby(field, sort=True):
                    remains = frame.loc[frame[field] != group]
                    if remains.empty:
                        continue
                    after = float(np.median(remains[metric]))
                    out.append({'requested_share':float(share),'duration_min':int(duration),
                                'comparison':metric,'leave_out_type':type_,
                                'removed_group':str(group),'removed_runs':len(removed),
                                'remaining_runs':len(remains),
                                'full_median_difference':full,'after_deletion_median_difference':after,
                                'sign_reversal': bool(full * after < 0)})
    return pd.DataFrame(out)


def analyse(input_path: Path, output: Path, bootstrap=2000, seed=20261122):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    data = pd.read_csv(input_path, dtype={'background_run_id':'string','vessel_clip_uid':'string',
                                           'vessel_hull_id':'string'})
    paired = prepare(data)
    summary = summarize(paired, bootstrap, seed)
    deletions = deletion_analysis(paired)
    deletion_summary = (deletions.groupby(['requested_share','duration_min','comparison','leave_out_type'],
                                          dropna=False)
                        .agg(groups=('removed_group','nunique'),
                             min_after=('after_deletion_median_difference','min'),
                             max_after=('after_deletion_median_difference','max'),
                             reversals=('sign_reversal','sum'),
                             max_removed_runs=('removed_runs','max')).reset_index())
    exposure = (paired.groupby(['station','vessel_hull_id'], dropna=False)
                .agg(background_runs=('background_run_id','nunique'),
                     clips=('vessel_clip_uid','nunique'),
                     rows=('background_run_id','size')).reset_index())
    station = (paired.groupby(['station','requested_share','duration_min'], dropna=False)
               .agg(background_runs=('background_run_id','nunique'),
                    vessel_hulls=('vessel_hull_id','nunique'),
                    median_original_vs_background=('delta_original_vs_background','median'),
                    median_original_vs_halfway=('delta_original_vs_halfway','median'),
                    fraction_original_above_background=('delta_original_vs_background',lambda x:float(np.mean(x>0))))
               .reset_index())
    manifest = {'audit':'experiment005_source_uncertainty_v1',
                'input':str(input_path), 'input_sha256':sha256(input_path),
                'source_rows':len(data), 'paired_scenario_rows':len(paired),
                'unique_background_runs':int(paired.background_run_id.nunique()),
                'unique_vessel_hulls':int(paired.vessel_hull_id.nunique()),
                'unique_vessel_clips':int(paired.vessel_clip_uid.nunique()),
                'stations':int(paired.station.nunique()),
                'seed':seed,'bootstrap_resamples':bootstrap,
                'metric':'paired change in max minute-to-minute 25-band RMS log-level step',
                'scope':'training-only, descriptive; no hidden test or new model',
                'limitations':[
                    'Run-cluster bootstrap does not account for vessel hull/clip or station cross-dependence',
                    'Repeated observations from the same recorder and recording deployment may remain dependent',
                    'Only one background run is available at Durban',
                    'Source-aligned spectra are counterfactual sensitivity controls, not calibrated vessel signals',
                    'Previously examined development sources, not an untouched external dataset']}
    note = [
        '# Experiment 005 — source-aware uncertainty checkpoint', '',
        f"Input: `{input_path}` (SHA-256 `{manifest['input_sha256']}`).", '',
        f"Matched intervention triplets: {len(paired)} across {manifest['unique_background_runs']} "
        f"background runs and {manifest['unique_vessel_hulls']} vessel hulls.", '',
        'Positive paired differences indicate that the first named intervention increases the '
        'maximum temporal-change statistic more than the second. Medians and percentiles are '
        'descriptive, not physically calibrated.', '',
        'Bootstrap confidence intervals resample *background runs* within each scenario. '
        'Because vessel sources repeat across runs and station/session effects persist, they '
        'are not fully source-independent intervals. Examine deletion sensitivity and station '
        'summaries before interpreting them.', '',
        'The file `paired_scenario_differences.csv` contains exact matched per-run values; '
        '`comparison_summary.csv` provides run-bootstrap intervals; '
        '`leave_one_source_out.csv` and `deletion_summary.csv` show sensitivity; '
        '`station_scenario_summary.csv` and `station_hull_exposure.csv` show coverage.', '',
        'Do not use this analysis to identify natural vessel presence or calibrate true hydrophone pressure.',
        'Do not use the sealed evaluation or leaderboard to select a winning spectral intervention.', ''
    ]
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.exp005_stage_',dir=output.parent) as temp:
        stage = Path(temp)/'completed'
        stage.mkdir()
        paired.to_csv(stage/'paired_scenario_differences.csv',index=False)
        summary.to_csv(stage/'comparison_summary.csv',index=False)
        deletions.to_csv(stage/'leave_one_source_out.csv',index=False)
        deletion_summary.to_csv(stage/'deletion_summary.csv',index=False)
        station.to_csv(stage/'station_scenario_summary.csv',index=False)
        exposure.to_csv(stage/'station_hull_exposure.csv',index=False)
        (stage/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (stage/'RESEARCH_NOTE.md').write_text('\n'.join(note))
        os.rename(stage,output)
    print('EXPERIMENT 005 SOURCE UNCERTAINTY: PASS')
    print('Intervention triplets:',len(paired),'background runs:',manifest['unique_background_runs'],
          'vessel hulls:',manifest['unique_vessel_hulls'],'clips:',manifest['unique_vessel_clips'])
    print('\nPAIRWISE COMPARISON (per scenario):')
    print(summary[['requested_share','duration_min','comparison','median_paired_difference',
                   'run_cluster_bootstrap_median_low','run_cluster_bootstrap_median_high',
                   'positive_fraction']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nSOURCE DELETION REVERSALS:')
    print(deletion_summary.loc[deletion_summary.reversals.gt(0)].to_string(index=False))
    print('Output:',output)
    print('CAUTION: Run bootstrap does not remove shared-hull, shared-clip or station dependence.')


def self_test():
    rng = np.random.default_rng(42)
    rows=[]
    for i in range(8):
        for q in (0.03,0.15):
            for d in (1,3):
                base=float(rng.uniform(0.1,3))
                for arm,offset in [('original_vtuad',3),('halfway_shape',1),('background_shaped_control',0)]:
                    rows.append({'station':f'S{i%3}','background_run_id':f'run{i}',
                                 'requested_share':q,'duration_min':d,
                                 'vessel_clip_uid':f'clip{i%5}', 'vessel_hull_id':f'hull{i%4}',
                                 'spectral_intervention':arm,'natural_max_step_rms_db':base,
                                 'injected_max_step_rms_db':base+offset,
                                 'change_in_max_step_rms_db':float(offset)})
    df=pd.DataFrame(rows)
    import tempfile as tmp
    with tmp.TemporaryDirectory() as path:
        source=Path(path)/'source.csv'; target=Path(path)/'report'
        df.to_csv(source,index=False)
        analyse(source,target,bootstrap=100,seed=5)
        result=pd.read_csv(target/'comparison_summary.csv')
        x=result.loc[result.comparison.eq('delta_original_vs_background')]
        assert len(x)==4 and np.allclose(x.median_paired_difference,3)
        assert (target/'audit_manifest.json').exists()
        assert len(pd.read_csv(target/'paired_scenario_differences.csv'))==32
        try:
            analyse(source,target,bootstrap=100)
        except FileExistsError:
            pass
        else: raise AssertionError('Refuse-overwrite safeguard failed')
        bad=df.iloc[:-1]
        try: prepare(bad)
        except ValueError: pass
        else: raise AssertionError('Unmatched scenarios were accepted')
    print('EXPERIMENT 005 SOURCE UNCERTAINTY SELF-TEST: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=INPUT)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only: self_test()
    else: analyse(a.input,a.output,a.bootstrap,a.seed)

if __name__=='__main__':main()
