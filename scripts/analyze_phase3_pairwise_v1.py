#!/usr/bin/env python3
"""Pair-aware retrospective analysis of Experiment 003 development predictions.

No refitting, no Test.csv, no leaderboard inputs. Treat matched background/source
pair (not repeated scenario row) as the resampling cluster. Results are
DESCRIPTIVE development evidence, not independent confirmation of transfer.
Run from project root:
 python scripts/analyze_phase3_pairwise_v1.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('submitted_max_shift', 'synthetic_original')
MASKS = ('complete','first','last','middle','both_edges','three_middle')
REQUIRED = {'pair','mask','model','observed_share','prediction',
            'requested_share','event_duration','station','example',
            'background_run_id','source_clip_uid'}

def check(df):
    absent = REQUIRED-set(df.columns)
    if absent: raise ValueError(f'Missing required columns: {sorted(absent)}')
    if set(df.model)!=set(MODELS): raise ValueError('Unexpected models')
    if set(df['mask'])!=set(MASKS): raise ValueError('Unexpected missingness masks')
    if df.duplicated(['pair','example','mask','model']).any():
        raise ValueError('Duplicate scenario/model/mask rows')
    if not np.isfinite(df[['observed_share','prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite values')
    if not df[['observed_share','prediction']].ge(0).all().all() or not df[['observed_share','prediction']].le(1).all().all():
        raise ValueError('Out-of-range values')
    for (pair, mask), g in df.groupby(['pair','mask']):
        a=g[g.model==MODELS[0]].sort_values('example')
        b=g[g.model==MODELS[1]].sort_values('example')
        if len(a)!=13 or len(b)!=13 or not np.array_equal(a.example.to_numpy(),b.example.to_numpy()):
            raise ValueError(f'Unpaired/incomplete 13 scenarios: {pair}/{mask}')
        if not np.allclose(a.observed_share.to_numpy(),b.observed_share.to_numpy(),atol=1e-12,rtol=0):
            raise ValueError(f'Inconsistent target across models: {pair}/{mask}')
    if df.groupby('pair').station.nunique().gt(1).any():
        raise ValueError('Pair spans multiple stations')
    return int(df.pair.nunique())

def boot_for_mask(g, seed, draws):
    e=g.assign(sqerr=(g.prediction-g.observed_share)**2)
    wide=(e.groupby(['pair','model'],as_index=False).sqerr.mean()
            .pivot(index='pair',columns='model',values='sqerr')
            .reindex(columns=MODELS))
    if wide.isna().any().any(): raise ValueError('Missing pair/model in bootstrap')
    z=wide.to_numpy(float)
    rng=np.random.default_rng(seed)
    n=len(z)
    # Memory-bounded vectorized bootstrap in chunks.
    diffs=[]
    for start in range(0,draws,1000):
        batch=min(1000,draws-start)
        inds=rng.integers(0,n,size=(batch,n))
        means=z[inds].mean(axis=1)
        diffs.extend((np.sqrt(means[:,0])-np.sqrt(means[:,1])).tolist())
    diffs=np.asarray(diffs)
    point=float(np.sqrt(z[:,0].mean())-np.sqrt(z[:,1].mean()))
    return {'mask':str(g['mask'].iloc[0]),'pairs':n,
            'rmse_submitted':float(np.sqrt(z[:,0].mean())),
            'rmse_original':float(np.sqrt(z[:,1].mean())),
            'rmse_advantage_original':point,
            'cluster_bootstrap_95pct_low':float(np.quantile(diffs,.025)),
            'cluster_bootstrap_95pct_high':float(np.quantile(diffs,.975)),
            'bootstrap_fraction_original_better':float(np.mean(diffs>0))}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',type=Path,default=Path('reports/robustness/phase3_matched_8pairs/paired_predictions.csv'))
    a.add_argument('--output',type=Path,default=Path('reports/robustness/phase3_matched_8pairs/pairwise_analysis'))
    a.add_argument('--draws',type=int,default=10000)
    a.add_argument('--seed',type=int,default=20261122)
    args=a.parse_args()
    if args.draws<1000: a.error('--draws must be >= 1000')
    df=pd.read_csv(args.input)
    n_pairs=check(df)
    if n_pairs<10: raise ValueError('Expected at least 10 pairs')
    stats=[boot_for_mask(df.loc[df['mask'].eq(m)],args.seed+i,args.draws) for i,m in enumerate(MASKS)]
    out=args.output
    if out.resolve() in [Path('.').resolve(),args.input.parent.resolve()]:
        raise ValueError('Use a dedicated output directory')
    out.mkdir(parents=True,exist_ok=True)
    pair_summary=pd.DataFrame(stats)
    pair_summary.to_csv(out/'pair_cluster_bootstrap.csv',index=False)
    # The same 48 pairs recur in every missingness condition; never pool them as independent.
    d=df.copy()
    d['squared_error']=(d.prediction-d.observed_share)**2
    d['absolute_error']=(d.prediction-d.observed_share).abs()
    d['signed_error']=d.prediction-d.observed_share
    columns=['model','mask','station','requested_share','event_duration']
    report=(d.groupby(columns,dropna=False)
              .agg(n=('example','size'),pairs=('pair','nunique'),mse=('squared_error','mean'),
                   mae=('absolute_error','mean'),bias=('signed_error','mean'),
                   mean_target=('observed_share','mean'),mean_pred=('prediction','mean'))
              .reset_index())
    report['rmse']=np.sqrt(report.pop('mse'))
    report.to_csv(out/'conditional_errors.csv',index=False)
    # Inspect source exposure. Different pairs may share background runs or source hulls.
    unique_pairs=(d[d.model.eq(MODELS[0]) & d['mask'].eq('complete')]
                  .sort_values('example').drop_duplicates('pair'))
    exposure=(unique_pairs.groupby('station')
              .agg(pairs=('pair','nunique'),background_runs=('background_run_id','nunique'),
                   distinct_source_clips=('source_clip_uid','nunique')).reset_index())
    exposure.to_csv(out/'pair_exposure_by_station.csv',index=False)
    info={'purpose':'retrospective pair-cluster uncertainty and conditional-error audit',
          'n_pairs':n_pairs,'n_predictions':len(df),'draws':args.draws,'seed':args.seed,
          'resampling_unit':'matched pair, 13 scenarios per pair, independently for each mask',
          'interpretation':'exploratory development uncertainty ONLY; previously inspected validation sources; 48 pairs not independent of reused runs/clips',
          'limitations':['No real hidden labels or unlabelled Test.csv measurements used',
                         'Cluster bootstrap treats pairs as exchangeable; repeated background runs or vessel sources can violate this assumption',
                         'Six masks and multiple conditional contrasts are exploratory, not confirmatory',
                         'No retraining or source-version verification is performed']}
    (out/'analysis_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print('PHASE 3 PAIR-AWARE ANALYSIS: PASS')
    print('Matched pairs:',n_pairs,'; predictions:',len(df))
    print(pair_summary.round(6).to_string(index=False))
    print('\nSTATION EXPOSURE')
    print(exposure.to_string(index=False))
    print('\nOutputs:',out)
    print('LIMIT: Development evidence only; bootstrap is pair-clustered, not source-run/hull independent.')

if __name__=='__main__':main()
