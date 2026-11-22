#!/usr/bin/env python3
"""Read-only data-to-model alignment audit for Synthetic Experiment 002.

Reads source provenance and saved validation predictions. Never accesses Zindi Test.csv.
Source-group results are descriptive, not causal, and do not prove natural-vessel detection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(frame, cols):
    rows=[]
    for key,g in frame.groupby(cols,dropna=False,sort=True):
        if not isinstance(key,tuple):key=(key,)
        rec=dict(zip(cols,key))
        e=g.error.to_numpy(float)
        rec.update(n=len(g),rmse=float(np.sqrt(np.mean(e*e))),
                   mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)),
                   actual_mean=float(g.actual_share.mean()),
                   predicted_mean=float(g.prediction.mean()),
                   top30_n=int(g.top30.sum()))
        rec['top30_expected']=float(30*len(g)/len(frame))
        rec['top30_enrichment']=float(rec['top30_n']/rec['top30_expected'])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=Path('reports/baselines/synthetic_v2/validation_predictions.csv'))
    p.add_argument('--provenance',type=Path,default=Path('reports/eda/synthetic_v2/validation_provenance.csv'))
    p.add_argument('--background-index',type=Path,default=Path('reports/eda/background_window_index.csv'))
    p.add_argument('--out',type=Path,default=Path('reports/eda/data_model_alignment'))
    p.add_argument('--model',default='rf_combined_96')
    args=p.parse_args()
    pred=pd.read_csv(args.predictions)
    prov=pd.read_csv(args.provenance,dtype={'vessel_hull_id':'string','vessel_clip_uid':'string'})
    required={'example','actual_share','station','target_regime','vessel_duration_min',args.model}
    if not required<=set(pred):raise ValueError('Missing prediction columns: '+str(required-set(pred)))
    if not {'example','background_run_id','vessel_hull_id','vessel_clip_uid','relative_scale'}<=set(prov):
        raise ValueError('Missing source provenance columns')
    if pred.example.duplicated().any() or prov.example.duplicated().any():raise ValueError('Duplicate examples')
    df=pred.merge(prov,on='example',how='left',validate='one_to_one',indicator=True,suffixes=('','_source'))
    if len(df)!=len(pred) or not df._merge.eq('both').all():raise ValueError('Missing provenance')
    df=df.drop(columns='_merge')
    if 'station_source' in df and not df.station.eq(df.station_source).all():raise ValueError('Station disagreement')
    y=df.actual_share.to_numpy(float)
    z=df[args.model].to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(z).all() or not np.all((0<=y)&(y<=1)) or not np.all((0<=z)&(z<=1)):
        raise ValueError('Nonfinite/out-of-bounds prediction or target')
    df['prediction']=z;df['error']=z-y;df['abs_error']=np.abs(df.error)
    df=df.sort_values(['abs_error','example'],ascending=[False,True]).reset_index(drop=True)
    n_top=min(30,len(df));df['top30']=False;df.loc[:n_top-1,'top30']=True
    if len(df)!=300:
        print('WARNING: expected 300 validation rows; top30 interpretation needs adjustment for N=',len(df))
    args.out.mkdir(parents=True,exist_ok=True)
    outputs={}
    for key,cols in {
        'station':['station'], 'target_regime':['target_regime'],
        'station_by_regime':['station','target_regime'],
        'duration':['vessel_duration_min'],
        'class':['vessel_class'],
        'background_run':['background_run_id'],
        'hull':['vessel_hull_id'], 'clip':['vessel_clip_uid']}.items():
        table=summarize(df,cols)
        table.to_csv(args.out/f'{key}_metrics.csv',index=False)
        outputs[key]=table
    if args.background_index.is_file():
        bg=pd.read_csv(args.background_index,usecols=['station','run_id'])
        index_counts=bg.groupby('station').agg(indexed_windows=('run_id','size'),indexed_runs=('run_id','nunique')).reset_index()
        joined=outputs['station'].merge(index_counts,on='station',how='left',validate='one_to_one')
        joined.to_csv(args.out/'station_with_archive_context.csv',index=False)
    # Across the whole validation partition, compare sample frequencies, not just top-error counts.
    grouped=outputs['hull'].sort_values(['top30_enrichment','n'],ascending=[False,False])
    grouped.loc[grouped.n>=10].to_csv(args.out/'hulls_with_10plus_examples.csv',index=False)
    # Show the actual pre-processing concern with reference to code, not a claim of causation.
    context={
        'experiment':'002', 'model':args.model, 'n_validation':len(df),
        'overall_rmse':float(np.sqrt(np.mean(df.error.to_numpy()**2))),
        'top_n':n_top,
        'interpretation':'Associations and exposure-adjusted descriptive statistics; not causal tests',
        'source_reuse':'Multiple examples may use one clip, hull, recorder, or run',
        'calibration_warning':'Generator converts background with 10**((dB - window_max)/10). Zindi input unit matching remains unverified.',
        'test_data_used':False,
        'recommended_next':'Matched same-background/source duration tests; normalization ablation; session-weighted robustness; independent real-data evidence'
    }
    (args.out/'audit_summary.json').write_text(json.dumps(context,indent=2)+'\n')
    print('DATA-TO-MODEL ALIGNMENT AUDIT: PASS')
    print(f'Examples={len(df)} overall_rmse={context["overall_rmse"]:.6f}')
    for key in ['station','target_regime','duration']:
        print('\n'+key.upper());print(outputs[key].round(4).to_string(index=False))
    print('\nLIMIT: This is a retrospective descriptive audit, not an independent evaluation.')
    print('Reports:',args.out)

if __name__=='__main__':main()
