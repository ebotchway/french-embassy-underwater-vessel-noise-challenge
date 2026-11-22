#!/usr/bin/env python3
"""Reproduce Submission 001's TRAIN/VALIDATION pipeline and measure detection.

NO test values, labels, leaderboard feedback or calibration are used. Train
and validation inputs come exclusively from synthetic_v2. Import the exact
fixed preprocessing, features, and model configuration of the submitted script.

Usage: python scripts/audit_submission001_detection.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
from train_synthetic_baselines_v1 import load_partition, audit_overlap
from train_synthetic_baselines_v2 import feature_families
from prepare_zindi_first_submission import normalize, impute_missing_minutes


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, score):
    positive = np.asarray(y) > 0
    out = {'n': len(y), 'zero': int((~positive).sum()), 'positive': int(positive.sum()),
           'share_rmse': float(np.sqrt(mean_squared_error(y,score))),
           'share_mae': float(mean_absolute_error(y,score))}
    if positive.any() and (~positive).any():
        out['roc_auc'] = float(roc_auc_score(positive,score))
        out['average_precision'] = float(average_precision_score(positive,score))
    else:
        out['roc_auc'] = None
        out['average_precision'] = None
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('reports/eda/synthetic_v2'))
    ap.add_argument('--output',type=Path,default=Path('reports/audits/submission001_detection'))
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    Xtr,ytr,ptr,htr=load_partition(args.input,'train')
    Xva,yva,pva,hva=load_partition(args.input,'validation')
    audit_overlap(ptr,pva)
    if len(Xtr)<100 or len(Xva)<30: raise ValueError('Unexpected dataset size')
    tr=feature_families(normalize(Xtr))['combined_96']
    va=feature_families(normalize(Xva))['combined_96']
    model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=args.seed,n_jobs=-1)
    model.fit(tr,ytr)
    pred=np.clip(model.predict(va),0,1)
    overall=metrics(yva,pred)
    print('SUBMISSION 001 PIPELINE — SYNTHETIC DETECTION AUDIT')
    for k,v in overall.items():print(f'  {k}: {v}')
    if abs(overall['share_rmse']-0.166477)>0.00005:
        raise ValueError('Reproduced synthetic RMSE does not match recorded Submission 001 result')
    rows=[]
    for station in sorted(pva.station.unique()):
        mask=pva.station.eq(station).to_numpy()
        rows.append({'station':station,**metrics(yva[mask],pred[mask]),
                     'background_runs':pva.loc[mask,'background_run_id'].nunique(),
                     'vessel_hulls':pva.loc[mask & (yva>0),'vessel_hull_id'].nunique()})
    by_station=pd.DataFrame(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    by_station.to_csv(args.output/'by_station.csv',index=False)
    predictions=pva[['example','station','background_run_id','vessel_hull_id','vessel_duration_min','target_regime']].copy()
    predictions['actual_share']=yva
    predictions['submitted_pipeline_share']=pred
    predictions.to_csv(args.output/'validation_predictions.csv',index=False)
    masks={'complete':[], 'first':[0], 'last':[14], 'middle':[7], 'both_edges':[0,14]}
    stress=[]
    for label,minutes in masks.items():
        X=Xva.copy()
        if minutes:
            X[:,minutes,:]=np.nan
            X,_=impute_missing_minutes(X)
        p=np.clip(model.predict(feature_families(normalize(X))['combined_96']),0,1)
        stress.append({'scenario':label,**metrics(yva,p)})
    stress=pd.DataFrame(stress)
    stress.to_csv(args.output/'missingness_stress_detection.csv',index=False)
    report={'purpose':'Like-for-like synthetic detection audit of Submission 001 inference model',
            'not_test_labels':True,'not_leaderboard_tuning':True,'seed':args.seed,
            'training_hashes':htr,'validation_hashes':hva,'overall':overall,
            'code_hashes':{p.name:digest(p) for p in [Path('scripts/prepare_zindi_first_submission.py'),
                Path('scripts/train_synthetic_baselines_v2.py'),Path('scripts/train_synthetic_baselines_v1.py')] if p.exists()},
            'caveats':['Same synthetic generator for training and validation',
                       'ROC AUC measures synthetic controlled additions only',
                       'Public AUC is on different unknown examples; score difference is not causal evidence',
                       'Repeated vessel clips and background runs reduce effective independence']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nMISSING-MINUTE DETECTION STRESS')
    print(stress[['scenario','share_rmse','roc_auc','average_precision']].to_string(index=False))
    print('\nSTATIONS')
    print(by_station[['station','n','roc_auc','share_rmse','background_runs']].to_string(index=False))
    print('\nPASS: read-only synthetic audit; outputs:',args.output)

if __name__=='__main__':main()
