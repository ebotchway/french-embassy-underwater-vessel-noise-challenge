#!/usr/bin/env python3
"""Create a cautious FIRST candidate CSV for the French Embassy noise challenge."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from train_synthetic_baselines_v2 import feature_families,score
from train_synthetic_baselines_v1 import load_partition,audit_overlap

BANDS=['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
'251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9',
'1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']


def normalize(X):
    """Per-example reference shift in log-level units; no learned statistics."""
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=3 or X.shape[1:]!=(15,25) or not np.isfinite(X).all():
        raise ValueError('Expected finite N x 15 x 25 levels')
    return X-X.max(axis=(1,2),keepdims=True)


def feature_names():
    return [f'm{minute:02d}_TOL_{band}' for minute in range(1,16) for band in BANDS]


def impute_missing_minutes(X):
    """Fixed within-window interpolation; missing level is NEVER treated as silence."""
    X=np.asarray(X,dtype=np.float64).copy()
    if X.ndim!=3 or X.shape[1:]!=(15,25):
        raise ValueError('Unexpected test acoustic array shape')
    if np.isinf(X).any():
        raise ValueError('Infinite levels in test data')
    original_missing=int(np.isnan(X).sum())
    for i in range(len(X)):
        for j in range(25):
            series=X[i,:,j]
            good=np.isfinite(series)
            if not good.any():
                raise ValueError(f'No observed minutes in test row {i}, band {j}')
            if not good.all():
                t=np.arange(15)
                X[i,:,j]=np.interp(t,t[good],series[good])
    if not np.isfinite(X).all():
        raise ValueError('Imputation failed to create finite values')
    return X,original_missing


def check_test(test,sample):
    needed=feature_names()
    if 'ID' not in test or list(sample.columns)!=['ID','share_added_vessel','score_added_vessel']:
        raise ValueError('Submission schema mismatch')
    if len(test)!=len(sample) or not test.ID.is_unique or not sample.ID.is_unique or not sample.ID.equals(test.ID):
        raise ValueError('Test IDs/order mismatch with sample submission')
    actual=[c for c in test if c!='ID']
    if len(actual)!=375 or set(actual)!=set(needed):
        raise ValueError('Test acoustic bands do not exactly match the verified 15x25 layout')
    if len(test)!=1500:raise ValueError('Expected 1500 test rows')
    X=test[needed].to_numpy(dtype=np.float64).reshape(-1,15,25)
    return impute_missing_minutes(X)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def synthetic_missingness_stress(model, Xva, yva):
    """Prespecified robustness probes, never derived from competition test values."""
    tests={'missing_first_minute':[0], 'missing_last_minute':[14],
           'missing_middle_minute':[7], 'missing_both_edges':[0,14]}
    reports={}
    for label,minutes in tests.items():
        corrupted=np.asarray(Xva,dtype=np.float64).copy()
        corrupted[:,minutes,:]=np.nan
        restored,n_missing=impute_missing_minutes(corrupted)
        values=feature_families(normalize(restored))['combined_96']
        estimates=np.clip(model.predict(values),0,1)
        metrics=score(yva,estimates)
        reports[label]={**metrics,'masked_minutes':minutes,'missing_cells':n_missing}
        print(f'  {label:26s} synthetic_RMSE={metrics["rmse"]:.6f} MAE={metrics["mae"]:.6f}')
    return reports


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--synthetic',type=Path,default=Path('reports/eda/synthetic_v2'))
    a.add_argument('--test',type=Path,default=Path('data/zindi/raw/Test.csv'))
    a.add_argument('--sample',type=Path,default=Path('data/zindi/raw/sample_submission.csv'))
    a.add_argument('--output',type=Path,default=Path('reports/submissions/first_candidate.csv'))
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--write-submission',action='store_true',help='Required explicit opt-in to write CSV')
    args=a.parse_args()
    Xtr,ytr,ptr,_=load_partition(args.synthetic,'train')
    Xva,yva,pva,_=load_partition(args.synthetic,'validation')
    audit_overlap(ptr,pva)
    if len(ytr)<100 or len(yva)<30:raise ValueError('Unexpectedly small synthetic dataset')
    # Apply SAME deterministic, per-example transform on all partitions.
    train_feat=feature_families(normalize(Xtr))['combined_96']
    valid_feat=feature_families(normalize(Xva))['combined_96']
    model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,
                                 random_state=args.seed,n_jobs=-1)
    model.fit(train_feat,ytr)
    val=np.clip(model.predict(valid_feat),0,1)
    metric=score(yva,val)
    print('REPRESENTATION-AWARE MODEL (new experiment; NOT the published 0.112061 model)')
    print(f'Synthetic validation: RMSE={metric["rmse"]:.6f} MAE={metric["mae"]:.6f}')
    print('Prespecified simulated missing-minute stress tests (not leaderboard results):')
    missing_stress=synthetic_missingness_stress(model,Xva,yva)
    # No hyperparameter tuning, no fitting on validation/test.
    test=pd.read_csv(args.test)
    sample=pd.read_csv(args.sample)
    Xt,missing_count=check_test(test,sample)
    print(f'Missing test cells: {missing_count}; imputed within each window by time interpolation/nearest endpoint.')
    test_feat=feature_families(normalize(Xt))['combined_96']
    share=np.clip(model.predict(test_feat),0,1)
    if not np.isfinite(share).all():raise ValueError('Invalid inference result')
    # Evidence score is deliberately a monotone, UNCALIBRATED proxy of estimated share.
    submission=sample[['ID']].copy()
    submission['share_added_vessel']=share
    submission['score_added_vessel']=share.copy()
    if list(submission.columns)!=list(sample.columns):raise ValueError('Wrong column order')
    if not submission.ID.equals(sample.ID):raise ValueError('Wrong IDs')
    if not np.isfinite(submission.iloc[:,1:].to_numpy()).all():raise ValueError('Nonfinite outputs')
    if not ((submission.iloc[:,1:]>=0)&(submission.iloc[:,1:]<=1)).all().all():raise ValueError('Out-of-range outputs')
    print('SUBMISSION SCHEMA: PASS; 1500 rows, 3 columns, finite [0,1] predictions')
    print('Predicted-share range: [{:.6f}, {:.6f}]'.format(float(share.min()),float(share.max())))
    print('CAUTION: Unknown hidden-test performance; evidence score is NOT calibrated.')
    if not args.write_submission:
        print('DRY RUN: no submission file saved. Add --write-submission after reviewing diagnostics.')
        return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing submission: '+str(args.output))
    submission.to_csv(args.output,index=False,float_format='%.10f')
    rec={'purpose':'first exploratory leaderboard candidate','synthetic_validation':metric,'synthetic_missingness_stress':missing_stress,
         'test_used_only_for_inference':True,'test_rows':len(test), 'test_missing_cells':missing_count,
         'missing_policy':'Within each window and band: linear time interpolation; nearest observed endpoint for edge gaps; no fitted parameters',
         'train_sha256':sha(args.synthetic/'train.npz'),
         'test_schema_sha256':hashlib.sha256(','.join(test.columns).encode()).hexdigest(),
         'seed':args.seed,'normalization':'subtract each window maximum from log-levels',
         'evidence_score':'uncalibrated copy of predicted share',
         'warnings':['Do not equate synthetic RMSE with leaderboard performance',
                     'Absolute level information is removed by design',
                     'Missing-minute interpolation is unvalidated on the synthetic validation data',
                     'External data sources and licences must be declared']}
    (args.output.parent/(args.output.stem+'_manifest.json')).write_text(json.dumps(rec,indent=2)+'\n')
    print('Candidate saved:',args.output)

if __name__=='__main__':main()
