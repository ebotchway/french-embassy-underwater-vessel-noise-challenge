#!/usr/bin/env python3
"""One-shot prespecified Experiment 004 frozen-group evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from train_synthetic_baselines_v2 import feature_families

DEFAULT_DATA=Path('reports/eda/synthetic_experiment004_v1')
DEFAULT_OUT=Path('reports/baselines/experiment004_frozen_v1')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def read(data,part):
    with np.load(data/f'{part}.npz') as z:
        X=z['X'].astype(np.float64)
        y=z['y'].astype(np.float64)
    provenance=pd.read_csv(data/f'{part}_provenance.csv',keep_default_na=False)
    if X.ndim!=3 or X.shape[1:]!=(15,25) or len(y)!=len(X) or len(provenance)!=len(y):
        raise ValueError('Bad array/provenance shapes')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any():
        raise ValueError('Nonfinite or out-of-range dataset')
    if not np.array_equal(provenance.example.to_numpy(int), np.arange(len(X))):
        raise ValueError('Provenance sample order mismatch')
    if not np.allclose(provenance.observed_share.to_numpy(float),y,atol=5e-7):
        raise ValueError('Target / metadata disagreement')
    return X,y,provenance

def metrics(y,p,d,s):
    rows={'n':len(y),'zero':int((d==0).sum()),'positive':int((d==1).sum()),
          'rmse':float(np.sqrt(np.mean((p-y)**2))),
          'mae':float(np.mean(np.abs(p-y))),
          'bias':float(np.mean(p-y))}
    rows['roc_auc']=float(roc_auc_score(d,s)) if len(np.unique(d)) == 2 else None
    rows['average_precision']=float(average_precision_score(d,s)) if len(np.unique(d)) == 2 else None
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Refusing to overwrite one-shot evaluation: '+str(a.output))
    Xt,yt,pt=read(a.data,'exp004_train')
    Xe,ye,pe=read(a.data,'exp004_eval')
    for key in ['background_run_id','vessel_hull_id','vessel_clip_uid']:
        if (set(pt[key])-{''}) & (set(pe[key])-{''}):
            raise ValueError('Train/eval group leakage: '+key)
    dt=(yt>1e-12).astype(int)
    de=(ye>1e-12).astype(int)
    if len(np.unique(dt))!=2 or len(np.unique(de))!=2:
        raise ValueError('Detection requires both positive and zero observations')
    a.output.mkdir(parents=True,exist_ok=False)
    summary=[]
    predictions=[]
    station=[]
    for name,shift in [('synthetic_original',False),('submitted_max_shift',True)]:
        xtr=Xt-Xt.max(axis=(1,2),keepdims=True) if shift else Xt
        xev=Xe-Xe.max(axis=(1,2),keepdims=True) if shift else Xe
        tr=feature_families(xtr)['combined_96']
        ev=feature_families(xev)['combined_96']
        reg=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,
                                  random_state=a.seed,n_jobs=-1)
        det=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,
                                   random_state=a.seed,n_jobs=-1,class_weight=None)
        reg.fit(tr,yt)
        det.fit(tr,dt)
        pred=np.clip(reg.predict(ev),0,1)
        prob=det.predict_proba(ev)[:,list(det.classes_).index(1)]
        results=metrics(ye,pred,de,prob)
        results.update({'representation':name,'fraction_estimator':'RandomForestRegressor',
                        'detector':'RandomForestClassifier'})
        summary.append(results)
        for i in range(len(ye)):
            predictions.append({'example':i,'representation':name,'true_observed_share':ye[i],
                                'estimated_share':pred[i],'observed_addition':int(de[i]),
                                'detection_evidence':prob[i]})
        for group,inds in pe.groupby('station').groups.items():
            idx=np.array(list(inds),dtype=int)
            r=metrics(ye[idx],pred[idx],de[idx],prob[idx])
            r.update({'representation':name,'station':group,
                      'background_runs':int(pe.iloc[idx].background_run_id.nunique()),
                      'vessel_hulls':int(pe.iloc[idx].loc[pe.iloc[idx].vessel_hull_id.ne(''),'vessel_hull_id'].nunique())})
            station.append(r)
        print(f'{name}: RMSE={results["rmse"]:.6f} ROC_AUC={results["roc_auc"]:.6f} AP={results["average_precision"]:.6f}',flush=True)
    pd.DataFrame(summary).to_csv(a.output/'summary.csv',index=False)
    pd.DataFrame(predictions).to_csv(a.output/'predictions.csv',index=False)
    pd.DataFrame(station).to_csv(a.output/'station_metrics.csv',index=False)
    manifest={'purpose':'one-shot source-group-disjoint Experiment004 synthetic evaluation',
              'limitations':['Evaluation groups were in original historical training pool; not untouched relative to earlier research',
                             'No independent real measured addition labels',
                             'Per-window acoustic levels and vessel references not physically cross-calibrated',
                             'Detection ROC AUC only for new synthetic controlled additions',
                             'Station/class counts and source-group dependence limit interpretation'],
              'seed':a.seed,'fixed_rf':{'n_estimators':200,'min_samples_leaf':2},
              'input_sha256':{str(a.data/f'{part}{suffix}'):sha(a.data/f'{part}{suffix}')
                              for part in ('exp004_train','exp004_eval') for suffix in ('.npz','_provenance.csv')},
              'train_examples':len(yt),'eval_examples':len(ye),
              'results':summary}
    (a.output/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('EXPERIMENT 004 ONE-SHOT EVALUATION: PASS; output:',a.output)
    print('Do not iterate hyperparameters or split selections against these evaluation results.')

if __name__ == '__main__':main()
