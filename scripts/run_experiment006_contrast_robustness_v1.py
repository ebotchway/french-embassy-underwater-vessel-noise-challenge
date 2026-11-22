#!/usr/bin/env python3
"""Experiment 006: source-group-separated INTERNAL DEVELOPMENT contrast robustness.

No exp004_eval, existing_validation, sealed_station_holdout or Zindi Test.csv.
Use only exp004_train source rows. Deterministically split its recording runs and
vessel hulls into INTERNAL train/dev pools, then freeze the chosen split by SHA256.
Compare equal-size original-template vs spectrum-varied RF models on matched
synthetic development pairs. This is a sensitivity experiment, NOT evidence of
real vessel acoustics or competition generalization.

Run from project root:
  python scripts/run_experiment006_contrast_robustness_v1.py --self-test-only
  python scripts/run_experiment006_contrast_robustness_v1.py --train 1200 --dev-pairs 120

Output directory must not exist. Frozen Experiment 004 files are only READ.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from build_synthetic_mixtures_v1 import BANDS, ParquetRowReader, SHAPE, db_to_relative_energy
from build_synthetic_mixtures_v2 import embed_event
from build_experiment004_frozen_mixtures import EXPECTED_FROZEN_SHA256, impute, observed_fraction, sha
from audit_simulation_realism_v1 import reshape_source_spectrum, mix_at_share
from train_synthetic_baselines_v2 import feature_families

FROZEN=Path('reports/splits/experiment004_frozen_v1')
OUT=Path('reports/baselines/experiment006_internal_contrast_v1')
SEED=20261123
GAMMAS=(0.,.5,1.)
MASKS={'complete':(), 'first':(0,), 'middle':(7,), 'last':(14,), 'both_edges':(0,14)}
MASK_PROB=(.6,.1,.1,.1,.1)
LABELS={0.:'original_vtuad',.5:'halfway_shape',1.:'background_shaped_control'}


def split_groups(frame,group_key,stratum_key,seed,share=.20):
    """Group-disjoint deterministic 80/20 split, rare strata train-only."""
    out={}
    for group,block in frame.groupby(group_key,sort=True):
        strata=sorted(set(block[stratum_key].astype(str)))
        if len(strata)!=1: raise ValueError(f'{group_key} {group} crosses strata: {strata}')
        out[str(group)]=strata[0]
    assignments={}
    for stratum in sorted(set(out.values())):
        groups=sorted(g for g,s in out.items() if s==stratum)
        # A single recording run or hull cannot be in two partitions.
        if len(groups)<2:
            assignments.update({g:'internal_train' for g in groups})
            continue
        ordered=sorted(groups,key=lambda g:hashlib.sha256(f'{seed}|{stratum}|{g}'.encode()).hexdigest())
        take=max(1,min(len(groups)-1,round(share*len(groups))))
        assignments.update({g:('internal_dev' if i<take else 'internal_train')
                            for i,g in enumerate(ordered)})
    return assignments


def verify_split(bg,vt,br,vh):
    for key,assign in [('run',br),('hull',vh)]:
        left={g for g,p in assign.items() if p=='internal_train'}
        right={g for g,p in assign.items() if p=='internal_dev'}
        if left & right or not right:raise ValueError(f'Invalid {key} split')
    if not set(br)==set(bg.run_id.astype(str)) or not set(vh)==set(vt.hull_id.dropna().astype(str)):
        raise ValueError('Split group coverage mismatch')


def make_clip_pool(vt,assign,part):
    rows=vt[vt.hull_id.astype(str).map(assign).eq(part)]
    clips={}; by_class={}
    for uid,group in rows.groupby('clip_uid',sort=True):
        group=group.sort_values('minute_index')
        if group.hull_id.nunique()!=1 or group.label.nunique()!=1:
            raise ValueError('Inconsistent clip')
        if not np.array_equal(group.minute_index.to_numpy(int),np.arange(len(group))):
            raise ValueError('Nonconsecutive clip minutes')
        array=group[BANDS].to_numpy(float)
        if not np.isfinite(array).all() or (array<0).any() or array.sum()<=0:
            raise ValueError('Invalid clip energy')
        hull=str(group.hull_id.iloc[0]); label=str(group.label.iloc[0])
        clips[str(uid)]=(hull,label,array)
        by_class.setdefault(label,[]).append(str(uid))
    if not clips:raise ValueError(f'Empty {part} clip pool')
    return clips,by_class


def pick_clip(clips,by_class,rng,duration):
    eligible={s:[uid for uid in uids if len(clips[uid][2])>=duration]
              for s,uids in by_class.items()}
    labels=sorted(s for s,ids in eligible.items() if ids)
    if not labels:raise ValueError('No clips long enough')
    label=labels[int(rng.integers(len(labels)))]
    ids=eligible[label]; uid=ids[int(rng.integers(len(ids)))]
    return uid,clips[uid]


def make_scenario(B,V,q,gamma,missing):
    if q==0:
        M=B.copy();alpha=0.;clipped=0
    else:
        adj,clipped=reshape_source_spectrum(V,B,gamma)
        M,_,alpha=mix_at_share(B,adj,q)
        V=adj
    obs=np.ones(SHAPE,dtype=bool)
    obs[list(missing),:]=False
    y=observed_fraction(B,V,alpha,obs)
    lev=10*np.log10(np.maximum(M,1e-300))
    lev[list(missing),:]=np.nan
    X=impute(lev).astype(np.float32)
    if not np.isfinite(X).all() or y<0 or y>=1:raise ValueError('Bad generated data')
    return X,y,clipped


def sample_backgrounds(pool,n,seed):
    # Station-balanced sampling, no repeated indexed windows; never cross split.
    rng=np.random.default_rng(seed)
    groups={s:rng.permutation(g.index.to_numpy()).tolist()
            for s,g in pool.groupby('station',sort=True)}
    picked=[];count={s:0 for s in groups}
    while len(picked)<n:
        viable=[s for s in sorted(groups) if groups[s]]
        if not viable:raise ValueError('Insufficient windows for desired sample count')
        minimum=min(count[s] for s in viable)
        tied=[s for s in viable if count[s]==minimum]
        s=tied[int(rng.integers(len(tied)))]
        picked.append(groups[s].pop());count[s]+=1
    return pool.loc[picked].reset_index(drop=True),count


def prespecified_q(rng):
    u=float(rng.random())
    if u<.2:return 0.
    if u<.48:return float(rng.uniform(.005,.10))
    return float(rng.uniform(.10,.80))


def build_training(bg,vt,br,vh,n,seed,reader):
    rng=np.random.default_rng(seed)
    pool=bg[bg.run_id.astype(str).map(br).eq('internal_train')].reset_index(drop=True)
    chosen,counts=sample_backgrounds(pool,n,seed)
    clips,by_class=make_clip_pool(vt,vh,'internal_train')
    X={'original_only':[],'contrast_varied':[]}; y={'original_only':[],'contrast_varied':[]}
    meta=[]
    for idx,row in enumerate(chosen.itertuples(index=False)):
        B=db_to_relative_energy(reader.read(str(row.source_parquet),json.loads(row.row_positions_json)))
        q=prespecified_q(rng)
        dur=int(rng.choice([1,3]))
        mask=str(rng.choice(list(MASKS),p=MASK_PROB));missing=MASKS[mask]
        V=np.zeros(SHAPE)
        uid=hull=label=''
        if q>0:
            uid,(hull,label,array)=pick_clip(clips,by_class,rng,dur)
            off=int(rng.integers(len(array)-dur+1))
            insertion=int(rng.integers(16-dur))
            V[insertion:insertion+dur]=array[off:off+dur]
        vary=float(rng.choice(GAMMAS))
        for name,gamma in [('original_only',0.),('contrast_varied',vary)]:
            xx,yy,_=make_scenario(B,V,q,gamma,missing)
            X[name].append(xx);y[name].append(yy)
        meta.append({'example':idx,'station':row.station,'background_run_id':row.run_id,
                     'vessel_clip_uid':uid,'vessel_hull_id':hull,'vessel_class':label,
                     'requested_share':q,'duration_min':dur,'mask':mask,'gamma_varied':vary})
        if (idx+1)%200==0 or idx+1==n:print(f'Internal training windows: {idx+1}/{n}',flush=True)
    return {k:(np.stack(X[k]),np.asarray(y[k],float)) for k in X},pd.DataFrame(meta),counts


def build_internal_dev(bg,vt,br,vh,n,seed,reader):
    rng=np.random.default_rng(seed)
    pool=bg[bg.run_id.astype(str).map(br).eq('internal_dev')].reset_index(drop=True)
    chosen,counts=sample_backgrounds(pool,n,seed)
    clips,by_class=make_clip_pool(vt,vh,'internal_dev')
    X=[];ys=[];records=[]
    for i,row in enumerate(chosen.itertuples(index=False)):
        B=db_to_relative_energy(reader.read(str(row.source_parquet),json.loads(row.row_positions_json)))
        mask=str(rng.choice(list(MASKS),p=MASK_PROB));missing=MASKS[mask]
        duration=int(rng.choice([1,3]))
        uid,(hull,label,array)=pick_clip(clips,by_class,rng,duration)
        offset=int(rng.integers(len(array)-duration+1))
        insertion=int(rng.integers(16-duration))
        V=np.zeros(SHAPE);V[insertion:insertion+duration]=array[offset:offset+duration]
        conditions=[(0.,0.)]+[(q,gamma) for q in (.03,.15) for gamma in GAMMAS]
        for q,gamma in conditions:
            xx,yy,clipped=make_scenario(B,V,q,gamma,missing)
            X.append(xx);ys.append(yy)
            records.append({'pair':i,'station':row.station,'background_run_id':row.run_id,
                            'vessel_clip_uid':uid,'vessel_hull_id':hull,'vessel_class':label,
                            'requested_share':q,'observed_share':yy,'duration_min':duration,
                            'mask':mask,'spectral_intervention':'no_addition' if q==0 else LABELS[gamma],
                            'gamma':gamma,'clipped_band_gains':clipped})
        if (i+1)%50==0 or i+1==n:print(f'Internal dev pairs: {i+1}/{n}',flush=True)
    return np.stack(X),np.asarray(ys),pd.DataFrame(records),counts


def fit_predict(train,dev,seed):
    X,y=train
    feat=feature_families(X)['combined_96']
    ev=feature_families(dev)['combined_96']
    cls=(y>1e-12).astype(int)
    if len(set(cls))<2:raise ValueError('Training needs zeros and positives')
    reg=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=seed,n_jobs=-1)
    det=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=seed,n_jobs=-1)
    reg.fit(feat,y);det.fit(feat,cls)
    return np.clip(reg.predict(ev),0,1),det.predict_proba(ev)[:,list(det.classes_).index(1)]


def score(grp):
    y=grp.observed_share.to_numpy(float)
    p=grp.estimated_share.to_numpy(float)
    d=(y>1e-12).astype(int)
    s=grp.detection_evidence.to_numpy(float)
    return {'n':len(grp),'pairs':int(grp.pair.nunique()),
            'rmse':float(np.sqrt(np.mean((p-y)**2))),
            'mae':float(np.mean(np.abs(p-y))),
            'bias':float(np.mean(p-y)),
            'roc_auc':float(roc_auc_score(d,s)) if len(set(d))==2 else np.nan,
            'average_precision':float(average_precision_score(d,s)) if len(set(d))==2 else np.nan,
            'zero_false_positive_rate_at_half':float(np.mean(s[d==0]>=.5)) if np.any(d==0) else np.nan}


def self_test():
    B=np.full(SHAPE,.2)
    V=np.zeros(SHAPE);V[7:8]=np.geomspace(.002,.5,25)
    for q in (0.,.03,.15):
        for gamma in GAMMAS:
            x,y,_=make_scenario(B,V,q,gamma,())
            assert x.shape==SHAPE and abs(y-q)<1e-9
            x2,y2,_=make_scenario(B,V,q,gamma,(7,))
            assert np.isfinite(x2).all() and y2==0
    frame=pd.DataFrame({'run_id':['a','b','c','d'],'station':['A','A','A','A']})
    groups=split_groups(frame,'run_id','station',42)
    assert set(groups.values())=={'internal_train','internal_dev'}
    print('EXPERIMENT 006 NUMERICAL AND SPLIT SELF-TEST: PASS')


def run(args):
    if args.output.exists():raise FileExistsError(f'Output already exists; refusing overwrite: {args.output}')
    hashes={}
    for name,expected in EXPECTED_FROZEN_SHA256.items():
        p=args.frozen/name
        if not p.exists() or sha(p)!=expected:raise ValueError(f'Frozen allocation integrity failure: {p}')
        hashes[name]=expected
    bg=pd.read_csv(args.frozen/'background_manifest_exp004.csv',low_memory=False)
    vt=pd.read_csv(args.frozen/'vessel_minute_manifest_exp004.csv',dtype={'hull_id':'string','clip_uid':'string'},low_memory=False)
    bg=bg.loc[bg.experiment004_partition.eq('exp004_train')].copy()
    vt=vt.loc[vt.experiment004_partition.eq('exp004_train')].copy()
    # Provenance ensures no use of previously reserved eval/validation/holdout.
    if bg.empty or vt.empty:raise ValueError('Empty permitted train pools')
    br=split_groups(bg,'run_id','station',args.seed)
    vh=split_groups(vt,'hull_id','label',args.seed+11)
    verify_split(bg,vt,br,vh)
    br_csv=pd.DataFrame([{'run_id':g,'station':s,'allocation':br[g]}
                          for g,s in sorted(bg.groupby('run_id').station.first().items())])
    vh_csv=pd.Data([{'hull_id':g,'vessel_class':s,'allocation':vh[g]}
                          for g,s in sorted(vt.groupby('hull_id').label.first().items())])
    reader=ParquetRowReader()
    train,train_meta,train_stations=build_training(bg,vt,br,vh,args.train,args.seed+101,reader)
    dev,yd,dev_meta,dev_stations=build_internal_dev(bg,vt,br,vh,args.dev_pairs,args.seed+303,reader)
    for key in ('background_run_id','vessel_hull_id','vessel_clip_uid'):
        x=set(train_meta[key].astype(str))-{''}
        z=set(dev_meta[key].astype(str))-{''}
        if x&z:raise ValueError(f'Leakage in {key}')
    results=[];all_preds=[]
    for model in ('original_only','contrast_varied'):
        p,s=fit_predict(train[model],dev,args.seed+404)
        pred=dev_meta.copy();pred['model']=model;pred['estimated_share']=p;pred['detection_evidence']=s
        all_preds.append(pred)
        # ROC AUC by intervention must include common matched zero-addition controls.
        # Calculating AUC on positives alone is undefined.
        nulls=pred.loc[pred.requested_share.eq(0)]
        positives=pred.loc[pred.requested_share.gt(0)]
        for intervention,block in positives.groupby('spectral_intervention',sort=True):
            comparison=pd.concat([nulls,block],ignore_index=True)
            r=score(comparison);r.update({'model':model,'scenario':intervention,'group':'intervention_plus_zeros'})
            results.append(r)
            r=score(block);r.update({'model':model,'scenario':intervention,'group':'intervention_positive_only'})
            results.append(r)
        for station,block in pred.groupby('station',sort=True):
            r=score(block);r.update({'model':model,'scenario':station,'group':'station'})
            results.append(r)
        for mask,block in pred.groupby('mask',sort=True):
            r=score(block);r.update({'model':model,'scenario':mask,'group':'mask'})
            results.append(r)
        for q,block in pred.groupby('requested_share',sort=True):
            r=score(block);r.update({'model':model,'scenario':f'{q:g}','group':'requested_share'})
            results.append(r)
        r=score(pred);r.update({'model':model,'scenario':'all','group':'overall'})
        results.append(r)
        print(f'{model}: RMSE={r["rmse"]:.6f} ROC_AUC={r["roc_auc"]:.6f} AP={r["average_precision"]:.6f}',flush=True)
    prediction=pd.concat(all_preds,ignore_index=True)
    summaries=pd.DataFrame(results)
    # Confirm both models were evaluated on exactly matched source-condition rows.
    paired=prediction.pivot(index=['pair','spectral_intervention','requested_share'],columns='model',values='estimated_share')
    if paired.isna().any().any() or len(paired)*2!=len(prediction):
        raise ValueError('Unmatched evaluation predictions')
    # No CI is reported: source dependencies and historical inspection preclude
    # a defensible independent generalization interval from these rows alone.
    manifest={'experiment':'006 training-only development spectral contrast sensitivity',
              'limitations':['Uses only exp004_train source groups; its internal dev is NOT external or real-world validation',
                             'Matched development rows repeat background runs/hulls; row counts overstate independence',
                             'Spectral shape adjustment is hypothetical and NOT calibrated or vessel-validated',
                             'Detection positive means observed injected energy; a physically present vessel may be unlabelled',
                             'Single prespecified comparison: no adaptive hyperparameter tuning against internal dev'],
              'seed':args.seed,'train_samples_per_model':args.train,'dev_background_pairs':args.dev_pairs,
              'training_gamma':{'original_only':[0.],'contrast_varied':[0.,.5,1.]},
              'development_levels':[0.,.03,.15],'development_gamma':list(GAMMAS),
              'masks':{k:list(v) for k,v in MASKS.items()},'mask_probabilities':MASK_PROB,
              'fixed_rf':{'n_estimators':200,'min_samples_leaf':2},
              'frozen_sha256':hashes,'train_station_samples':train_stations,'dev_station_pairs':dev_stations,
              'group_allocation':'Hash-ordered 20% of training-only runs/hulls per station/class reserved for INTERNAL dev; rare groups train-only',
              'summary':summaries.loc[summaries.group.eq('overall')].to_dict('records')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='experiment006_stage_',dir=args.output.parent) as td:
        stage=Path(td)
        br_csv.to_csv(stage/'internal_background_group_allocation.csv',index=False)
        vh_csv.to_csv(stage/'internal_vessel_group_allocation.csv',index=False)
        train_meta.to_csv(stage/'training_provenance.csv',index=False)
        dev_meta.to_csv(stage/'development_provenance.csv',index=False)
        prediction.to_csv(stage/'matched_development_predictions.csv',index=False)
        summaries.to_csv(stage/'development_summary.csv',index=False)
        manifest['output_sha256']={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
        (stage/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if args.output.exists():raise FileExistsError('Output created by another process')
        stage.rename(args.output)
    print('EXPERIMENT 006 INTERNAL DEVELOPMENT: PASS; output:',args.output)
    print('No Experiment 004 frozen evaluation, original validation, sealed station holdout or Zindi Test.csv used.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,default=FROZEN)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--train',type=int,default=1200)
    p.add_argument('--dev-pairs',type=int,default=120)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--self-test-only',action='store_true')
    args=p.parse_args()
    self_test()
    if not args.self_test_only:run(args)

if __name__=='__main__':main()
