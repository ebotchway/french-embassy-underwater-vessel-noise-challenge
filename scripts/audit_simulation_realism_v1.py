#!/usr/bin/env python3
"""Training-only simulation realism sensitivity audit (no model fitting)."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

FROZEN = Path('reports/splits/experiment004_frozen_v1')
SAMPLED = Path('reports/audits/real_acoustic_development_v1/background_sampled_runs.csv')
OUTPUT = Path('reports/audits/simulation_realism_v1')
EXPECTED = {
    'background_manifest_exp004.csv': 'a1d3dd26c08fbf7a2704c6535f416542179249a89a29522899b4d0b6448862bd',
    'vessel_minute_manifest_exp004.csv': 'f33693c3fd96bca8ac4121af5df45b95240e04fa86f6bdbbc994c5794145f6c5',
    'split_protocol.json': '2df9ddcecf78cdd429867fa182f41f68745addc90a2f5634e59ccd1a644a5c55',
}
BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5',
         '251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9',
         '1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
COLS = ['TOL_'+x for x in BANDS]
SEED = 20261122


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def temporal_features(levels):
    x=np.asarray(levels,dtype=float)
    if x.shape!=(15,25) or not np.isfinite(x).all():
        raise ValueError('Temporal feature requires finite 15x25 levels')
    differences=np.sqrt(np.mean(np.square(np.diff(x,axis=0)),axis=1))
    return {'max_step_rms_db':float(differences.max()),
            'mean_step_rms_db':float(differences.mean()),
            'p90_step_rms_db':float(np.quantile(differences,.9))}


def reshape_source_spectrum(vessel, background, gamma):
    """Gamma=0 original; .5 blend; 1 background-aligned band mean."""
    v=np.asarray(vessel,dtype=float)
    b=np.asarray(background,dtype=float)
    if v.shape!=(15,25) or b.shape!=(15,25) or gamma not in (0.,.5,1.):
        raise ValueError('Invalid source shape experiment')
    if not np.isfinite(v).all() or not np.isfinite(b).all() or (v<0).any() or (b<=0).any() or v.sum()<=0:
        raise ValueError('Invalid relative powers')
    active=v.sum(axis=1)>0
    vb=v[active].mean(axis=0)
    bb=b.mean(axis=0)
    # Relative shape only; mean-centering log gains removes a global scaling.
    gain=np.log(np.maximum(bb,1e-300))-np.log(np.maximum(vb,1e-300))
    gain-=gain.mean()
    clipped=np.clip(gain,-20,20)
    adjusted=v*np.exp(gamma*clipped)[None,:]
    if not np.isfinite(adjusted).all() or adjusted.sum()<=0:
        raise ValueError('Invalid adjusted template')
    return adjusted,int(np.count_nonzero(gain!=clipped))


def mix_at_share(background, source, fraction):
    """Same linear-energy share equation as Experiment 002 (self-testable without Parquet)."""
    b=np.asarray(background,dtype=float);v=np.asarray(source,dtype=float)
    if b.shape!=(15,25) or v.shape!=(15,25) or not np.isfinite(b).all() or not np.isfinite(v).all():
        raise ValueError('Invalid mixture shape or nonfinite input')
    if (b<0).any() or (v<0).any() or b.sum()<=0 or not 0<=fraction<1:
        raise ValueError('Invalid source energy or fraction')
    if fraction>0 and v.sum()<=0:raise ValueError('Empty positive-share template')
    alpha=0. if fraction==0 else (fraction/(1-fraction))*(b.sum()/v.sum())
    result=b+alpha*v
    share=float((alpha*v).sum()/result.sum())
    if not np.isclose(share,fraction,atol=1e-10,rtol=0):raise AssertionError('Share mismatch')
    return result,share,alpha


def db_to_relative_energy(levels):
    x=np.asarray(levels,dtype=float)
    if x.shape!=(15,25) or not np.isfinite(x).all():raise ValueError('Invalid acoustic level window')
    return 10.**((x-x.max())/10.)


def self_test():
    # Original is exactly preserved, every contrast has the same target share,
    # and a null addition gives exactly the baseline background features.
    rng=np.random.default_rng(9)
    b=rng.uniform(.01,1.,(15,25))
    v=np.zeros((15,25));v[5:8]=rng.uniform(.001,.2,(3,25))
    original=v.copy()
    for gamma in (0.,.5,1.):
        adjusted,clip_count=reshape_source_spectrum(v,b,gamma)
        assert clip_count>=0 and np.isfinite(adjusted).all()
        if gamma==0:assert np.array_equal(adjusted,original)
        for q in (.03,.15):
            m,observed,_=mix_at_share(b,adjusted,q)
            assert abs(observed-q)<1e-10
            assert temporal_features(10*np.log10(m))['max_step_rms_db']>=0
    m,share,_=mix_at_share(b,np.zeros((15,25)),0.)
    assert share==0 and np.array_equal(m,b)
    assert np.array_equal(v,original)
    print('SIMULATION REALISM NUMERICAL SELF-TEST: PASS')


def pick_sources(vt):
    clipmap={}; class_hulls={}
    data=vt.loc[vt.experiment004_partition.eq('exp004_train')]
    for uid,group in data.groupby('clip_uid',sort=True):
        group=group.sort_values('minute_index')
        if group.hull_id.nunique()!=1 or group.label.nunique()!=1 or not np.array_equal(group.minute_index.to_numpy(int),np.arange(len(group))):
            raise ValueError(f'Invalid training clip: {uid}')
        arr=group[COLS].to_numpy(float)
        if not np.isfinite(arr).all() or (arr<0).any() or arr.sum()<=0:
            raise ValueError(f'Invalid training clip powers: {uid}')
        hull=str(group.hull_id.iloc[0]); label=str(group.label.iloc[0])
        clipmap[str(uid)]=(hull,label,arr)
        class_hulls.setdefault(label,{}).setdefault(hull,[]).append(str(uid))
    return clipmap,class_hulls


def sample_one_clip(class_hulls,clip_map,rng,required_duration):
    # Restrict to clips long enough, then class -> hull -> clip, prespecified.
    qualified={}
    for label,hulls in class_hulls.items():
        eligible={h:[c for c in clips if clip_map[c][2].shape[0]>=required_duration]
                  for h,clips in hulls.items()}
        eligible={h:cs for h,cs in eligible.items() if cs}
        if eligible:qualified[label]=eligible
    labels=sorted(qualified)
    if not labels:raise ValueError('No clips support requested duration')
    label=labels[int(rng.integers(len(labels)))]
    hulls=sorted(qualified[label]);hull=hulls[int(rng.integers(len(hulls)))]
    uids=qualified[label][hull];uid=uids[int(rng.integers(len(uids)))]
    return uid,hull,label


def compute(input_root,frozen,output,seed,limit):
    from build_synthetic_mixtures_v1 import ParquetRowReader
    if output.exists():raise FileExistsError(f'Refusing to overwrite: {output}')
    hashes={}
    for name,expected in EXPECTED.items():
        p=frozen/name
        if not p.is_file() or sha(p)!=expected:raise RuntimeError(f'Frozen file hash mismatch: {p}')
        hashes[name]=sha(p)
    sample_path=input_root/'background_sampled_runs.csv'
    if not sample_path.is_file():raise FileNotFoundError(sample_path)
    hashes['background_sampled_runs.csv']=sha(sample_path)
    bg=pd.read_csv(frozen/'background_manifest_exp004.csv',low_memory=False)
    vt=pd.read_csv(frozen/'vessel_minute_manifest_exp004.csv',dtype={'hull_id':'string','clip_uid':'string'},low_memory=False)
    sample=pd.read_csv(sample_path,low_memory=False)
    if not {'station','run_id','start_time'}<=set(sample):raise ValueError('Sample index incomplete')
    if len(sample)!=len(sample[['station','run_id','start_time']].drop_duplicates()):raise ValueError('Duplicated sample keys')
    tr=bg.loc[bg.experiment004_partition.eq('exp004_train')].copy()
    selection=sample[['station','run_id','start_time']].merge(tr,on=['station','run_id','start_time'],how='left',validate='one_to_one',indicator=True)
    if not selection._merge.eq('both').all():raise ValueError('Sample includes non-training or missing background windows')
    selection=selection.drop(columns='_merge').sort_values(['station','run_id']).reset_index(drop=True)
    if limit is not None:
        # Test convenience only: per-station deterministic subset, not selected by model performance.
        selection=selection.groupby('station',group_keys=False,sort=True).head(limit).reset_index(drop=True)
    clip_map,classes=pick_sources(vt)
    if not clip_map:raise ValueError('Empty train vessel pool')
    rng=np.random.default_rng(seed);reader=ParquetRowReader();records=[]
    schedules=[(.03,1),(.03,3),(.15,1),(.15,3)]
    for i,row in enumerate(selection.itertuples(index=False)):
        levels=reader.read(str(row.source_parquet),json.loads(row.row_positions_json))
        b=db_to_relative_energy(np.asarray(levels,dtype=float))
        natural=temporal_features(np.asarray(levels,dtype=float))
        for q,duration in schedules:
            uid,hull,label=sample_one_clip(classes,clip_map,rng,duration)
            source=clip_map[uid][2]
            srcstart=int(rng.integers(len(source)-duration+1))
            insertion=5 if duration==3 else 7
            v=np.zeros((15,25),dtype=float)
            v[insertion:insertion+duration]=source[srcstart:srcstart+duration]
            for gamma,name in [(0.,'original_vtuad'),(.5,'halfway_shape'),(1.,'background_shaped_control')]:
                vv,clipped=reshape_source_spectrum(v,b,gamma)
                mixture,actual,alpha=mix_at_share(b,vv,q)
                assert abs(actual-q)<1e-9
                features=temporal_features(10*np.log10(np.maximum(mixture,1e-300)))
                record={'station':str(row.station),'background_run_id':str(row.run_id),
                        'background_start':str(row.start_time),'vessel_clip_uid':uid,
                        'vessel_hull_id':hull,'vessel_class':label,'duration_min':duration,
                        'insertion_min':insertion,'requested_share':q,'realized_share':actual,
                        'spectral_intervention':name,'gamma':gamma,'clipped_band_gains':clipped,
                        'natural_max_step_rms_db':natural['max_step_rms_db'],
                        'natural_mean_step_rms_db':natural['mean_step_rms_db'],
                        'injected_max_step_rms_db':features['max_step_rms_db'],
                        'injected_mean_step_rms_db':features['mean_step_rms_db'],
                        'change_in_max_step_rms_db':features['max_step_rms_db']-natural['max_step_rms_db'],
                        'change_in_mean_step_rms_db':features['mean_step_rms_db']-natural['mean_step_rms_db']}
                records.append(record)
        if (i+1)%100==0 or i+1==len(selection):print(f'Training background runs processed: {i+1}/{len(selection)}',flush=True)
    df=pd.DataFrame(records)
    if df.empty:raise ValueError('No generated comparisons')
    if not np.isfinite(df.select_dtypes(include='number').to_numpy()).all():raise ValueError('Nonfinite comparisons')
    if df.groupby(['background_run_id','requested_share','duration_min']).spectral_intervention.nunique().ne(3).any():
        raise ValueError('Incomplete matched source interventions')
    summary=(df.groupby(['spectral_intervention','requested_share','duration_min'],as_index=False)
             .agg(background_runs=('background_run_id','nunique'),
                  median_natural_step=('natural_max_step_rms_db','median'),
                  median_injected_step=('injected_max_step_rms_db','median'),
                  median_change_step=('change_in_max_step_rms_db','median'),
                  p90_change_step=('change_in_max_step_rms_db',lambda x:float(x.quantile(.9))),
                  fraction_increase=('change_in_max_step_rms_db',lambda x:float((x>0).mean())),
                  clipped_gain_examples=('clipped_band_gains',lambda x:int((x>0).sum()))))
    station_summary=(df.groupby(['station','spectral_intervention','requested_share','duration_min'],as_index=False)
                     .agg(background_runs=('background_run_id','nunique'),median_change_step=('change_in_max_step_rms_db','median'),
                          fraction_increase=('change_in_max_step_rms_db',lambda x:float((x>0).mean()))))
    # Atomic staging: no failed half-written final directory.
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.simulation_realism_',dir=output.parent) as scratch:
        stage=Path(scratch)/'data';stage.mkdir()
        df.to_csv(stage/'paired_temporal_changes.csv',index=False)
        summary.to_csv(stage/'scenario_summary.csv',index=False)
        station_summary.to_csv(stage/'station_scenario_summary.csv',index=False)
        manifest={'scope':'exp004_train ONLY; previously sampled train-run windows; no model fitting',
                  'input_hashes':hashes,'seed':seed,'runs_used':len(selection),
                  'source_hulls':int(vt.loc[vt.experiment004_partition.eq('exp004_train'),'hull_id'].nunique()),
                  'scenarios':'q=0.03/0.15, duration=1/3 minutes, insertion=7/5; gamma=0/0.5/1',
                  'method':'Per-window relative energy mix, fixed full-window target; gamma shape intervention changes source band powers before target scaling',
                  'assumptions':['Altered spectra are sensitivity controls, not realistic or isolated vessel acoustics',
                    'Natural adjacent-minute changes are not proven vessel-free negatives',
                    'Clip AIS labels and backgrounds have unknown/calibration-incompatible acoustic units',
                    'This compares temporal disturbance statistics, not actual vessel-detection performance',
                    'Runs are not independent deployments; no hidden or frozen evaluation data read'],
                  'files':{p.name:sha(p) for p in stage.glob('*.csv')}}
        (stage/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        output.mkdir(exist_ok=False)
        for p in stage.iterdir():p.replace(output/p.name)
    print('TRAIN-ONLY SIMULATION REALISM AUDIT: PASS')
    print(f'Background runs: {len(selection)}; matched scenario rows: {len(df)}')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Outputs:',output)
    print('LIMIT: Counterfactual sensitivity audit, not calibrated sound or test-domain accuracy.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('reports/audits/real_acoustic_development_v1'))
    p.add_argument('--frozen',type=Path,default=FROZEN)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--limit-runs-per-station',type=int)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args();self_test()
    if a.self_test_only:return
    if a.limit_runs_per_station is not None and a.limit_runs_per_station<1:raise ValueError('limit must be >=1')
    compute(a.input,a.frozen,a.output,a.seed,a.limit_runs_per_station)

if __name__=='__main__':main()
