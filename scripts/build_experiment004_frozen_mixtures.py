#!/usr/bin/env python3
"""Experiment 004: frozen-group-only synthetic mixtures, with observed-cell labels.

Use only the EXP004 frozen source manifests. Never reads Test.csv, public scores,
or previously inspected validation sources. Outputs are synthetic research evidence,
not calibrated absolute underwater acoustic pressures.

Invocation (from project root):
 python scripts/build_experiment004_frozen_mixtures.py --self-test-only
 python scripts/build_experiment004_frozen_mixtures.py --train 1200 --eval 300

Important: --eval is a single new-model evaluation resource. Do not iterate models
or masks against it. Missingness distribution is prespecified independently of test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_synthetic_mixtures_v1 import BANDS, SHAPE, ParquetRowReader, db_to_relative_energy, mix_at_share
from build_synthetic_mixtures_v2 import sample_fraction, embed_event

DEFAULT = Path('reports/splits/experiment004_frozen_v1')
OUT = Path('reports/eda/synthetic_experiment004_v1')
EXPECTED_FROZEN_SHA256 = {
    'background_manifest_exp004.csv': 'a1d3dd26c08fbf7a2704c6535f416542179249a89a29522899b4d0b6448862bd',
    'vessel_minute_manifest_exp004.csv': 'f33693c3fd96bca8ac4121af5df45b95240e04fa86f6bdbbc994c5794145f6c5',
    'split_protocol.json': '2df9ddcecf78cdd429867fa182f41f68745addc90a2f5634e59ccd1a644a5c55',
}
MASKS = {'complete': (), 'first': (0,), 'last': (14,),
         'middle': (7,), 'both_edges': (0,14), 'three_middle': (6,7,8)}
MASK_PROBS = (.50, .10, .10, .10, .10, .10)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def observed_fraction(B, V, alpha, observed):
    total = (B + alpha*V)[observed].sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Invalid observed energy denominator')
    return float((alpha*V)[observed].sum(dtype=np.float64) / total)

def impute(X):
    """Fixed, within-window, per-band time interpolation; no fitted statistics."""
    X = np.asarray(X, dtype=np.float64).copy()
    for band in range(25):
        good = np.isfinite(X[:,band])
        if not good.any():
            raise ValueError('No observed minute in band')
        if not good.all():
            t = np.arange(15)
            X[:,band] = np.interp(t, t[good], X[good,band])
    if X.shape != SHAPE or not np.isfinite(X).all():
        raise ValueError('Invalid imputed shape/values')
    return X

def self_test():
    rng = np.random.default_rng(99)
    B = rng.uniform(.1,3., SHAPE)
    V = np.zeros(SHAPE)
    V[6:9] = rng.uniform(.02,.3,(3,25))
    for q in (0.,.03,.25,.7):
        M,_,alpha = mix_at_share(B,V,q)
        obs = np.ones(SHAPE,dtype=bool)
        assert np.isclose(observed_fraction(B,V,alpha,obs),q,atol=1e-10)
        obs[6:9] = False
        assert observed_fraction(B,V,alpha,obs) == 0.
        L = 10*np.log10(M)
        L[6:9] = np.nan
        assert np.isfinite(impute(L)).all()
    print('EXPERIMENT 004 OBSERVED-CELL AND IMPUTATION SELF-TEST: PASS')

def eligible_vessel_pool(vt, part):
    clips = {}
    for uid, group in vt.loc[vt.experiment004_partition.eq(part)].groupby('clip_uid', sort=True):
        group = group.sort_values('minute_index')
        if group.hull_id.nunique() != 1 or group.label.nunique() != 1:
            raise ValueError('Inconsistent clip source metadata')
        if not np.array_equal(group.minute_index.to_numpy(int), np.arange(len(group))):
            raise ValueError(f'Nonconsecutive source minute indices: {uid}')
        arr = group[BANDS].to_numpy(np.float64)
        if not np.isfinite(arr).all() or (arr < 0).any() or not (arr.sum() > 0):
            raise ValueError('Invalid vessel source band power')
        clips[str(uid)] = (str(group.hull_id.iloc[0]),str(group.label.iloc[0]),arr)
    if not clips:
        raise ValueError(f'No vessel clips in {part}')
    by_class = {}
    for uid,(hull,label,_) in clips.items():
        by_class.setdefault(label,[]).append(uid)
    return clips,by_class

def select_backgrounds(pool,n,rng):
    """Allocate as evenly as possible among stations with windows available."""
    stations = sorted(pool.station.unique())
    remaining = {s:rng.permutation(pool.index[pool.station.eq(s)].to_numpy()).tolist()
                 for s in stations}
    counts = {s:0 for s in stations}
    selected = []
    while len(selected) < n:
        eligible = [s for s in stations if remaining[s]]
        if not eligible: raise ValueError('Insufficient background windows')
        minimum = min(counts[s] for s in eligible)
        tied = [s for s in eligible if counts[s] == minimum]
        s = tied[int(rng.integers(len(tied)))]
        selected.append(remaining[s].pop())
        counts[s] += 1
    return selected,counts

def validate_partition(bg,vt):
    bp = bg.experiment004_partition.astype(str)
    vp = vt.experiment004_partition.astype(str)
    for frame,allowed,name in [(bp,{'exp004_train','exp004_eval','existing_validation','existing_excluded','sealed_station_holdout'},'bg'),
                               (vp,{'exp004_train','exp004_eval','existing_validation'},'vessel')]:
        if not set(frame.unique()) <= allowed:
            raise ValueError(f'Unexpected {name} allocation labels')
    for f,key,name in [(bg,'run_id','background run'),(vt,'hull_id','vessel hull'),(vt,'clip_uid','vessel clip')]:
        tr = set(f.loc[f.experiment004_partition.eq('exp004_train'),key].dropna().astype(str))
        ev = set(f.loc[f.experiment004_partition.eq('exp004_eval'),key].dropna().astype(str))
        if not tr or not ev or tr.intersection(ev):
            raise ValueError(f'Empty or overlapping frozen {name} sets')
    if not bg.loc[bg.experiment004_partition.eq('sealed_station_holdout')].empty:
        assert not bg.loc[bg.experiment004_partition.eq('sealed_station_holdout'),'station'].isin(
            bg.loc[bp.isin(['exp004_train','exp004_eval']),'station']).any(), 'Sealed station leaked'

def generate(bg,vt,part,n,seed,out,reader):
    rng = np.random.default_rng(seed)
    bg = bg.loc[bg.experiment004_partition.eq(part)].reset_index(drop=True)
    selected,station_counts = select_backgrounds(bg,n,rng)
    clips,by_class = eligible_vessel_pool(vt,part)
    classes = sorted(by_class)
    X = np.empty((n,*SHAPE),dtype=np.float32)
    y = np.empty(n,dtype=np.float32)
    records = []
    masks = list(MASKS)
    for i,idx in enumerate(selected):
        row = bg.iloc[idx]
        positions = json.loads(row.row_positions_json)
        B = db_to_relative_energy(reader.read(str(row.source_parquet),positions))
        if not np.isfinite(B).all() or (B <= 0).any():
            raise ValueError('Invalid background energy')
        q,regime = sample_fraction(rng,.20)
        uid = hull = label = ''
        duration = source_offset = insertion = 0
        V = np.zeros(SHAPE,dtype=np.float64)
        if q:
            label = classes[int(rng.integers(len(classes)))]
            uids = by_class[label]
            uid = uids[int(rng.integers(len(uids)))]
            hull,_,rows = clips[uid]
            V,duration,source_offset,insertion = embed_event(rows,rng)
        M,full_share,alpha = mix_at_share(B,V,q)
        mask_name = masks[int(rng.choice(len(masks),p=MASK_PROBS))]
        missing = MASKS[mask_name]
        observed = np.ones(SHAPE,dtype=bool)
        observed[list(missing),:] = False
        observed_share = observed_fraction(B,V,alpha,observed)
        levels = 10*np.log10(np.maximum(M,1e-30))
        levels[list(missing),:] = np.nan
        X[i] = impute(levels).astype(np.float32)
        y[i] = observed_share
        records.append({'example':i,'partition':part,'station':str(row.station),
                        'background_run_id':str(row.run_id),'background_start':str(row.start_time),
                        'vessel_clip_uid':uid,'vessel_hull_id':hull,'vessel_class':label,
                        'vessel_duration_min':duration,'source_offset':source_offset,
                        'insertion_min':insertion,'target_regime':regime,'requested_full_share':q,
                        'realized_full_share':full_share,'observed_share':observed_share,
                        'observation_mask':mask_name,'missing_minute_positions':json.dumps(missing),
                        'observed_positive':int(observed_share>1e-12)})
        if (i+1)%200 == 0 or i+1 == n:
            print(f'{part}: {i+1}/{n}',flush=True)
    np.savez_compressed(out/f'{part}.npz',X=X,y=y)
    pd.DataFrame(records).to_csv(out/f'{part}_provenance.csv',index=False)
    print(f'{part}: station counts {station_counts}; positive observed share {(y>1e-12).sum()}/{n}')
    return pd.DataFrame(records)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--train',type=int,default=1200)
    p.add_argument('--eval',type=int,default=300)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--self-test-only',action='store_true')
    a = p.parse_args()
    self_test()
    if a.self_test_only: return
    if a.train < 1 or a.eval < 1: raise ValueError('Positive sample counts required')
    if a.output.exists(): raise FileExistsError(f'Refusing overwrite: {a.output}')
    bg_path = a.frozen/'background_manifest_exp004.csv'
    vt_path = a.frozen/'vessel_minute_manifest_exp004.csv'
    protocol = a.frozen/'split_protocol.json'
    inputs = [bg_path,vt_path,protocol]
    if not all(x.is_file() for x in inputs): raise FileNotFoundError('Frozen split incomplete')
    for path in inputs:
        if sha(path) != EXPECTED_FROZEN_SHA256[path.name]:
            raise ValueError(f'Frozen manifest SHA256 mismatch: {path}; refusing to generate')
    bg = pd.read_csv(bg_path,low_memory=False)
    vt = pd.read_csv(vt_path,dtype={'hull_id':'string','clip_uid':'string'},low_memory=False)
    required_bg = {'station','run_id','source_parquet','row_positions_json','start_time','experiment004_partition'}
    required_vt = {'hull_id','clip_uid','label','minute_index','experiment004_partition',*BANDS}
    if not required_bg <= set(bg) or not required_vt <= set(vt):
        raise ValueError('Frozen split schema mismatch')
    validate_partition(bg,vt)
    a.output.mkdir(parents=True,exist_ok=False)
    reader=ParquetRowReader()
    summaries={}
    for part,n,seed in [('exp004_train',a.train,a.seed),('exp004_eval',a.eval,a.seed+1000003)]:
        df=generate(bg,vt,part,n,seed,a.output,reader)
        summaries[part]={'examples':n,'background_runs':int(df.background_run_id.nunique()),
                         'vessel_hulls':int(df.loc[df.vessel_hull_id.ne(''),'vessel_hull_id'].nunique()),
                         'source_clips':int(df.loc[df.vessel_clip_uid.ne(''),'vessel_clip_uid'].nunique()),
                         'mask_counts':df.observation_mask.value_counts().to_dict()}
    tr = pd.read_csv(a.output/'exp004_train_provenance.csv',keep_default_na=False)
    ev = pd.read_csv(a.output/'exp004_eval_provenance.csv',keep_default_na=False)
    for key in ('background_run_id','vessel_hull_id','vessel_clip_uid'):
        left = set(tr[key])- {''}
        right = set(ev[key])- {''}
        if left & right: raise ValueError(f'Generated sample leakage in {key}')
    manifest={'experiment':'004 frozen-group synthetic baseline',
              'seed':a.seed,'eval_seed':a.seed+1000003,
              'partition_origin':'Original train pool only; historically touched in earlier experiments',
              'target':'Added linear relative power / combined power over OBSERVED minute-band cells',
              'observed_mask_probabilities':dict(zip(MASKS,MASK_PROBS)),
              'missing_handling':'Replace masked levels with within-window per-band linear interpolation for inputs ONLY',
              'simulation':'Background per-window maximum reference; vessel class-uniform sampling, hull-isolated splits',
              'physical_calibration':'Not calibrated; source-vessel audio contains ambient sound',
              'input_hashes':{str(x):sha(x) for x in inputs},'sampling':summaries,
              'output_hashes':{f.name:sha(f) for f in a.output.iterdir() if f.is_file()}}
    (a.output/'generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('EXPERIMENT 004 GENERATION: PASS; output:',a.output)
    print('NO original validation or sealed station holdout used. No Zindi Test.csv used.')

if __name__ == '__main__':main()
