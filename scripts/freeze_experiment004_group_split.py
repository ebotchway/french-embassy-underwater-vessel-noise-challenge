#!/usr/bin/env python3
"""Freeze a metadata-only run/hull-disjoint Experiment 004 split."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BG = Path('reports/eda/background_split_manifest_v2.csv')
DEFAULT_VT = Path('reports/eda/vtuad_minute_band_power_split_v2.csv')
DEFAULT_OUT = Path('reports/splits/experiment004_frozen_v1')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(frame, names, label):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'{label}: required columns missing {sorted(missing)}')


def clean_id(series):
    return series.astype('string').str.strip()


def stable_selection(groups, seed, tag, fraction):
    """Select nearest fraction by stable cryptographic ordering, retain >=1 group."""
    ids = sorted(set(str(x) for x in groups))
    if len(ids) < 2:
        return set()
    n = max(1, min(len(ids) - 1, int(np.floor(len(ids) * fraction + .5))))
    ranked = sorted(ids, key=lambda g: hashlib.sha256(f'{seed}|{tag}|{g}'.encode()).hexdigest())
    return set(ranked[:n])


def freeze(background: Path, vessels: Path, output: Path, seed: int, holdout_fraction: float):
    if not (0 < holdout_fraction < 0.5):
        raise ValueError('holdout fraction must be >0 and <0.5')
    if output.exists():
        raise FileExistsError(f'Frozen output already exists; refusing overwrite: {output}')
    bg = pd.read_csv(background, low_memory=False)
    vt = pd.read_csv(vessels, dtype={'hull_id':'string', 'clip_uid':'string'}, low_memory=False)
    require(bg, ['split','station','run_id','source_parquet','row_positions_json'], 'background')
    require(vt, ['source_split','hull_id','clip_uid','label','minute_index'], 'vessel')
    bg = bg.copy(); vt = vt.copy()
    for col in ['split','station','run_id']:
        bg[col] = clean_id(bg[col])
    for col in ['source_split','hull_id','clip_uid','label']:
        vt[col] = clean_id(vt[col])
    if bg[['split','station','run_id']].isna().any().any() or vt[['source_split','clip_uid','label']].isna().any().any():
        raise ValueError('Missing partition, station, run, clip, or label IDs')
    for f, cols in [(bg,['station','run_id']), (vt,['clip_uid','hull_id','label'])]:
        if (f[cols] == '').any().any():
            raise ValueError('Empty group identifier/label')
    if not set(bg.split.dropna()).issubset({'train','validation','excluded','station_holdout'}):
        raise ValueError('Unexpected original background split name')
    if not set(vt.source_split.dropna()).issubset({'train','validation'}):
        raise ValueError('Unexpected original vessel split name')
    if vt.duplicated(['clip_uid','minute_index']).any():
        raise ValueError('Duplicate vessel clip/minute combination')
    clip_groups = vt.groupby('clip_uid',dropna=False)
    if (clip_groups.hull_id.nunique(dropna=False)>1).any() or (clip_groups.label.nunique(dropna=False)>1).any() or (clip_groups.source_split.nunique()>1).any():
        raise ValueError('Clip has inconsistent hull, label, or original split')
    if (vt.loc[vt.source_split.eq('train'),'hull_id'].isna()).any():
        raise ValueError('Missing hull ID in training pool: cannot ensure independent hull split')
    if (vt.loc[vt.source_split.eq('train'),'hull_id'] == '').any():
        raise ValueError('Blank hull ID in training pool')
    # If a run ID corresponds to multiple stations or partitions, isolation is ambiguous.
    if (bg.groupby('run_id').station.nunique()>1).any() or (bg.groupby('run_id').split.nunique()>1).any():
        raise ValueError('Background run_id spans stations or existing splits')
    if (vt.groupby('hull_id',dropna=False).source_split.nunique()>1).any():
        raise ValueError('Vessel hull already crosses original train/validation split')

    bg_train = bg[bg.split.eq('train')]
    vt_train = vt[vt.source_split.eq('train')]
    if bg_train.empty or vt_train.empty:
        raise ValueError('One of the existing training pools is empty')
    # Per-station reservation prevents an entire station from disappearing from training.
    selected_runs = set()
    station_rows = []
    for station, group in bg_train.groupby('station', sort=True):
        all_runs = sorted(group.run_id.unique().tolist())
        chosen = stable_selection(all_runs,seed,f'background:{station}',holdout_fraction)
        selected_runs.update(chosen)
        station_rows.append({'station':str(station),'available_runs':len(all_runs),
                             'reserved_eval_runs':len(chosen),'remaining_train_runs':len(all_runs)-len(chosen),
                             'reserved_eval_windows':int(group.run_id.isin(chosen).sum()),
                             'remaining_train_windows':int((~group.run_id.isin(chosen)).sum())})
    if not selected_runs:
        raise ValueError('No background source runs eligible for reservation')
    # Prefer each label represented by >=1 hull in both subsets; hull is the atomic unit.
    hull_labels = vt_train[['hull_id','label']].drop_duplicates()
    if hull_labels.groupby('hull_id').label.nunique().max() > 1:
        raise ValueError('A hull spans multiple vessel classes; require explicit allocation policy')
    selected_hulls = set()
    class_rows = []
    for label, g in hull_labels.groupby('label',sort=True):
        all_hulls = sorted(g.hull_id.unique().tolist())
        chosen = stable_selection(all_hulls,seed,f'vessel:{label}',holdout_fraction)
        selected_hulls.update(chosen)
        class_rows.append({'label':str(label),'available_hulls':len(all_hulls),
                           'reserved_eval_hulls':len(chosen),'remaining_train_hulls':len(all_hulls)-len(chosen)})
    if not selected_hulls:
        raise ValueError('No vessel hulls eligible for reservation')

    bg['experiment004_partition'] = bg.split.map({'train':'exp004_train','validation':'existing_validation',
                                                   'excluded':'existing_excluded','station_holdout':'sealed_station_holdout'})
    vt['experiment004_partition'] = vt.source_split.map({'train':'exp004_train','validation':'existing_validation'})
    bg.loc[bg.split.eq('train') & bg.run_id.isin(selected_runs),'experiment004_partition'] = 'exp004_eval'
    vt.loc[vt.source_split.eq('train') & vt.hull_id.isin(selected_hulls),'experiment004_partition'] = 'exp004_eval'
    btrain = set(bg.loc[bg.experiment004_partition.eq('exp004_train'),'run_id'])
    beval = set(bg.loc[bg.experiment004_partition.eq('exp004_eval'),'run_id'])
    vtrain = set(vt.loc[vt.experiment004_partition.eq('exp004_train'),'hull_id'])
    veval = set(vt.loc[vt.experiment004_partition.eq('exp004_eval'),'hull_id'])
    ctrain = set(vt.loc[vt.experiment004_partition.eq('exp004_train'),'clip_uid'])
    ceval = set(vt.loc[vt.experiment004_partition.eq('exp004_eval'),'clip_uid'])
    if not beval or not veval or btrain & beval or vtrain & veval or ctrain & ceval:
        raise AssertionError('Train/evaluation run, hull or clip overlap, or empty reserve')
    if bg.loc[~bg.split.eq('train'),'experiment004_partition'].isin(['exp004_train','exp004_eval']).any():
        raise AssertionError('Protected background partition was changed')
    if vt.loc[~vt.source_split.eq('train'),'experiment004_partition'].isin(['exp004_train','exp004_eval']).any():
        raise AssertionError('Existing vessel validation partition was changed')

    output.mkdir(parents=True, exist_ok=False)
    bg_path = output/'background_manifest_exp004.csv'
    vt_path = output/'vessel_minute_manifest_exp004.csv'
    bg.to_csv(bg_path,index=False)
    vt.to_csv(vt_path,index=False)
    pd.DataFrame(station_rows).to_csv(output/'station_reservation_counts.csv',index=False)
    pd.DataFrame(class_rows).to_csv(output/'vessel_class_reservation_counts.csv',index=False)
    meta = {
        'design':'Experiment004 frozen group allocation v1',
        'seed':seed,'holdout_fraction':holdout_fraction,
        'source_manifest_sha256':{'background':digest(background),'vessels':digest(vessels)},
        'frozen_manifest_sha256':{'background':digest(bg_path),'vessels':digest(vt_path)},
        'background_partition_rows':bg.experiment004_partition.value_counts().to_dict(),
        'vessel_partition_minute_rows':vt.experiment004_partition.value_counts().to_dict(),
        'background_eval_runs':len(beval),'vessel_eval_hulls':len(veval),'vessel_eval_clips':len(ceval),
        'assertions':{'train_eval_run_disjoint':True,'train_eval_hull_disjoint':True,'train_eval_clip_disjoint':True,
                      'existing_validation_untouched':True,'sealed_station_holdout_untouched':True},
        'limitations':[
            'Reserve was selected from the prior training SOURCE POOL, not never-seen historical data.',
            'Prior Experiment002 models have trained on some of these groups; old models cannot be evaluated as independent on this reserve.',
            'For a new Experiment004 model, training must EXCLUDE all exp004_eval runs and hulls, and include only exp004_train.',
            'Allocation is based only on metadata and deterministic hashes, not acoustic values or predictions.',
            'This alone does not create independent stations or a real measured ground-truth test set.',
            'After reporting model accuracy on exp004_eval, stop using that reserve for model selection.'
        ]}
    (output/'split_protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('EXPERIMENT 004 GROUP SPLIT: PASS')
    print('Background window allocations:',meta['background_partition_rows'])
    print('Vessel minute allocations:',meta['vessel_partition_minute_rows'])
    print(f'Exp004 evaluation: {len(beval)} background runs, {len(veval)} vessel hulls, {len(ceval)} clips')
    print('Train/eval run, hull, clip overlap: 0, 0, 0')
    print('Output:',output)
    print('CAUTION: Historical training pool reuse; independence applies only to NEW models trained with exp004_train.')
    return meta


def self_test():
    with tempfile.TemporaryDirectory() as t:
        base=Path(t)
        bg_rows=[]
        for station,runs in [('A',4),('B',3),('C',1)]:
            for r in range(runs):
                for w in range(2):
                    bg_rows.append({'split':'train','station':station,'run_id':f'{station}{r}',
                        'source_parquet':'x.parquet','row_positions_json':'[0]'})
        bg_rows += [{'split':'validation','station':'V','run_id':'V0','source_parquet':'x.parquet','row_positions_json':'[0]'},
                    {'split':'station_holdout','station':'H','run_id':'H0','source_parquet':'x.parquet','row_positions_json':'[0]'}]
        vt_rows=[]
        for label,hulls in [('ship',4),('tug',3),('pilot',1)]:
            for k in range(hulls):
                for minute in range(2):
                    vt_rows.append({'source_split':'train','hull_id':f'{label}_{k}',
                                    'clip_uid':f'{label}_c{k}','label':label,'minute_index':minute})
        vt_rows += [{'source_split':'validation','hull_id':'val_hull','clip_uid':'val_clip',
                     'label':'ship','minute_index':0}]
        bp=base/'bg.csv';vp=base/'vt.csv'
        pd.DataFrame(bg_rows).to_csv(bp,index=False)
        pd.DataFrame(vt_rows).to_csv(vp,index=False)
        out=base/'frozen'
        m=freeze(bp,vp,out,2026,.20)
        assert m['background_eval_runs'] >=2 and m['vessel_eval_hulls'] >=2
        try:
            freeze(bp,vp,out,2026,.20)
            raise AssertionError('Overwrite was allowed')
        except FileExistsError:pass
        print('SELF-TEST: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',type=Path,default=DEFAULT_BG)
    p.add_argument('--vessels',type=Path,default=DEFAULT_VT)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--fraction',type=float,default=.20)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only:
        self_test()
    else:
        freeze(a.background,a.vessels,a.output,a.seed,a.fraction)

if __name__=='__main__':main()
