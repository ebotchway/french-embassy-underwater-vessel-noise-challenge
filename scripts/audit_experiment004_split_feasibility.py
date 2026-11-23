#!/usr/bin/env python3
"""Experiment 004: metadata-only feasibility audit for independent source evaluation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def require(df, cols, name):
    absent=set(cols)-set(df.columns)
    if absent: raise ValueError(f'{name} missing columns: {sorted(absent)}')


def category(s):
    return s.fillna('<missing>').astype(str).str.strip().replace('', '<missing>')


def overlap_table(df, part_col, group_col):
    z=df[[part_col,group_col]].copy()
    z[part_col]=category(z[part_col]); z[group_col]=category(z[group_col])
    z=z[z[group_col]!='<missing>']
    cross=z.groupby(group_col)[part_col].agg(lambda v:'|'.join(sorted(set(v))))
    shared=cross[cross.str.contains('train') & cross.str.contains('validation')]
    return shared.reset_index(name='partitions')


def run(args):
    bg=pd.read_csv(args.background,low_memory=False)
    vt=pd.read_csv(args.vessels,dtype={'hull_id':'string'},low_memory=False)
    require(bg,['split','station','run_id','source_parquet','row_positions_json'], 'background')
    require(vt,['source_split','clip_uid','hull_id','label','minute_index'], 'vessel')
    bg['split']=category(bg['split'])
    vt['source_split']=category(vt['source_split'])
    bg['run_id']=category(bg['run_id'])
    vt['clip_uid']=category(vt['clip_uid'])
    vt['hull_id']=category(vt['hull_id'])
    if bg['run_id'].eq('<missing>').any() or vt['clip_uid'].eq('<missing>').any():
        raise ValueError('Missing recording-run or vessel clip IDs; source audit cannot guarantee isolation')
    if vt.duplicated(['clip_uid','minute_index']).any():
        raise ValueError('Duplicate clip/minute entries')
    if bg['split'].eq('<missing>').any() or vt['source_split'].eq('<missing>').any():
        raise ValueError('Missing partition assignment')
    # Avoid loading any background samples or reservoir/holdout audio.
    bg_summary=(bg.groupby(['station','split'],dropna=False).agg(
        indexed_windows=('run_id','size'),background_runs=('run_id','nunique'),
        parquet_files=('source_parquet','nunique')).reset_index().sort_values(['station','split']))
    vessel_summary=(vt.groupby(['source_split','label'],dropna=False).agg(
        minute_rows=('clip_uid','size'),clips=('clip_uid','nunique'),hulls=('hull_id','nunique'))
        .reset_index().sort_values(['source_split','label']))
    bg_over=overlap_table(bg[bg.split.isin(['train','validation'])],'split','run_id')
    hull_over=overlap_table(vt[vt.source_split.isin(['train','validation'])],'source_split','hull_id')
    clip_over=overlap_table(vt[vt.source_split.isin(['train','validation'])],'source_split','clip_uid')
    # Candidate feasibility is counts only: never allocate or expose held-out sample values.
    available=bg[bg.split=='train'].groupby('station').agg(
        candidate_train_windows=('run_id','size'),candidate_train_runs=('run_id','nunique')).reset_index()
    available['can_reserve_one_run_and_leave_one_run_for_training']=available.candidate_train_runs.ge(2)
    vessel_pool=vt[vt.source_split=='train'].drop_duplicates('clip_uid')
    vessel_feasibility=(vessel_pool.groupby('label').agg(
        candidate_train_clips=('clip_uid','size'),candidate_train_hulls=('hull_id','nunique')).reset_index())
    vessel_feasibility['can_reserve_one_hull_and_leave_one_hull_for_training']=vessel_feasibility.candidate_train_hulls.ge(2)
    reserved=bg[~bg.split.isin(['train','validation'])].groupby(['split','station']).agg(
        reserved_windows=('run_id','size'),reserved_runs=('run_id','nunique')).reset_index()
    flags=[]
    if not bg_over.empty: flags.append('background run overlaps train/validation')
    if not hull_over.empty: flags.append('vessel hull overlaps train/validation')
    if not clip_over.empty: flags.append('vessel clip overlaps train/validation')
    missing_hulls=int((vt['hull_id']=='<missing>').sum())
    if missing_hulls: flags.append(f'{missing_hulls} vessel minute rows have unavailable hull IDs')
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    csvs={
        'station_partition_inventory.csv':bg_summary,
        'vessel_class_partition_inventory.csv':vessel_summary,
        'background_run_overlap.csv':bg_over,
        'vessel_hull_overlap.csv':hull_over,
        'vessel_clip_overlap.csv':clip_over,
        'train_background_reservation_feasibility.csv':available,
        'train_vessel_reservation_feasibility.csv':vessel_feasibility,
        'nondevelopment_partition_counts_only.csv':reserved,
    }
    for name,frame in csvs.items(): frame.to_csv(out/name,index=False)
    report={
        'purpose':'Experiment 004 preregistration planning, metadata-only',
        'manifest_hashes_sha256':{'background':sha256(args.background),'vessels':sha256(args.vessels)},
        'background_rows':len(bg),'vessel_minute_rows':len(vt),
        'background_splits':bg.split.value_counts().sort_index().to_dict(),
        'vessel_splits':vt.source_split.value_counts().sort_index().to_dict(),
        'background_train_validation_run_overlap':len(bg_over),
        'vessel_train_validation_hull_overlap':len(hull_over),
        'vessel_train_validation_clip_overlap':len(clip_over),
        'train_stations_with_two_or_more_runs':int(available.can_reserve_one_run_and_leave_one_run_for_training.sum()),
        'train_vessel_classes_with_two_or_more_hulls':int(vessel_feasibility.can_reserve_one_hull_and_leave_one_hull_for_training.sum()),
        'flags':flags,
        'guardrails':['No new split has been created','No acoustic values from reserved stations read',
                      'Counts do not imply independent evaluation: design needs preregistered run/hull isolation',
                      'Existing validation was previously inspected and must remain development evidence',
                      'Avoid choosing holdout runs/hulls using acoustic values, errors or leaderboard results'],
    }
    (out/'feasibility_summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('EXPERIMENT 004 SPLIT FEASIBILITY: '+('PASS' if not flags else 'REVIEW FLAGS'))
    print('Background:',len(bg),'indexed windows; vessel:',len(vt),'minute feature rows')
    print('Background splits:',report['background_splits'])
    print('Vessel splits:',report['vessel_splits'])
    print('Train/validation overlap: runs=',len(bg_over),'hulls=',len(hull_over),'clips=',len(clip_over))
    print('Train stations with >=2 distinct runs:',report['train_stations_with_two_or_more_runs'])
    print('Train vessel classes with >=2 distinct hulls:',report['train_vessel_classes_with_two_or_more_hulls'])
    if flags: print('FLAGS:', '; '.join(flags))
    print('Reports:',out)
    print('No new split, new model, or reserved acoustic analysis performed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',type=Path,default=Path('reports/eda/background_split_manifest_v2.csv'))
    p.add_argument('--vessels',type=Path,default=Path('reports/eda/vtuad_minute_band_power_split_v2.csv'))
    p.add_argument('--output',type=Path,default=Path('reports/audits/experiment004_split_feasibility'))
    run(p.parse_args())

if __name__=='__main__':main()
