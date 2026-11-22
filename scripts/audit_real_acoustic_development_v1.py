#!/usr/bin/env python3
"""Experiment 005 precursor: training-only real-acoustic domain audit.

Reads only exp004_train background windows and exp004_train vessel minute band powers.
Does not read exp004_eval acoustic measurements, original validation, sealed station
holdouts, Zindi Test.csv, or any labels. Spectral comparisons are RELATIVE SHAPE
only: VTUAD digital power and South African levels have no common calibration.

Run from project root:
 python scripts/audit_real_acoustic_development_v1.py --self-test-only
 python scripts/audit_real_acoustic_development_v1.py --max-runs-per-station 120
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path('reports/splits/experiment004_frozen_v1')
OUT = Path('reports/audits/real_acoustic_development_v1')
EXPECTED = {
 'background_manifest_exp004.csv':'a1d3dd26c08fbf7a2704c6535f416542179249a89a29522899b4d0b6448862bd',
 'vessel_minute_manifest_exp004.csv':'f33693c3fd96bca8ac4121af5df45b95240e04fa86f6bdbbc994c5794145f6c5',
 'split_protocol.json':'2df9ddcecf78cdd429867fa182f41f68745addc90a2f5634e59ccd1a644a5c55',
}
BANDS = ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']
COLS = ['TOL_'+b for b in BANDS]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()

def log_band_shape(levels):
 """Summed linear band energy then dB, normalized across frequency; no calibration."""
 a=np.asarray(levels,dtype=np.float64)
 if a.ndim!=2 or a.shape[1]!=25 or not np.isfinite(a).all():raise ValueError('bad level matrix')
 mx=float(a.max())
 e=np.power(10.,(a-mx)/10.)
 p=e.mean(axis=0)
 v=10.*np.log10(np.maximum(p,1e-300))
 return v-v.max()

def linear_power_shape(power):
 """VTUAD relative digital power, collapsed over time and dB normalized."""
 a=np.asarray(power,dtype=np.float64)
 if a.ndim!=2 or a.shape[1]!=25 or not np.isfinite(a).all() or (a<0).any() or a.sum()<=0:raise ValueError('bad power matrix')
 p=a.mean(axis=0)
 v=10.*np.log10(np.maximum(p,1e-300))
 return v-v.max()

def self_test():
 x=np.tile(np.linspace(40.,60.,25),(15,1))
 s=log_band_shape(x)
 assert s.shape==(25,) and abs(s.max())<1e-9
 assert np.allclose(s,log_band_shape(x+37))  # absolute shift is removed
 v=linear_power_shape(np.power(10.,x/10.))
 assert np.allclose(s,v,atol=1e-8)
 print('REAL ACOUSTIC SHAPE SELF-TEST: PASS')

def selected_run_windows(frame,max_runs,seed):
 """One prespecified window per randomly sampled run; prevents long-run domination."""
 rng=np.random.default_rng(seed)
 chunks=[]
 for station,station_frame in frame.groupby('station',sort=True):
  run_ids=station_frame.run_id.drop_duplicates().to_numpy()
  chosen=set(rng.choice(run_ids,size=min(len(run_ids),max_runs),replace=False).tolist())
  selected=station_frame.loc[station_frame.run_id.isin(chosen)].groupby('run_id',sort=True).sample(n=1,random_state=seed)
  chunks.append(selected)
 return pd.concat(chunks,ignore_index=True)

def audit(a):
 for name,want in EXPECTED.items():
  p=a.frozen/name
  if not p.is_file() or sha(p)!=want:raise RuntimeError(f'Frozen source hash mismatch: {p}')
 bg=pd.read_csv(a.frozen/'background_manifest_exp004.csv',low_memory=False)
 vt=pd.read_csv(a.frozen/'vessel_minute_manifest_exp004.csv',dtype={'hull_id':'string','clip_uid':'string'},low_memory=False)
 need_bg={'experiment004_partition','station','run_id','source_parquet','row_positions_json','recorder','start_time'}
 need_vt={'experiment004_partition','clip_uid','hull_id','label','minute_index',*COLS}
 if not need_bg<=set(bg) or not need_vt<=set(vt):raise ValueError('manifest schema mismatch')
 # The only acoustic readings below originate from the TRAIN partitions.
 bg=bg.loc[bg.experiment004_partition.eq('exp004_train')].copy()
 vt=vt.loc[vt.experiment004_partition.eq('exp004_train')].copy()
 if bg.empty or vt.empty:raise ValueError('empty training pool')
 picks=selected_run_windows(bg,a.max_runs_per_station,a.seed)
 from build_synthetic_mixtures_v1 import ParquetRowReader
 reader=ParquetRowReader()
 records=[]; shapes=[]
 for i,row in picks.iterrows():
  x=np.asarray(reader.read(str(row.source_parquet),json.loads(row.row_positions_json)),dtype=np.float64)
  if x.shape!=(15,25):raise ValueError(f'Unexpected background shape: {x.shape}')
  sh=log_band_shape(x)
  # These level statistics are recorded as observed level units only, NOT calibrated dB re 1 uPa.
  records.append({'station':str(row.station),'run_id':str(row.run_id),'recorder':str(row.recorder),
                  'start_time':str(row.start_time),'window_mean_level':float(x.mean()),
                  'window_min_level':float(x.min()),'window_max_level':float(x.max()),
                  'within_window_level_range':float(np.ptp(x)),
                  'within_minute_variability':float(np.mean(np.std(x,axis=0)))})
  shapes.append(sh)
  if (i+1)%100==0 or i+1==len(picks):print(f'Background runs read: {i+1}/{len(picks)}',flush=True)
 sample=pd.DataFrame(records)
 shapes=np.asarray(shapes)
 station_shapes=[]
 for station,g in sample.groupby('station',sort=True):
  idx=g.index.to_numpy()
  med=np.median(shapes[idx],axis=0)
  for j,b in enumerate(BANDS):
   station_shapes.append({'station':station,'band_hz':float(b),'median_relative_shape_db':float(med[j]),
                          'p10_relative_shape_db':float(np.quantile(shapes[idx,j],.1)),
                          'p90_relative_shape_db':float(np.quantile(shapes[idx,j],.9)),
                          'sampled_distinct_runs':len(g)})
 # Each clip contributes one relative-shape summary; never weight clips by their number of minutes.
 vrows=[]; vshapes=[]
 for uid,g in vt.groupby('clip_uid',sort=True):
  if g.hull_id.nunique()!=1 or g.label.nunique()!=1:raise ValueError('clip inconsistent')
  sh=linear_power_shape(g[COLS].to_numpy(float))
  vrows.append({'clip_uid':str(uid),'hull_id':str(g.hull_id.iloc[0]),'vessel_class':str(g.label.iloc[0]),'minutes':len(g)})
  vshapes.append(sh)
 vf=pd.DataFrame(vrows); vshapes=np.asarray(vshapes)
 vessel_shapes=[]
 for cl,g in vf.groupby('vessel_class',sort=True):
  idx=g.index.to_numpy()
  med=np.median(vshapes[idx],axis=0)
  for j,b in enumerate(BANDS):
   vessel_shapes.append({'vessel_class':cl,'band_hz':float(b),'median_relative_shape_db':float(med[j]),
                         'clips':len(g),'hulls':g.hull_id.nunique()})
 a.output.mkdir(parents=True,exist_ok=False)
 sample.to_csv(a.output/'background_sampled_runs.csv',index=False)
 pd.DataFrame(station_shapes).to_csv(a.output/'background_station_spectral_shapes.csv',index=False)
 vf.to_csv(a.output/'vessel_clip_inventory.csv',index=False)
 pd.DataFrame(vessel_shapes).to_csv(a.output/'vessel_class_spectral_shapes.csv',index=False)
 # Distinguish observed acoustic level from the shape-only cross-dataset comparison.
 stat=sample.groupby('station').agg(sampled_runs=('run_id','nunique'),recorders=('recorder','nunique'),
       mean_observed_level=('window_mean_level','mean'),median_observed_level=('window_mean_level','median'),
       median_within_window_range=('within_window_level_range','median'),
       median_temporal_band_variability=('within_minute_variability','median')).reset_index()
 stat.to_csv(a.output/'station_descriptive_summary.csv',index=False)
 # Descriptive spectral-shape distances only; not a learned model or calibration.
 pairs=[]
 for station in sorted(sample.station.unique()):
  b=np.median(shapes[sample.station.eq(station).to_numpy()],axis=0)
  for cl in sorted(vf.vessel_class.unique()):
   v=np.median(vshapes[vf.vessel_class.eq(cl).to_numpy()],axis=0)
   pairs.append({'station':station,'vessel_class':cl,
                 'median_shape_difference_rms_db':float(np.sqrt(np.mean((b-v)**2))),
                 'background_runs':int(sample.station.eq(station).sum()),
                 'vessel_clips':int(vf.vessel_class.eq(cl).sum())})
 pd.DataFrame(pairs).to_csv(a.output/'shape_distance_descriptive.csv',index=False)
 result={'scope':'Training-only descriptive acoustic audit. No evaluation or holdout acoustic measurements read.',
         'sample_rule':'One randomly selected 15-minute window per distinct sampled train run per station; deterministic seed',
         'seed':a.seed,'max_runs_per_station':a.max_runs_per_station,
         'background_sampled_runs_by_station':stat.set_index('station').sampled_runs.to_dict(),
         'vessel_train_clips':len(vf),'vessel_train_hulls':int(vf.hull_id.nunique()),
         'sampled_background_windows':len(sample),
         'frequency_shapes':'Each 25-band shape normalized to own strongest band; not absolute pressure levels',
         'warnings':['Different hydrophone sources/units are not physically calibrated or directly level-comparable',
                     'One Durban train run: station uncertainty cannot be estimated from independent runs',
                     'Temporal proximity/overlap between train and eval can remain despite disjoint run IDs',
                     'VTUAD source clips contain ambient contributions',
                     'No causal attribution to vessel movements from unlabeled South African acoustic data'],
         'input_sha256':{name:sha(a.frozen/name) for name in EXPECTED}}
 (a.output/'audit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print('TRAIN-ONLY REAL ACOUSTIC AUDIT: PASS')
 print(stat.to_string(index=False,float_format=lambda z:f'{z:.3f}'))
 print('Vessel clips:',len(vf),'hulls:',vf.hull_id.nunique())
 print('Reports:',a.output)
 print('CAUTION: Shape-only comparisons; source calibration and session independence not established.')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--frozen',type=Path,default=ROOT)
 p.add_argument('--output',type=Path,default=OUT)
 p.add_argument('--max-runs-per-station',type=int,default=120)
 p.add_argument('--seed',type=int,default=4204)
 p.add_argument('--self-test-only',action='store_true')
 a=p.parse_args()
 self_test()
 if a.self_test_only:return
 if a.max_runs_per_station<1:raise ValueError('max runs must be positive')
 if a.output.exists():raise FileExistsError(f'Refusing overwrite: {a.output}')
 audit(a)
if __name__=='__main__':main()
