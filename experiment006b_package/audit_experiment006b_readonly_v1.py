#!/usr/bin/env python3
"""Experiment 006B: READ-ONLY source/protocol audit of Experiment 006.

Uses ONLY the Experiment 006 outputs, never the synthetic source archives,
Experiment 004 evaluation/validation/holdouts, or Zindi test data.

Examples:
 python scripts/audit_experiment006b_readonly_v1.py --self-test-only
 python scripts/audit_experiment006b_readonly_v1.py
 python scripts/audit_experiment006b_readonly_v1.py --bootstrap 1000 --output reports/audits/experiment006b_v2

Statistical interval: paired development-background-pair bootstrap, conditional on
this single internal split and its reused vessel sources. Hull and station
sensitivities are separately reported, not mistakenly called independent CIs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

DEFAULT_INPUT=Path('reports/baselines/experiment006_internal_contrast_v1')
DEFAULT_OUT=Path('reports/audits/experiment006b_readonly_v1')
REQUIRED={
 'matched_development_predictions.csv','training_provenance.csv','development_provenance.csv',
 'internal_background_group_allocation.csv','internal_vessel_group_allocation.csv',
 'development_summary.csv','experiment_manifest.json'
}
MODELS=('original_only','contrast_varied')
CONDITIONS=('no_addition','original_vtuad','halfway_shape','background_shaped_control')
POS=('original_vtuad','halfway_shape','background_shaped_control')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()

def ensure(condition,message):
 if not condition:raise ValueError(message)

def read_and_verify(root):
 missing=REQUIRED-{p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else REQUIRED
 ensure(not missing,f'Missing experiment files: {sorted(missing)}')
 man=json.loads((root/'experiment_manifest.json').read_text())
 for name,sha in man.get('output_sha256',{}).items():
  ensure((root/name).is_file(),f'Manifest file missing: {name}')
  ensure(digest(root/name)==sha,f'Output checksum mismatch: {name}')
 return ({key:pd.read_csv(root/key,low_memory=False) for key in REQUIRED if key.endswith('.csv')},man)

def validate(tables,manifest):
 pred=tables['matched_development_predictions.csv']; tr=tables['training_provenance.csv'];
 dev=tables['development_provenance.csv'];bg=tables['internal_background_group_allocation.csv'];
 vh=tables['internal_vessel_group_allocation.csv']
 required=['pair','model','station','background_run_id','vessel_clip_uid','vessel_hull_id',
           'requested_share','observed_share','spectral_intervention','estimated_share',
           'detection_evidence','mask','duration_min']
 ensure(not (set(required)-set(pred.columns)),f'Missing prediction columns {set(required)-set(pred.columns)}')
 ensure(pred[required].notna().all().all(),'Unexpected NULL in required prediction values')
 ensure(set(pred.model)==set(MODELS),'Model identities missing/unexpected')
 ensure(set(pred.spectral_intervention)==set(CONDITIONS),'Condition identities missing/unexpected')
 ensure(pred[['estimated_share','detection_evidence','observed_share']].apply(lambda s:pd.to_numeric(s,errors='coerce')).notna().all().all(),'Non-numeric scores')
 for col in ['estimated_share','detection_evidence','observed_share']:
  ensure(pred[col].between(0,1).all(),f'Out-of-range {col}')
 ensure(pred.requested_share.isin([0.,.03,.15]).all(),'Unexpected requested share')
 ensure(len(pred)==2*len(dev),'Prediction/development row count mismatch')
 ensure(dev['pair'].nunique()==manifest['dev_background_pairs'],'Manifest dev pair-count mismatch')
 ensure(len(dev)==7*dev['pair'].nunique(),'Expected one zero + six positive scenario rows per pair')
 ensure(len(tr)==manifest['train_samples_per_model'],'Training example-count mismatch')
 id_cols=['pair','station','background_run_id','vessel_clip_uid','vessel_hull_id',
          'requested_share','observed_share','spectral_intervention','mask','duration_min']
 sorted_cols=['pair','requested_share','spectral_intervention']
 p1=pred[pred.model==MODELS[0]].sort_values(sorted_cols).reset_index(drop=True)
 p2=pred[pred.model==MODELS[1]].sort_values(sorted_cols).reset_index(drop=True)
 ensure(len(p1)==len(p2),'Unequal model prediction counts')
 pd.testing.assert_frame_equal(p1[id_cols],p2[id_cols],check_dtype=False,check_exact=True)
 d=dev.sort_values(sorted_cols).reset_index(drop=True)
 pd.testing.assert_frame_equal(p1[id_cols],d[id_cols],check_dtype=False,check_exact=True)
 for _,chunk in d.groupby('pair',sort=False):
  ensure(len(chunk)==7,'Bad scenario count in pair')
  ensure(chunk.spectral_intervention.value_counts().to_dict()=={'no_addition':1,'original_vtuad':2,'halfway_shape':2,'background_shaped_control':2},'Unbalanced scenario triplets')
  ensure((chunk.loc[chunk.spectral_intervention=='no_addition','requested_share']==0).all(),'Zero control incorrect')
  ensure((chunk.loc[chunk.spectral_intervention!='no_addition','requested_share']>0).all(),'Positive scenario incorrect')
  for col in ('station','background_run_id','vessel_clip_uid','vessel_hull_id','mask','duration_min'):
   ensure(chunk[col].nunique()==1,f'Pair-level mismatch {col}')
 ensure(not tr.background_run_id.astype(str).isin(dev.background_run_id.astype(str)).any(),'BACKGROUND RUN LEAKAGE')
 ensure(not tr.vessel_hull_id.dropna().astype(str).isin(dev.vessel_hull_id.dropna().astype(str)).any(),'HULL LEAKAGE')
 ensure(not tr.vessel_clip_uid.dropna().astype(str).isin(dev.vessel_clip_uid.dropna().astype(str)).any(),'CLIP LEAKAGE')
 for frame,name,key in [(bg,'background','run_id'),(vh,'vessel','hull_id')]:
  ensure(frame[key].is_unique,f'Duplicate {name} allocation keys')
  ensure(set(frame.allocation)=={'internal_train','internal_dev'},f'Unexpected {name} allocation labels')
  m=frame.set_index(key).allocation.to_dict()
  for part,block in [('internal_train',tr),('internal_dev',dev)]:
   fld='background_run_id' if name=='background' else 'vessel_hull_id'
   vals=block[fld].dropna().astype(str)
   if name=='vessel':vals=vals[vals!='']
   ensure(vals.map(m).eq(part).all(),f'{name} allocation mismatch for {part}')
 return p1,p2,dev,tr

def metrics(frame):
 yy=frame.observed_share.to_numpy(float);pp=frame.estimated_share.to_numpy(float)
 ss=frame.detection_evidence.to_numpy(float);label=(yy>1e-12).astype(int)
 return {'n':len(frame),'pairs':frame.pair.nunique(),
         'rmse':np.sqrt(np.mean((pp-yy)**2)),'mae':np.mean(abs(pp-yy)),
         'auc':roc_auc_score(label,ss) if label.min()!=label.max() else np.nan,
         'ap':average_precision_score(label,ss) if label.min()!=label.max() else np.nan,
         'fpr_observed_zero_at_05':float(np.mean(ss[label==0]>=.5)) if np.any(label==0) else np.nan,
         'fpr_no_addition_at_05':float(np.mean(frame.loc[frame.spectral_intervention.eq('no_addition'),'detection_evidence']>=.5)) if (frame.spectral_intervention=='no_addition').any() else np.nan,
         'masked_positive_zero_count':int(((frame.requested_share>0)&(yy<=1e-12)).sum())}

def contrasts(p1,p2):
 key=['pair','requested_share','spectral_intervention']
 merged=p1.merge(p2,on=key,validate='one_to_one',suffixes=('_original','_varied'))
 ensure(len(merged)==len(p1),'Unmatched rows')
 yy=merged.observed_share_original.to_numpy(float)
 ensure(np.allclose(yy,merged.observed_share_varied.to_numpy(float),atol=0,rtol=0),'Target mismatch')
 merged['sq_gain']=(merged.estimated_share_original-yy)**2-(merged.estimated_share_varied-yy)**2
 merged['abs_gain']=abs(merged.estimated_share_original-yy)-abs(merged.estimated_share_varied-yy)
 return merged

def boot_metrics(p1,p2,merged,rounds,seed):
 rng=np.random.default_rng(seed)
 pairs=np.array(sorted(p1.pair.unique()))
 # One paired draw per background pair; same weights for both models.
 split1={k:g for k,g in p1.groupby('pair')};split2={k:g for k,g in p2.groupby('pair')}
 summary=[]
 for i in range(rounds):
  ids=rng.choice(pairs,len(pairs),replace=True)
  x=pd.concat([split1[k] for k in ids],ignore_index=True)
  y=pd.concat([split2[k] for k in ids],ignore_index=True)
  a=metrics(x);b=metrics(y)
  summary.append({'bootstrap':i,'rmse_reduction':a['rmse']-b['rmse'],
                  'mae_reduction':a['mae']-b['mae'],
                  'auc_gain':b['auc']-a['auc'],
                  'ap_gain':b['ap']-a['ap'],
                  'fpr_no_addition_increase':b['fpr_no_addition_at_05']-a['fpr_no_addition_at_05'],
                  'fpr_observed_zero_increase':b['fpr_observed_zero_at_05']-a['fpr_observed_zero_at_05']})
 return pd.DataFrame(summary)

def threshold_table(p1,p2):
 out=[]
 for model,frame in [(MODELS[0],p1),(MODELS[1],p2)]:
  for scope,data in [('overall',frame),*[(f'station:{k}',g) for k,g in frame.groupby('station')],*[(f'mask:{k}',g) for k,g in frame.groupby('mask')]]:
   label=(data.observed_share>1e-12).to_numpy();none=(data.spectral_intervention=='no_addition').to_numpy()
   s=data.detection_evidence.to_numpy(float)
   for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
    out.append({'model':model,'scope':scope,'threshold':threshold,
                'n':len(data),'n_no_addition':int(none.sum()),'n_observed_zero':int((~label).sum()),
                'tpr_observed_positive':float(np.mean(s[label]>=threshold)) if label.any() else np.nan,
                'fpr_observed_zero':float(np.mean(s[~label]>=threshold)) if (~label).any() else np.nan,
                'fpr_no_addition':float(np.mean(s[none]>=threshold)) if none.any() else np.nan})
 return pd.DataFrame(out)

def segment_table(p1,p2):
 out=[]
 groups=[('overall','all',lambda z:z),
         ('zero','no_addition',lambda z:z[z.spectral_intervention=='no_addition'])]
 for field in ('station','mask','requested_share','spectral_intervention'):
  for val in sorted(p1[field].unique(),key=str):
   groups.append((field,str(val),lambda z,field=field,val=val:z[z[field]==val]))
 for group,scenario,select in groups:
  a=metrics(select(p1));b=metrics(select(p2))
  for model,m in [(MODELS[0],a),(MODELS[1],b)]:
   out.append({'group':group,'scenario':scenario,'model':model,**m})
 return pd.DataFrame(out)

def deletion_table(p1,p2):
 out=[]
 for field in ('station','background_run_id','vessel_hull_id','vessel_clip_uid'):
  for group in sorted(p1[field].dropna().unique(),key=str):
   z1=p1[p1[field]!=group];z2=p2[p2[field]!=group]
   if not len(z1):continue
   a=metrics(z1);b=metrics(z2)
   out.append({'deleted_type':field,'deleted_group':str(group), 'pairs_remaining':z1.pair.nunique(),
               'rmse_reduction':a['rmse']-b['rmse'],
               'auc_gain':b['auc']-a['auc'],
               'fpr_no_addition_increase':b['fpr_no_addition_at_05']-a['fpr_no_addition_at_05']})
 return pd.DataFrame(out)

def analyze(root,out,bootstrap,seed):
 ensure(not out.exists(),f'Output exists; refusing overwrite: {out}')
 tables,manifest=read_and_verify(root)
 p1,p2,dev,tr=validate(tables,manifest)
 merged=contrasts(p1,p2)
 sampled=boot_metrics(p1,p2,merged,bootstrap,seed)
 seg=segment_table(p1,p2);thresholds=threshold_table(p1,p2);deletion=deletion_table(p1,p2)
 m1=metrics(p1);m2=metrics(p2)
 delta={'rmse_reduction':m1['rmse']-m2['rmse'],'mae_reduction':m1['mae']-m2['mae'],
        'auc_gain':m2['auc']-m1['auc'],'ap_gain':m2['ap']-m1['ap'],
        'fpr_no_addition_increase':m2['fpr_no_addition_at_05']-m1['fpr_no_addition_at_05'],
        'fpr_observed_zero_increase':m2['fpr_observed_zero_at_05']-m1['fpr_observed_zero_at_05']}
 ci=[]
 for metric,value in delta.items():
  finite=sampled[metric].dropna().to_numpy(float)
  lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
  ci.append({'metric':metric,'point_estimate':value,'pair_bootstrap_95_low':lo,'pair_bootstrap_95_high':hi,
             'bootstrap_non_null':len(finite)})
 ci=pd.DataFrame(ci)
 # Provenance diagnostics: repeated hulls/ clips/ background runs are visible.
 devpairs=dev.drop_duplicates('pair')
 src={'background_pairs':len(devpairs),'distinct_background_runs':devpairs.background_run_id.nunique(),
      'distinct_vessel_hulls':devpairs.vessel_hull_id.nunique(),
      'distinct_vessel_clips':devpairs.vessel_clip_uid.nunique(),
      'train_examples':len(tr),'train_runs':tr.background_run_id.nunique(),
      'train_hulls':tr.vessel_hull_id.replace('',np.nan).nunique(),
      'train_clips':tr.vessel_clip_uid.replace('',np.nan).nunique(),
      'masked_positive_with_zero_observed_share':int(((p1.requested_share>0)&(p1.observed_share<=1e-12)).sum())}
 note=['# Experiment 006B — Read-only development audit','',
       'Source: existing Experiment 006 prediction, provenance, allocations, manifest. No retraining, frozen evaluation, or hidden test.',
       '',f"- Paired background conditions: {src['background_pairs']}; unique development hulls: {src['distinct_vessel_hulls']}; clips: {src['distinct_vessel_clips']}.",
       f"- Original RMSE {m1['rmse']:.6f}; varied RMSE {m2['rmse']:.6f}; reduction {delta['rmse_reduction']:.6f}.",
       f"- Original ROC AUC {m1['auc']:.6f}; varied ROC AUC {m2['auc']:.6f}; gain {delta['auc_gain']:.6f}.",
       f"- At threshold .5, **no-addition controls** false-positive rate: {m1['fpr_no_addition_at_05']:.3f} vs {m2['fpr_no_addition_at_05']:.3f}.",
       f"- At threshold .5, **all observed-zero rows** false-positive rate: {m1['fpr_observed_zero_at_05']:.3f} vs {m2['fpr_observed_zero_at_05']:.3f}.",
       f"- Requested-positive but observed-zero rows: {src['masked_positive_with_zero_observed_share']} (possible missing-event masking); inspect separate definitions before interpreting specificity.",
       '', '## Limitations','',
       '- Pair-bootstrap intervals condition on a single internal split and reused hulls/clips; they are NOT independent out-of-domain uncertainty.',
       '- Source deletion is descriptive sensitivity, not a substitute for deployment-level or hull-crossed confidence intervals.',
       '- The original all-rows detection metric uses observed share > 0; a requested positive whose entire event is masked is labelled zero.',
       '- Threshold sweep is exploratory; choosing a threshold on these same examples and reporting its apparent performance would be optimistic.',
       '- No model should be promoted or submitted based on this audit alone.',
       '', '## Paired interval summary','',ci.to_markdown(index=False,floatfmt='.6f') if False else ci.to_string(index=False,float_format=lambda x:f'{x:.6f}'),
       '', '## Station-level review','',seg[seg.group.eq('station')].to_string(index=False,float_format=lambda x:f'{x:.5f}'),
       '']
 report={'experiment':'006B','type':'read_only_internal_dev_audit','seed':seed,'bootstrap':bootstrap,
         'input_sha256':{name:digest(root/name) for name in sorted(REQUIRED)},
         'sample':src,'point_estimates':delta,
         'checks':'passed: manifest hashes, exact paired rows, split allocation, group disjointness, pair composition, observed labels'}
 out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='exp006b_staging_',dir=out.parent) as tmp:
  stage=Path(tmp)
  for name,df in [('paired_differences.csv',merged[['pair','station_original','background_run_id_original','vessel_hull_id_original','requested_share','spectral_intervention','sq_gain','abs_gain']]),
                  ('overall_uncertainty.csv',ci),('paired_bootstrap_draws.csv',sampled),
                  ('segment_diagnostics.csv',seg),('threshold_sensitivity.csv',thresholds),
                  ('leave_one_source_out.csv',deletion)]:df.to_csv(stage/name,index=False)
  (stage/'RESEARCH_NOTE.md').write_text('\n'.join(note)+'\n')
  report['output_sha256']={p.name:digest(p) for p in stage.iterdir() if p.is_file()}
  (stage/'audit_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
  ensure(not out.exists(),'Output path unexpectedly exists')
  stage.rename(out)
 print('EXPERIMENT 006B READ-ONLY AUDIT: PASS')
 print('Internal dev pairs:',src['background_pairs'],'hulls:',src['distinct_vessel_hulls'],'clips:',src['distinct_vessel_clips'])
 print(ci.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
 print('Requested-positive, observed-zero:',src['masked_positive_with_zero_observed_share'])
 print('Output:',out)

def self_test():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'inputs';root.mkdir()
  rows=[];dev=[];train=[]
  for i in range(12):
   for q,gamma,name in [(0,0,'no_addition'),(.03,0,'original_vtuad'),(.03,.5,'halfway_shape'),(.03,1,'background_shaped_control'),(.15,0,'original_vtuad'),(.15,.5,'halfway_shape'),(.15,1,'background_shaped_control')]:
    r={'pair':i,'station':'A' if i<6 else 'B','background_run_id':f'dr{i}','vessel_clip_uid':f'dc{i%4}','vessel_hull_id':f'dh{i%3}',
       'requested_share':q,'observed_share':q,'spectral_intervention':name,'mask':'complete', 'duration_min':1,'gamma':gamma}
    dev.append(r)
    for model,noise in [('original_only',.06),('contrast_varied',.02)]:
     p=r.copy();p.update({'model':model,'estimated_share':min(1,q+noise),'detection_evidence':(.2 if q==0 else .9)+noise})
     rows.append(p)
  for i in range(12):train.append({'example':i,'station':'A','background_run_id':f'tr{i}','vessel_clip_uid':f'tc{i%4}','vessel_hull_id':f'th{i%3}'})
  files={'matched_development_predictions.csv':pd.DataFrame(rows), 'training_provenance.csv':pd.DataFrame(train),
         'development_provenance.csv':pd.DataFrame(dev),
         'internal_background_group_allocation.csv':pd.DataFrame([{'run_id':f'{part}r{i}','allocation':alloc}
          for part,alloc in [('t','internal_train'),('d','internal_dev')] for i in range(12)]),
         'internal_vessel_group_allocation.csv':pd.DataFrame([{'hull_id':f'{part}h{i}','allocation':alloc}
          for part,alloc in [('t','internal_train'),('d','internal_dev')] for i in range(3)]),
         'development_summary.csv':pd.DataFrame({'group':['overall']})}
  # Train and dev clip/hull IDs aligned with allocation.
  for key,frame in files.items():frame.to_csv(root/key,index=False)
  man={'dev_background_pairs':12,'train_samples_per_model':12,'output_sha256':{k:digest(root/k) for k in files}}
  (root/'experiment_manifest.json').write_text(json.dumps(man))
  analyze(root,Path(tmp)/'results',40,42)
  ensure((Path(tmp)/'results'/'RESEARCH_NOTE.md').is_file(),'Missing report')
  print('EXPERIMENT 006B CONSTRUCTED SELF-TEST: PASS')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',type=Path,default=DEFAULT_INPUT)
 ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
 ap.add_argument('--bootstrap',type=int,default=1000)
 ap.add_argument('--seed',type=int,default=20261122)
 ap.add_argument('--self-test-only',action='store_true')
 args=ap.parse_args()
 ensure(args.bootstrap>=20,'Bootstrap must be >=20')
 if args.self_test_only:self_test()
 else:analyze(args.input,args.output,args.bootstrap,args.seed)

if __name__=='__main__':main()
