#!/usr/bin/env python3
"""Experiment 006C: source-group-disjoint threshold calibration and assessment."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import audit_experiment006b_readonly_v1 as prior

INPUT = Path('reports/baselines/experiment006_internal_contrast_v1')
OUTPUT = Path('reports/audits/experiment006c_calibration_v1')
SEED = 20261122


def ensure(ok, why):
    if not ok:
        raise ValueError(why)


def component_partition(pairs):
    """Connected components of shared backgrounds, hulls and clips."""
    pairs = pairs.sort_values('pair').reset_index(drop=True)
    parent = list(range(len(pairs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    for field in ('background_run_id','vessel_hull_id','vessel_clip_uid'):
        seen={}
        for i, val in enumerate(pairs[field].tolist()):
            if pd.isna(val) or str(val).strip()=='' or str(val).lower()=='nan':
                continue
            key=str(val)
            if key in seen:
                union(i,seen[key])
            else:
                seen[key]=i
    groups={}
    for i in range(len(pairs)):
        groups.setdefault(find(i), []).append(i)
    parts=sorted(groups.values(),key=lambda x:(-len(x),min(x)))
    assignments={}
    for cid,indices in enumerate(parts):
        for i in indices:
            assignments[pairs.at[i,'pair']] = cid
    pairs['source_component']=pairs['pair'].map(assignments)
    return pairs, [len(x) for x in parts]


def allocate(pairs, seed, fraction=.3, minimum=12):
    """Search deterministic group-disjoint subset with minimum examples."""
    size=pairs.groupby('source_component').size().to_dict()
    n=len(pairs);target=round(n*fraction)
    if len(size)<2:return None
    rng=np.random.default_rng(seed)
    best=None
    components=np.array(sorted(size))
    for _ in range(12000):
        cand=rng.permutation(components)
        selected=set();k=0
        for c in cand:
            s=size[int(c)]
            if k+s<=target or (k<minimum and k+s<n-minimum):
                selected.add(int(c));k+=s
            if k>=target:break
        if minimum<=k<=n-minimum:
            cal=pairs[~pairs.source_component.isin(selected)]
            assess=pairs[pairs.source_component.isin(selected)]
            # Coverage is a tie-break only, not a label-dependent selection.
            stations=assess.station.nunique()
            objective=(abs(k-target),-stations)
            if best is None or objective<best[0]: best=(objective,selected)
    if best is None:return None
    selected=best[1]
    out=pairs.copy()
    out['partition']=np.where(out.source_component.isin(selected),'assessment','calibration')
    return out


def label_and_scores(frame):
    labels=(frame['observed_share'].to_numpy(float)>1e-12).astype(int)
    score=frame['detection_evidence'].to_numpy(float)
    control=frame['spectral_intervention'].astype(str).eq('no_addition').to_numpy()
    return labels,score,control


def measure(frame, threshold, score_col='detection_evidence', share_col='estimated_share'):
    y=(frame.observed_share.to_numpy(float)>1e-12).astype(int)
    s=frame[score_col].to_numpy(float);z=frame.spectral_intervention.eq('no_addition').to_numpy()
    observed_zero=y==0
    q=frame.observed_share.to_numpy(float);r=frame[share_col].to_numpy(float)
    return {
       'n':len(frame),'pairs':frame.pair.nunique(),'n_control':int(z.sum()),
       'n_masked_positive_zero':int(((frame.requested_share>0)&(frame.observed_share<=1e-12)).sum()),
       'threshold':float(threshold),
       'tpr_observed_positive':float(np.mean(s[y==1]>=threshold)) if (y==1).any() else np.nan,
       'fpr_no_addition':float(np.mean(s[z]>=threshold)) if z.any() else np.nan,
       'fpr_observed_zero':float(np.mean(s[observed_zero]>=threshold)) if observed_zero.any() else np.nan,
       'auc':float(roc_auc_score(y,s)) if len(set(y))==2 else np.nan,
       'average_precision':float(average_precision_score(y,s)) if len(set(y))==2 else np.nan,
       'brier':float(brier_score_loss(y,s)),
       'rmse':float(np.sqrt(np.mean((r-q)**2))),
       'mae':float(np.mean(abs(r-q)))}


def select_threshold(frame,cap=.25):
    """Predefined rule: under calibration FPR cap, maximize sensitivity."""
    y,s,z=label_and_scores(frame)
    ensure(z.any() and (y==1).any(),'Calibration lacks positive or genuine controls')
    grid=np.unique(np.r_[np.arange(.05,1.001,.05),1.0])
    rows=[]
    for t in grid:
        fpr=np.mean(s[z]>=t)
        tpr=np.mean(s[y==1]>=t)
        rows.append({'threshold':round(float(t),3),'cal_fpr_no_addition':fpr,'cal_tpr':tpr,
                     'feasible':bool(fpr<=cap+1e-12)})
    good=[r for r in rows if r['feasible']]
    ensure(good,'No feasible threshold')
    chosen=sorted(good,key=lambda r:(-r['cal_tpr'],r['cal_fpr_no_addition'],-r['threshold']))[0]
    return chosen,pd.DataFrame(rows)


def platt_fit_and_measure(cal,assess):
    ycal,scal,_=label_and_scores(cal)
    ensure(len(set(ycal))==2,'Cannot fit diagnostic calibration with one class')
    def logit(scores):
        scores=np.clip(scores,1e-5,1-1e-5)
        return np.log(scores/(1-scores)).reshape(-1,1)
    model=LogisticRegression(C=1.,max_iter=1000,random_state=0)
    model.fit(logit(scal),ycal)
    adjusted=assess.copy()
    adjusted['platt_score']=model.predict_proba(logit(assess.detection_evidence.to_numpy(float)))[:,1]
    y=label_and_scores(assess)[0]
    return {'raw_brier':float(brier_score_loss(y,assess.detection_evidence)),
            'platt_brier':float(brier_score_loss(y,adjusted.platt_score)),
            'raw_auc':float(roc_auc_score(y,assess.detection_evidence)) if len(set(y))==2 else np.nan,
            'platt_auc':float(roc_auc_score(y,adjusted.platt_score)) if len(set(y))==2 else np.nan,
            'slope':float(model.coef_[0,0]),'intercept':float(model.intercept_[0])}


def compute(pred, seed=SEED, fpr_cap=.25, min_pairs=12):
    a=pred[pred.model.eq('original_only')].copy()
    b=pred[pred.model.eq('contrast_varied')].copy()
    ensure(len(a)==len(b) and len(a)>0,'Missing or unequal predictions')
    pair_cols=['pair','station','background_run_id','vessel_hull_id','vessel_clip_uid']
    pair_meta=a[pair_cols].drop_duplicates('pair')
    ensure(len(pair_meta)==a.pair.nunique(),'Invalid pair metadata')
    components,sizes=component_partition(pair_meta)
    allocation=allocate(components,seed,minimum=min_pairs)
    report={'total_pairs':len(components),'source_components':len(sizes),
            'component_sizes_desc':sizes, 'largest_component_pairs':max(sizes),
            'feasible':allocation is not None,
            'rule':f'Choose threshold on calibration with no-addition FPR <= {fpr_cap}; maximize observed-positive TPR',
            'warning':'Existing model was previously inspected on all dev pairs; assessment is not pristine model-selection holdout.'}
    if allocation is None:
        report['reason']='No feasible source-run/hull/clip-disjoint calibration/assessment partition with minimum pair counts.'
        return report, components, {}, None
    ensure(not allocation.pair.duplicated().any(),'Duplicate allocation pairs')
    # Verify exact exclusion constraints, and avoid shared run/hull/clip across partitions.
    c=allocation[allocation.partition.eq('calibration')]
    e=allocation[allocation.partition.eq('assessment')]
    for field in ('background_run_id','vessel_hull_id','vessel_clip_uid'):
        ensure(set(c[field].dropna().astype(str)).isdisjoint(set(e[field].dropna().astype(str))),
               f'{field} is not disjoint')
    report.update({'calibration_pairs':len(c),'assessment_pairs':len(e),
                   'calibration_stations':sorted(c.station.unique().tolist()),
                   'assessment_stations':sorted(e.station.unique().tolist())})
    pred=pred.merge(allocation[['pair','partition','source_component']],on='pair',validate='many_to_one')
    results=[];sweeps=[];platt=[]
    for model in ('original_only','contrast_varied'):
        fr=pred[pred.model.eq(model)].copy()
        train=fr[fr.partition.eq('calibration')]
        test=fr[fr.partition.eq('assessment')]
        selection,sweep=select_threshold(train,cap=fpr_cap)
        sweep.insert(0,'model',model);sweeps.append(sweep)
        for partition,part in [('calibration',train),('assessment',test)]:
            for mode,t in [('fixed_05',.5),('selected',selection['threshold'])]:
                results.append({'model':model,'partition':partition,'policy':mode,**measure(part,t)})
        p=platt_fit_and_measure(train,test);p['model']=model;platt.append(p)
        report[model+'_chosen_threshold']=selection
    # Explicit hybrid: varied share regressor with ORIGINAL detector.
    orig=pred[pred.model.eq('original_only')];var=pred[pred.model.eq('contrast_varied')]
    key=['pair','requested_share','spectral_intervention']
    hybrid=var.merge(orig[key+['detection_evidence']],on=key,validate='one_to_one',suffixes=('','_original'))
    hybrid['detection_evidence']=hybrid['detection_evidence_original']
    hybrid=hybrid.drop(columns=['detection_evidence_original'])
    for part in ('calibration','assessment'):
        sub=hybrid[hybrid.partition.eq(part)]
        for mode,t in [('fixed_05',.5),('original_selected',report['original_only_chosen_threshold']['threshold'])]:
            results.append({'model':'hybrid_varied_share_original_detector','partition':part,'policy':mode,**measure(sub,t)})
    # Segment summaries use only the assessment allocation and the two
    # preselected thresholds; small segment denominators remain visible.
    segments=[]
    for model in ('original_only','contrast_varied'):
        subset=pred[(pred.model==model)&(pred.partition=='assessment')]
        th=report[model+'_chosen_threshold']['threshold']
        for field in ('station','mask','spectral_intervention'):
            for value,part in subset.groupby(field,dropna=False):
                segments.append({'model':model,'field':field,'value':str(value),
                                 'policy':'selected','note':'Descriptive small-group assessment',
                                 **measure(part,th)})
    # Explicit zero-control and observed-positive counts protect metric interpretation.
    return report,allocation,{'policy_metrics.csv':pd.DataFrame(results),
                               'threshold_calibration_sweep.csv':pd.concat(sweeps,ignore_index=True),
                               'platt_calibration_diagnostic.csv':pd.DataFrame(platt),
                               'assessment_segments.csv':pd.DataFrame(segments)},pred


def execute(root,out,seed,cap,min_pairs):
    ensure(not out.exists(),f'Output exists, refusing overwrite: {out}')
    tables,old_manifest=prior.read_and_verify(root)
    prior.validate(tables,old_manifest)
    pred=tables['matched_development_predictions.csv']
    report,allocation,products,augmented=compute(pred,seed,cap,min_pairs)
    report.update({'experiment':'006C','input_root':str(root),'seed':seed,'fpr_cap':cap,
                   'source_manifest_sha256':prior.digest(root/'experiment_manifest.json'),
                   'prediction_sha256':prior.digest(root/'matched_development_predictions.csv'),
                   'read_only':True,'frozen_eval_used':False})
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='exp006c_stage_',dir=out.parent) as temp:
        stage=Path(temp)
        allocation.to_csv(stage/'source_component_allocation.csv',index=False)
        for name,table in products.items():table.to_csv(stage/name,index=False)
        if augmented is not None:
            diagnostics=augmented.loc[augmented.spectral_intervention.eq('no_addition'),
                ['pair','partition','station','background_run_id','vessel_hull_id',
                 'model','detection_evidence','estimated_share']]
            diagnostics.to_csv(stage/'no_addition_scores.csv',index=False)
        (stage/'audit_manifest.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        lines=['# Experiment 006C — source-disjoint calibration feasibility and assessment','',
               f"- Source groups: {report['source_components']} connected components across {report['total_pairs']} pairs.",
               f"- Largest component: {report['largest_component_pairs']} pairs.",
               '- No frozen evaluation, original validation, sealed holdouts, test labels or model retraining used.',
               '- The previously inspected Experiment 006 development predictions are reused; this is not a pristine independent final test.',
               '']
        if report['feasible']:
            lines += [f"- Calibration pairs: {report['calibration_pairs']}; assessment pairs: {report['assessment_pairs']}.",
                      f"- Calibration FPR cap on genuine no-addition controls: {cap:.3f}.",
                      '- Threshold decisions use calibration rows only; assessment reports the resulting metrics once.',
                      '- Platt calibration is a diagnostic mapping on calibration rows; not a recommended competition calibration.',
                      '', '## Assessment metrics', '',
                      products['policy_metrics.csv'].query("partition=='assessment'").to_string(index=False,float_format=lambda x:f'{x:.5f}'),'',
                      '## Limitations','',
                      '- Internal synthetic comparisons, not real-vessel detection or hidden-test accuracy.',
                      '- Paired examples and source components limit effective sample size.',
                      '- Synthetic label prevalence is artificial, so Brier and AP are prevalence-dependent.',
                      '- One partition and post hoc experiment history cannot validate a newly selected leaderboard model.',
                      '- Do not choose a threshold after inspecting assessment results or submit based on these outputs alone.']
        else:
            lines += ['**Strict calibration/assessment split is infeasible with the current reused source graph.**',
                      report['reason'],'',
                      'No calibration threshold or assessment claim was generated. A genuinely new source-separated development pool is required.']
        (stage/'RESEARCH_NOTE.md').write_text('\n'.join(lines)+'\n')
        ensure(not out.exists(),'Output created by another process')
        stage.rename(out)
    print('EXPERIMENT 006C:', 'PASS' if report['feasible'] else 'FEASIBILITY BLOCK')
    print('Components:',report['source_components'],'largest:',report['largest_component_pairs'],'of',report['total_pairs'])
    if report['feasible']:
        print(products['policy_metrics.csv'].query("partition=='assessment'").to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Output:',out)


def self_test():
    rows=[]
    # Make 6 completely disconnected source components of 6 pairs each.
    for comp in range(6):
        for i in range(6):
            pair=comp*6+i
            for q,name in [(0,'no_addition'),(.03,'original_vtuad'),(.03,'halfway_shape'),(.03,'background_shaped_control'),
                           (.15,'original_vtuad'),(.15,'halfway_shape'),(.15,'background_shaped_control')]:
                for model in ('original_only','contrast_varied'):
                    rows.append({'pair':pair,'station':f'S{comp}', 'background_run_id':f'r{comp}',
                     'vessel_hull_id':f'h{comp}','vessel_clip_uid':f'c{comp}',
                     'requested_share':q,'observed_share':q,'spectral_intervention':name,'mask':'complete',
                     'model':model,'detection_evidence':(0.36 if q==0 else 0.84)+(0.04 if model=='contrast_varied' else 0),
                     'estimated_share':max(0,min(1,q+(.04 if model=='original_only' else .02)))})
    frame=pd.DataFrame(rows)
    report,alloc,products,_=compute(frame,seed=42,min_pairs=6)
    ensure(report['feasible'],'Fixture should be split-feasible')
    ensure(set(alloc[alloc.partition=='calibration'].source_component).isdisjoint(
           set(alloc[alloc.partition=='assessment'].source_component)),'Component overlap')
    ensure(len(products['policy_metrics.csv'])==12,'Missing policy rows')
    # Single connected component must REFUSE fake source independence.
    one=frame.copy();one['vessel_hull_id']='identical_hull'
    report2,_,prods,_=compute(one,seed=42,min_pairs=6)
    ensure(not report2['feasible'] and not prods,'Leakage fixture did not block')
    print('EXPERIMENT 006C SELF-TEST: PASS (group-disjoint and infeasibility safeguards)')


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--input',type=Path,default=INPUT)
    pa.add_argument('--output',type=Path,default=OUTPUT)
    pa.add_argument('--seed',type=int,default=SEED)
    pa.add_argument('--fpr-cap',type=float,default=.25)
    pa.add_argument('--min-pairs',type=int,default=12)
    pa.add_argument('--self-test-only',action='store_true')
    a=pa.parse_args()
    ensure(0<a.fpr_cap<1,'Invalid FPR cap')
    ensure(a.min_pairs>=6,'Minimum pairs must be >=6')
    if a.self_test_only:self_test()
    else:execute(a.input,a.output,a.seed,a.fpr_cap,a.min_pairs)

if __name__=='__main__':main()
