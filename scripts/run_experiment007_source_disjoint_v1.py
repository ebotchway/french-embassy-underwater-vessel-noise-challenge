#!/usr/bin/env python3
"""Experiment 007: preallocated, source-disjoint synthetic train/calibration/assessment."""
from __future__ import annotations
import argparse, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from build_synthetic_mixtures_v1 import ParquetRowReader, BANDS, SHAPE
from build_experiment004_frozen_mixtures import EXPECTED_FROZEN_SHA256, sha
from run_experiment006_contrast_robustness_v1 import (
    build_training, build_internal_dev, feature_families, MASKS, GAMMAS,
)

OUT = Path('reports/baselines/experiment007_source_disjoint_v1')
FROZEN = Path('reports/splits/experiment004_frozen_v1')
SEED = 20261124
PARTS = ('train', 'calibration', 'assessment')
THRESHOLDS = np.round(np.arange(0.05, 1.001, 0.05), 3)
MAX_CAL_FPR = 0.25


def allocation(frame, key, stratum, seed):
    """Freeze 3-way group partition, keeping rare strata train-only."""
    classes = {}
    if frame[key].isna().any() or frame[stratum].isna().any():
        raise ValueError(f'Missing group/stratum identifier in {key}/{stratum}')
    for group, rows in frame.groupby(key, sort=True):
        labels = set(rows[stratum].astype(str))
        if len(labels) != 1: raise ValueError(f'{key} {group!r} crosses strata {labels}')
        classes[str(group)] = next(iter(labels))
    alloc = {}
    for label in sorted(set(classes.values())):
        groups = sorted((g for g, s in classes.items() if s == label),
                        key=lambda g: hashlib.sha256(f'{seed}|{label}|{g}'.encode()).hexdigest())
        if len(groups) < 3:
            alloc.update({g: 'train' for g in groups})
            continue
        # At least one group in each of calibration/assessment/train.
        n_cal = max(1, round(len(groups) * 0.15))
        n_assess = max(1, round(len(groups) * 0.15))
        if n_cal + n_assess >= len(groups):
            n_cal = n_assess = 1
        for i, g in enumerate(groups):
            alloc[g] = ('calibration' if i < n_cal else
                        'assessment' if i < n_cal + n_assess else 'train')
    return alloc


def audit_allocations(bg, vt, br, vh):
    if set(bg.run_id.astype(str)) != set(br): raise ValueError('Background run coverage mismatch')
    if set(vt.hull_id.astype(str)) != set(vh): raise ValueError('Vessel hull coverage mismatch')
    if set(br.values()) != set(PARTS): raise ValueError('Empty background partition')
    if set(vh.values()) != set(PARTS): raise ValueError('Empty hull partition')
    for part in PARTS:
        if not bg.run_id.astype(str).map(br).eq(part).any(): raise ValueError(f'No bg {part}')
        if not vt.hull_id.astype(str).map(vh).eq(part).any(): raise ValueError(f'No vessel {part}')
    # A clip can belong to only one hull and thus only one partition.
    if vt.groupby('clip_uid').hull_id.nunique().gt(1).any():
        raise ValueError('Clip linked to multiple hulls')
    for key, frame, groups in [('background',bg,'run_id'),('vessel',vt,'hull_id')]:
        if frame[groups].isna().any(): raise ValueError(f'Missing {key} group')


def adapt(assign, part):
    """Reuse Experiment 006 generator with all other partitions excluded."""
    label = 'internal_train' if part == 'train' else 'internal_dev'
    return {key: (label if p == part else 'excluded') for key,p in assign.items()}


def output_predictions(regs, dets, X, metadata):
    features = feature_families(X)['combined_96']
    rows=[]
    for policy, rm, dm in [('original_only','original_only','original_only'),
                           ('contrast_varied','contrast_varied','contrast_varied'),
                           ('hybrid','contrast_varied','original_only')]:
        f=metadata.copy()
        f['policy']=policy
        f['predicted_share']=np.clip(regs[rm].predict(features),0,1)
        clf=dets[dm]
        f['detection_score']=clf.predict_proba(features)[:,list(clf.classes_).index(1)]
        f['no_addition_control']=f.requested_share.eq(0)
        f['observed_positive']=f.observed_share.gt(1e-12)
        f['requested_positive_observed_zero']=f.requested_share.gt(0)&~f.observed_positive
        rows.append(f)
    out=pd.concat(rows,ignore_index=True)
    o=out.loc[out.policy.eq('original_only'),'detection_score'].to_numpy()
    h=out.loc[out.policy.eq('hybrid'),'detection_score'].to_numpy()
    v=out.loc[out.policy.eq('contrast_varied'),'predicted_share'].to_numpy()
    hv=out.loc[out.policy.eq('hybrid'),'predicted_share'].to_numpy()
    if not np.array_equal(o,h) or not np.array_equal(v,hv):
        raise AssertionError('Hybrid must exactly reuse original detector and varied regressor')
    return out


def metrics(d, threshold):
    y=d.observed_share.to_numpy(float)
    p=d.predicted_share.to_numpy(float)
    pos=d.observed_positive.to_numpy(bool)
    null=d.no_addition_control.to_numpy(bool)
    s=d.detection_score.to_numpy(float)
    result={
        'n':len(d),'pairs':d.pair.nunique(),
        'n_no_addition':int(null.sum()),'n_observed_positive':int(pos.sum()),
        'n_requested_positive_observed_zero':int(d.requested_positive_observed_zero.sum()),
        'rmse':float(np.sqrt(np.mean((y-p)**2))),
        'mae':float(np.mean(np.abs(y-p))),
        'bias':float(np.mean(p-y)),
        'auc':float(roc_auc_score(pos,s)) if 0 < pos.sum() < len(pos) else np.nan,
        'ap':float(average_precision_score(pos,s)) if 0 < pos.sum() < len(pos) else np.nan,
        'threshold':float(threshold),
        'tpr_observed_positive':float(np.mean(s[pos]>=threshold)) if pos.any() else np.nan,
        'fpr_no_addition':float(np.mean(s[null]>=threshold)) if null.any() else np.nan,
        'fpr_observed_zero':float(np.mean(s[~pos]>=threshold)) if (~pos).any() else np.nan,
        'brier_observed':float(brier_score_loss(pos,s)) if 0 < pos.sum() < len(pos) else np.nan,
    }
    return result


def choose_threshold(cal):
    """Predeclared high-sensitivity policy with FPR cap on genuine zeros."""
    null=cal.no_addition_control.to_numpy(bool)
    pos=cal.observed_positive.to_numpy(bool)
    scores=cal.detection_score.to_numpy(float)
    if null.sum()<10 or pos.sum()<10: raise ValueError('Too few calibration controls/positives')
    feasible=[]; table=[]
    for t in THRESHOLDS:
        fpr=float(np.mean(scores[null]>=t));tpr=float(np.mean(scores[pos]>=t))
        table.append({'threshold':float(t),'calibration_tpr':tpr,'calibration_fpr_no_addition':fpr,
                      'meets_cap':bool(fpr<=MAX_CAL_FPR)})
        if fpr<=MAX_CAL_FPR: feasible.append((tpr, float(t)))
    if not feasible:
        return None,pd.DataFrame(table)
    # Maximize sensitivity, ties prefer higher threshold for specificity.
    best=max(feasible,key=lambda item:(item[0],item[1]))[1]
    return best,pd.DataFrame(table)


def fit_platt(cal, ass):
    """Diagnostic only: logistic map fitted on calibration raw classifier scores."""
    y=cal.observed_positive.to_numpy(int)
    if len(np.unique(y))!=2: raise ValueError('Calibration labels single-class')
    def logit(s):
        z=np.clip(np.asarray(s,float),1e-5,1-1e-5)
        return np.log(z/(1-z)).reshape(-1,1)
    m=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=0)
    m.fit(logit(cal.detection_score),y)
    pred=m.predict_proba(logit(ass.detection_score))[:,1]
    labels=ass.observed_positive.to_numpy(int)
    raw=np.asarray(ass.detection_score,float)
    return {'raw_assessment_brier':float(brier_score_loss(labels,raw)),
            'platt_assessment_brier':float(brier_score_loss(labels,pred)),
            'raw_assessment_log_loss':float(log_loss(labels,np.clip(raw,1e-8,1-1e-8),labels=[0,1])),
            'platt_assessment_log_loss':float(log_loss(labels,np.clip(pred,1e-8,1-1e-8),labels=[0,1])),
            'platt_slope':float(m.coef_[0,0]),'platt_intercept':float(m.intercept_[0])}


def evaluate(cal_predictions,ass_predictions):
    thresholds=[];policies=[];platt=[];segments=[]
    for name in ('original_only','contrast_varied','hybrid'):
        cal=cal_predictions.loc[cal_predictions.policy.eq(name)]
        ass=ass_predictions.loc[ass_predictions.policy.eq(name)]
        t,tab=choose_threshold(cal)
        tab.insert(0,'policy',name)
        thresholds.append(tab)
        selected = [('.5_fixed',.5)] if t is None else [('.5_fixed',.5),('calibration_selected',t)]
        for label,threshold in selected:
            m=metrics(ass,threshold)
            m.update({'policy':name,'threshold_policy':label,
                      'calibration_feasible':t is not None})
            policies.append(m)
            if label=='calibration_selected':
                for field in ('station','mask','spectral_intervention'):
                    for value,block in ass.groupby(field,sort=True):
                        b=metrics(block,threshold)
                        b.update({'policy':name,'group_type':field,'group_value':str(value),
                                  'threshold_policy':label})
                        segments.append(b)
        # Assessment labels only scored; never passed to calibration fitter.
        p=fit_platt(cal,ass)
        p['policy']=name
        platt.append(p)
    return (pd.concat(thresholds,ignore_index=True),pd.DataFrame(policies),
            pd.DataFrame(platt),pd.DataFrame(segments))


def source_overlap(train_meta,cal_meta,ass_meta):
    audits=[]
    for field in ('background_run_id','vessel_hull_id','vessel_clip_uid'):
        values=[]
        for df in (train_meta,cal_meta,ass_meta):
            vals=set(df[field].dropna().astype(str))-{'','nan','None'}
            values.append(vals)
        for i in range(3):
            for j in range(i+1,3):
                overlap=values[i]&values[j]
                audits.append({'field':field,'partition_a':PARTS[i],'partition_b':PARTS[j],
                               'overlap_count':len(overlap)})
                if overlap: raise ValueError(f'SOURCE LEAKAGE: {field} {PARTS[i]} vs {PARTS[j]}')
    return pd.DataFrame(audits)



def assessment_sensitivity(predictions, thresholds, draws=600, seed=24680):
    """Paired assessment uncertainty, with recording-run clustering."""
    by_policy={p:d.copy() for p,d in predictions.groupby('policy',sort=True)}
    base=by_policy['original_only']; varied=by_policy['contrast_varied']; hybrid=by_policy['hybrid']
    ids={p:set(df.pair.astype(int)) for p,df in by_policy.items()}
    if len(set(map(frozenset,ids.values())))!=1:raise ValueError('Assessment pairs not matched')
    pairs=base[['pair','background_run_id','vessel_hull_id','vessel_clip_uid','station']].drop_duplicates()
    if len(pairs)!=len(ids['original_only']):raise ValueError('Inconsistent pair provenance')
    assigned={r: list(g.pair.astype(int)) for r,g in pairs.groupby('background_run_id',sort=True)}
    def score_for(policy, chosen_ids, threshold):
        frame=by_policy[policy]
        # repeated bootstrap clusters require weighting via row replication
        index=frame.set_index('pair')
        picked=index.loc[chosen_ids].reset_index()
        return metrics(picked,threshold)
    rng=np.random.default_rng(seed)
    runs=sorted(assigned)
    out=[]
    for method in ('original_vs_varied','original_vs_hybrid'):
        right='contrast_varied' if method=='original_vs_varied' else 'hybrid'
        for threshold_type in ('fixed_0_5','calibration_selected'):
            if threshold_type=='fixed_0_5': a=b=.5
            else:
                a=thresholds.get('original_only');b=thresholds.get(right)
                if a is None or b is None:continue
            full_a=metrics(base,a);full_b=metrics(by_policy[right],b)
            point={'rmse_reduction':full_a['rmse']-full_b['rmse'],
                   'auc_gain':full_b['auc']-full_a['auc'],
                   'fpr_no_addition_increase':full_b['fpr_no_addition']-full_a['fpr_no_addition'],
                   'tpr_gain':full_b['tpr_observed_positive']-full_a['tpr_observed_positive']}
            draws_by_metric={k:[] for k in point}
            for iteration in range(draws):
                sampled_runs=rng.choice(runs,size=len(runs),replace=True)
                selected=[pair for run in sampled_runs for pair in assigned[run]]
                x=score_for('original_only',selected,a)
                y=score_for(right,selected,b)
                values={'rmse_reduction':x['rmse']-y['rmse'],
                        'auc_gain':y['auc']-x['auc'],
                        'fpr_no_addition_increase':y['fpr_no_addition']-x['fpr_no_addition'],
                        'tpr_gain':y['tpr_observed_positive']-x['tpr_observed_positive']}
                for k,val in values.items():
                    if np.isfinite(val):draws_by_metric[k].append(float(val))
            for k,value in point.items():
                values=draws_by_metric[k]
                out.append({'comparison':method,'threshold_type':threshold_type,
                            'metric':k,'point_estimate':value,'bootstrap_valid_draws':len(values),
                            'run_bootstrap_95_low':float(np.quantile(values,.025)) if values else np.nan,
                            'run_bootstrap_95_high':float(np.quantile(values,.975)) if values else np.nan})
    del_rows=[]
    for key in ('station','background_run_id','vessel_hull_id','vessel_clip_uid'):
        for group,part in pairs.groupby(key,sort=True):
            remaining=sorted(ids['original_only']-set(part.pair.astype(int)))
            if len(remaining)<2:continue
            original=score_for('original_only',remaining,.5)
            varied=score_for('contrast_varied',remaining,.5)
            del_rows.append({'deleted_type':key,'deleted_group':str(group),
                             'pairs_remaining':len(remaining),
                             'rmse_reduction':original['rmse']-varied['rmse'],
                             'auc_gain':varied['auc']-original['auc'],
                             'fpr_no_addition_increase':varied['fpr_no_addition']-original['fpr_no_addition']})
    return pd.DataFrame(out),pd.DataFrame(del_rows)


def study(bg,vt,reader,args,hashes):
    # Independent grouping is done BEFORE generating a single background/source pairing.
    br=allocation(bg,'run_id','station',args.seed)
    vh=allocation(vt,'hull_id','label',args.seed+11)
    audit_allocations(bg,vt,br,vh)
    alloc_bg=pd.DataFrame([{'run_id':g,'station':s,'partition':br[g]}
                           for g,s in sorted(bg.groupby('run_id').station.first().items())])
    alloc_vt=pd.DataFrame([{'hull_id':g,'vessel_class':s,'partition':vh[g]}
                           for g,s in sorted(vt.groupby('hull_id').label.first().items())])
    print('Background runs by partition:\n',alloc_bg.groupby(['station','partition']).size().to_string(),flush=True)
    print('Vessel hulls by partition:\n',alloc_vt.groupby('partition').size().to_string(),flush=True)
    train,tr_meta,train_counts=build_training(bg,vt,adapt(br,'train'),adapt(vh,'train'),
                                              args.train,args.seed+101,reader)
    cal_X,cal_y,cal_meta,cal_counts=build_internal_dev(bg,vt,adapt(br,'calibration'),
                                                       adapt(vh,'calibration'),args.cal_pairs,args.seed+303,reader)
    ass_X,ass_y,ass_meta,ass_counts=build_internal_dev(bg,vt,adapt(br,'assessment'),
                                                       adapt(vh,'assessment'),args.assess_pairs,args.seed+505,reader)
    for df, name in [(cal_meta,'calibration'),(ass_meta,'assessment')]:
        for col in ('station','mask','background_run_id','vessel_clip_uid','vessel_hull_id'):
            if df[col].isna().any(): raise ValueError(f'Missing {col} in {name}')
        if len(df)!=7*(args.cal_pairs if name=='calibration' else args.assess_pairs):
            raise ValueError('Unexpected matched cases')
    source_audit=source_overlap(tr_meta,cal_meta,ass_meta)
    regs={}; dets={}
    for name in ('original_only','contrast_varied'):
        X,y=train[name]
        F=feature_families(X)['combined_96']
        cls=(y>1e-12).astype(int)
        if np.unique(cls).size!=2: raise ValueError('Training labels lack zeros or positives')
        reg=RandomForestRegressor(n_estimators=args.trees,min_samples_leaf=2,
                                   random_state=args.seed+404,n_jobs=1)
        det=RandomForestClassifier(n_estimators=args.trees,min_samples_leaf=2,
                                    random_state=args.seed+404,n_jobs=1)
        reg.fit(F,y);det.fit(F,cls)
        regs[name]=reg;dets[name]=det
        print(f'Trained {name}; observed-positive: {cls.sum()}/{len(cls)}',flush=True)
    cal=output_predictions(regs,dets,cal_X,cal_meta)
    ass=output_predictions(regs,dets,ass_X,ass_meta)
    threshold,policy,platt,segments=evaluate(cal,ass)
    chosen={}
    for name,frame in threshold.groupby('policy'):
        viable=frame.loc[frame.meets_cap]
        if len(viable):
            # Use the same predeclared rule as evaluate/choose_threshold.
            chosen[name]=max(zip(viable.calibration_tpr,viable.threshold),key=lambda z:(z[0],z[1]))[1]
        else:chosen[name]=None
    uncertainty,deletion=assessment_sensitivity(ass,chosen,draws=args.bootstrap,seed=args.seed+909)
    if len(policy)<3 or not policy.policy.eq('hybrid').any():raise ValueError('Incomplete policies')
    # File output is fully staged; never replace an existing research result.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(f'Refusing overwrite: {args.output}')
    manifest={
        'experiment':'007 source-disjoint internal train/calibration/assessment',
        'source_partition':'exp004_train ONLY',
        'historical_warning':'The underlying source pool was previously inspected in Experiments 004-006; this is newly independent WITHIN 007 but NOT a pristine external confirmation.',
        'seed':args.seed,'bootstrap_resamples':args.bootstrap,'n_train_per_model':args.train,'calibration_pairs':args.cal_pairs,
        'assessment_pairs':args.assess_pairs,'rf_estimators':args.trees,
        'allocation':'Before pairing: SHA256-deterministic per-station run and per-class hull partitions; 15% calibration, 15% assessment, rest train; classes/stations with <3 groups train-only.',
        'calibration_rule':'On calibration only, threshold grid 0.05...1.00; highest observed-positive TPR while genuine no-addition FPR <= 0.25; tie: larger threshold. If no feasible candidate, no calibrated policy.',
        'label_definition':'observed_share > 1e-12, including masked positive-requested cases as observed-zero',
        'negative_definitions':['genuine requested_share=0 controls','all observed_share=0, including fully masked additions'],
        'fraction_models':{'original_only':'original_only','contrast_varied':'contrast_varied','hybrid':'contrast_varied'},
        'detector_models':{'original_only':'original_only','contrast_varied':'contrast_varied','hybrid':'original_only'},
        'random_forest':{'n_estimators':args.trees,'min_samples_leaf':2,'features':'combined_96'},
        'gamma_original':[0.0],'gamma_varied':[0.0,0.5,1.0],
        'mask_conditions':{k:list(v) for k,v in MASKS.items()},
        'source_sha256':hashes,'training_station_count':train_counts,
        'calibration_station_pairs':cal_counts,'assessment_station_pairs':ass_counts,
        'limitations':['Synthetic injected-energy ground truth, not real vessel attribution',
            'No physical calibration across WAV and South African recording levels',
            'Assessment shares underlying historical research sources; not new outside-domain holdout',
            'Station balanced sampling but station-specific single-run strata may be train-only',
            'Run-clustered bootstrap does not account for shared hulls or deployments and is conditional on this split',
            'AP depends on simulated high positive prevalence; not a natural event prevalence',
            'Thresholds evaluated once on assessment; do NOT adapt or retune to these outputs'],
    }
    note=("# Experiment 007 — Source-disjoint calibration and assessment\n\n"
          "This is a newly allocated, prespecified internal study using **only exp004_train**. "
          "Its assessment is independent of Experiment 007 training/calibration by known "
          "recording runs, vessel hull IDs and clip IDs, but the underlying research sources "
          "were previously inspected in historical experiments. It is not external validation.\n\n"
          "The calibration threshold maximizes observed-positive sensitivity subject to "
          "genuine background-only FPR <= 25% on the calibration partition only. "
          "Assessment FPR may exceed that cap. Platt mappings are diagnostic, not proof "
          "that scores represent real vessel probabilities.\n\n"
          "## Assessment metrics\n\n"+policy.to_string(index=False)+"\n\n"
          "## Caveats\n\n"+'\n'.join('- '+v for v in manifest['limitations'])+'\n')
    with tempfile.TemporaryDirectory(prefix='experiment007_stage_',dir=args.output.parent) as temp:
        stage=Path(temp)
        outputs={'background_group_allocation.csv':alloc_bg,'vessel_group_allocation.csv':alloc_vt,
                 'source_overlap_audit.csv':source_audit,'train_provenance.csv':tr_meta,
                 'calibration_provenance.csv':cal_meta,'assessment_provenance.csv':ass_meta,
                 'calibration_predictions.csv':cal,'assessment_predictions.csv':ass,
                 'threshold_selection_grid.csv':threshold,'assessment_policy_metrics.csv':policy,
                 'platt_calibration_diagnostic.csv':platt,'assessment_segments.csv':segments,
                 'assessment_run_bootstrap.csv':uncertainty,
                 'assessment_source_deletion.csv':deletion}
        for fn,frame in outputs.items():frame.to_csv(stage/fn,index=False)
        (stage/'RESEARCH_NOTE.md').write_text(note)
        manifest['output_sha256']={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
        (stage/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
        if args.output.exists():raise FileExistsError(f'Output appeared during computation: {args.output}')
        stage.rename(args.output)
    print('\nEXPERIMENT 007: PASS — internal source-disjoint study',flush=True)
    print(policy[['policy','threshold_policy','threshold','rmse','auc','tpr_observed_positive','fpr_no_addition']].to_string(index=False),flush=True)
    print('Outputs:',args.output,flush=True)
    print('NO frozen Experiment004 eval, prior validation, sealed holdout or Zindi Test.csv used.',flush=True)


def selftest():
    from types import SimpleNamespace
    # Complete in-memory runs and clip sources; runs/hulls disjoint by design.
    bg=[]; vt=[]
    for si,station in enumerate(('A','B','C','D')):
        for run in range(8):
            for win in range(4):
                key=si*10000+run*100+win
                bg.append({'run_id':f'{station}-run-{run}','station':station,
                           'source_parquet':'fixture', 'row_positions_json':json.dumps(list(range(key,key+15))),
                           'experiment004_partition':'exp004_train'})
    for cls in ('cargo','tug','fishing'):
        for hi in range(6):
            for minute in range(3):
                row={'hull_id':f'{cls}-{hi}','label':cls,'clip_uid':f'{cls}-{hi}-clip',
                     'minute_index':minute,'experiment004_partition':'exp004_train'}
                row.update({band:float(0.05+(j+1)*0.003*(1+hi*.05)) for j,band in enumerate(BANDS)})
                vt.append(row)
    class FixtureReader:
        def read(self,path,positions):
            t=np.asarray(positions,dtype=float)
            return np.asarray([55+np.sin(t*.17+j*.31)*2+j*.12 for j in range(25)]).T
    a=SimpleNamespace(output=None,seed=SEED,train=60,cal_pairs=12,assess_pairs=12,trees=12,bootstrap=20)
    with tempfile.TemporaryDirectory(prefix='exp007_selftest_') as d:
        a.output=Path(d)/'study'
        study(pd.DataFrame(bg),pd.DataFrame(vt),FixtureReader(),a,{'fixture':'in-memory'})
        from verify_experiment007_artifacts_v1 import validate
        validate(a.output)
        m=pd.read_csv(a.output/'assessment_policy_metrics.csv')
        overlaps=pd.read_csv(a.output/'source_overlap_audit.csv')
        if len(m)!=6 or not (overlaps.overlap_count==0).all():
            raise AssertionError('Incorrect assessment rows or source overlap')
        if set(m.policy)!=set(('original_only','contrast_varied','hybrid')):
            raise AssertionError('Missing architecture')
    print('EXPERIMENT 007 CONSTRUCTED END-TO-END SELF-TEST: PASS',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',type=Path,default=FROZEN)
    ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--train',type=int,default=1200)
    ap.add_argument('--cal-pairs',type=int,default=120)
    ap.add_argument('--assess-pairs',type=int,default=120)
    ap.add_argument('--trees',type=int,default=200)
    ap.add_argument('--bootstrap',type=int,default=600)
    ap.add_argument('--self-test-only',action='store_true')
    a=ap.parse_args()
    if a.self_test_only:
        selftest();return
    if min(a.train,a.cal_pairs,a.assess_pairs,a.trees,a.bootstrap)<1:
        ap.error('All counts must be positive')
    if a.output.exists():raise FileExistsError(f'Output already exists: {a.output}')
    hashes={}
    for name,expected in EXPECTED_FROZEN_SHA256.items():
        path=a.frozen/name
        if not path.exists() or sha(path)!=expected:
            raise ValueError(f'Frozen Experiment004 manifest missing/changed: {path}')
        hashes[name]=expected
    bg=pd.read_csv(a.frozen/'background_manifest_exp004.csv',low_memory=False)
    vt=pd.read_csv(a.frozen/'vessel_minute_manifest_exp004.csv',
                   dtype={'hull_id':'string','clip_uid':'string'},low_memory=False)
    bg=bg.loc[bg.experiment004_partition.eq('exp004_train')].copy()
    vt=vt.loc[vt.experiment004_partition.eq('exp004_train')].copy()
    if bg.empty or vt.empty:raise ValueError('Empty allowed training source pool')
    study(bg,vt,ParquetRowReader(),a,hashes)

if __name__=='__main__':main()
