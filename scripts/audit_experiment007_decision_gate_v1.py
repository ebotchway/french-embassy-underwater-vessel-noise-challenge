#!/usr/bin/env python3
"""Read-only evidence gate for completed Experiment 007."""
from __future__ import annotations
import argparse, hashlib, json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT=Path('reports/baselines/experiment007_source_disjoint_v1')
OUTPUT=Path('reports/audits/experiment007_decision_gate_v1')
POLICIES=('original_only','contrast_varied','hybrid')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def read_input(root):
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    if manifest.get('source_partition')!='exp004_train ONLY':
        raise ValueError('Source partition changed; audit stopped')
    for name,expected in manifest['output_sha256'].items():
        path=root/name
        if not path.is_file() or digest(path)!=expected:
            raise ValueError('Integrity failure: '+str(path))
    files={name:pd.read_csv(root/name) for name in (
        'assessment_predictions.csv','calibration_predictions.csv','assessment_policy_metrics.csv',
        'threshold_selection_grid.csv','assessment_run_bootstrap.csv',
        'assessment_source_deletion.csv','assessment_segments.csv',
        'source_overlap_audit.csv','background_group_allocation.csv','vessel_group_allocation.csv')}
    return manifest,files

def check(files):
    overlap=files['source_overlap_audit.csv']
    if len(overlap)!=9 or overlap.overlap_count.ne(0).any():raise ValueError('Source overlap audit not clean')
    for field,name in [('run_id','background_group_allocation.csv'),('hull_id','vessel_group_allocation.csv')]:
        table=files[name]
        if table[field].duplicated().any() or set(table.partition)!={'train','calibration','assessment'}:
            raise ValueError('Group allocation anomaly: '+name)
    for name in ('calibration_predictions.csv','assessment_predictions.csv'):
        df=files[name]
        if set(df.policy)!=set(POLICIES):raise ValueError('Missing policy: '+name)
        keys=['pair','spectral_intervention','requested_share']
        groups={p:df[df.policy==p].sort_values(keys).reset_index(drop=True) for p in POLICIES}
        for p,g in groups.items():
            if g.duplicated(keys).any():raise ValueError('Duplicated prediction case '+p)
            if not g.predicted_share.between(0,1).all() or not g.detection_score.between(0,1).all():
                raise ValueError('Prediction range invalid')
        for p in ('contrast_varied','hybrid'):
            if not groups['original_only'][keys].equals(groups[p][keys]):
                raise ValueError('Unmatched scenarios')
        if not np.allclose(groups['original_only'].detection_score,groups['hybrid'].detection_score,atol=0,rtol=0):
            raise ValueError('Hybrid detector mismatch')
        if not np.allclose(groups['contrast_varied'].predicted_share,groups['hybrid'].predicted_share,atol=0,rtol=0):
            raise ValueError('Hybrid regressor mismatch')
        for p,g in groups.items():
            if not np.allclose(g.no_addition_control.astype(int),g.requested_share.eq(0).astype(int)):
                raise ValueError('No-addition label mismatch')
            if not np.allclose(g.observed_positive.astype(int),g.observed_share.gt(1e-12).astype(int)):
                raise ValueError('Observed-positive label mismatch')
    return True

def summarize_deletion(src):
    out=[]
    for key,g in src.groupby('deleted_type'):
        out.append({'deleted_type':key,'groups':g.deleted_group.nunique(),
                    'min_rmse_reduction':g.rmse_reduction.min(),
                    'max_rmse_reduction':g.rmse_reduction.max(),
                    'min_auc_gain':g.auc_gain.min(),
                    'max_auc_gain':g.auc_gain.max(),
                    'min_fpr_increase_at_half':g.fpr_no_addition_increase.min(),
                    'max_fpr_increase_at_half':g.fpr_no_addition_increase.max(),
                    'rmse_nonpositive_count':int((g.rmse_reduction<=0).sum()),
                    'auc_nonpositive_count':int((g.auc_gain<=0).sum()),
                    'fpr_nonpositive_count':int((g.fpr_no_addition_increase<=0).sum())})
    return pd.DataFrame(out)

def calibration_gap(files):
    policy=files['assessment_policy_metrics.csv']
    grid=files['threshold_selection_grid.csv']
    rows=[]
    for p in POLICIES:
        selected=policy[(policy.policy==p)&(policy.threshold_policy=='calibration_selected')]
        if len(selected)!=1:continue
        observed=selected.iloc[0]
        feasible=grid[(grid.policy==p)&(grid.meets_cap.astype(str).str.lower().eq('true'))]
        matching=feasible[np.isclose(feasible.threshold.astype(float),float(observed.threshold))]
        if len(matching)!=1:raise ValueError('Threshold missing from feasible calibration grid: '+p)
        row=matching.iloc[0]
        rows.append({'policy':p,'calibration_selected_threshold':observed.threshold,
                     'calibration_fpr_no_addition':row.calibration_fpr_no_addition,
                     'assessment_fpr_no_addition':observed.fpr_no_addition,
                     'calibration_tpr':row.calibration_tpr,
                     'assessment_tpr':observed.tpr_observed_positive,
                     'assessment_minus_calibration_fpr':observed.fpr_no_addition-row.calibration_fpr_no_addition,
                     'assessment_fpr_below_0_25':bool(observed.fpr_no_addition<=.25)})
    return pd.DataFrame(rows)

def counts_by_source(files):
    df=files['assessment_predictions.csv']; df=df[df.policy=='original_only']
    pairs=df.drop_duplicates('pair')
    result={'pairs':len(pairs),'unique_background_runs':pairs.background_run_id.nunique(),
            'unique_vessel_hulls':pairs.vessel_hull_id.nunique(),
            'unique_vessel_clips':pairs.vessel_clip_uid.nunique(),
            'stations':pairs.station.nunique(),
            'requested_positive_observed_zero_rows':int(df.requested_positive_observed_zero.sum()),
            'genuine_no_addition_rows':int(df.no_addition_control.sum()),
            'observed_positive_rows':int(df.observed_positive.sum()),
            'assessment_prevalence_observed_positive':float(df.observed_positive.mean())}
    return result

def decision(files):
    m=files['assessment_policy_metrics.csv']
    a=m[(m.policy=='original_only')&(m.threshold_policy=='.5_fixed')].iloc[0]
    b=m[(m.policy=='contrast_varied')&(m.threshold_policy=='calibration_selected')].iloc[0]
    h=m[(m.policy=='hybrid')&(m.threshold_policy=='.5_fixed')].iloc[0]
    return {'rmse_reduction_original_vs_varied':float(a.rmse-b.rmse),
            'rmse_reduction_pct':float((a.rmse-b.rmse)/a.rmse*100),
            'fixed_hybrid_no_addition_fpr':float(h.fpr_no_addition),
            'calibrated_varied_assessment_fpr':float(b.fpr_no_addition),
            'calibrated_varied_assessment_tpr':float(b.tpr_observed_positive),
            'hybrid_fraction_equal_varied':bool(np.isclose(h.rmse,b.rmse)),
            'assessment_can_be_reused_for_selection':False,
            'model_promoted_to_zindi':False}

def audit(root,output):
    if output.exists():raise FileExistsError('Refusing to overwrite existing audit: '+str(output))
    manifest,files=read_input(root)
    check(files)
    deletion=summarize_deletion(files['assessment_source_deletion.csv'])
    calibration=calibration_gap(files)
    coverage=counts_by_source(files)
    results=decision(files)
    policy=files['assessment_policy_metrics.csv'].copy()
    policy=policy[['policy','threshold_policy','threshold','rmse','mae','bias','auc','ap',
                   'tpr_observed_positive','fpr_no_addition','fpr_observed_zero',
                   'n_no_addition','n_observed_positive']]
    bootstrap=files['assessment_run_bootstrap.csv']
    source_note='Input Experiment 007 results validated by recorded SHA-256 and group-overlap audits.'
    note=['# Experiment 007 — Read-only assessment decision gate','',source_note,'',
          '## Predeclared policy outcomes','',policy.to_markdown(index=False,floatfmt='.5f'),'',
          '## Calibration-to-assessment differences','',calibration.to_markdown(index=False,floatfmt='.5f'),'',
          '## Leave-one-source-out summary','',deletion.to_markdown(index=False,floatfmt='.5f'),'',
          '## Assessment run-bootstrap (reported; not rerun or tuned)','',
          bootstrap.to_markdown(index=False,floatfmt='.5f'),'',
          '## Scientific interpretation','',
          f"- RMSE reduction for varied/hybrid regression: {results['rmse_reduction_original_vs_varied']:.6f} ({results['rmse_reduction_pct']:.1f}% relative).",
          '- The hybrid is not an additional independently fitted model; its detection score exactly equals the original-only detector, and its predicted fraction exactly equals the varied regressor.',
          '- Calibration selected the threshold using calibration sources only. Assessment outcomes must NOT be used to change the threshold or hyperparameters while calling the same assessment independent.',
          '- The 25% false-positive cap applies only to calibration. Assessment source counts can be small; threshold false-positive figures have binomial granularity.',
          '- Run-level bootstrap intervals are conditional and do not fully represent reused hulls or recorder deployment uncertainty.',
          '- Source split independence is internal to Experiment 007 and not pristine external validation: the broader pool was explored previously.',
          '- No real natural-vessel labels, physical acoustic calibration, or verified deployment coordinates were acquired in this audit.',
          '- No Zindi second submission is justified solely by these results.',
          '', '## Immediate research gates','',
          '1. Preserve the Experiment 007 assessment and avoid threshold/model retuning on these rows.',
          '2. Verify and publicly declare actual VTUAD release/version, source/license/access details.',
          '3. Establish authoritative hydrophone deployments before spatial activity maps; working station candidates are not exact sensor positions.',
          '4. Prepare manuscript figures, coverage, calibration caveats, source limitations and reproducibility appendix.',
          '5. Make any further model evaluation a separately preregistered single use of reserved sources, after the research report evidence is assembled.',
          '']
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='exp007_gate_',dir=output.parent) as tmp:
        stage=Path(tmp)
        policy.to_csv(stage/'policy_comparison.csv',index=False)
        calibration.to_csv(stage/'calibration_assessment_gap.csv',index=False)
        deletion.to_csv(stage/'source_deletion_summary.csv',index=False)
        (stage/'RESEARCH_NOTE.md').write_text('\n'.join(note))
        metadata={'source_root':str(root),'source_manifest_sha256':digest(root/'experiment_manifest.json'),
                  'audit':'read_only_experiment007_decision_gate_v1',
                  'source_partition':manifest['source_partition'],
                  'coverage':coverage,'decision':results,
                  'input_sha256':{k:digest(root/k) for k in (
                    'assessment_predictions.csv','calibration_predictions.csv','assessment_policy_metrics.csv',
                    'threshold_selection_grid.csv','assessment_run_bootstrap.csv','assessment_source_deletion.csv')},
                  'limitations':['Previously inspected underlying sources','Source-dependent uncertainty not fully independent',
                    'Synthetic labels are not field vessel events']}
        (stage/'AUDIT_MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if output.exists():raise FileExistsError('Audit output appeared while staging')
        stage.rename(output)
    print('EXPERIMENT 007 READ-ONLY DECISION GATE: PASS')
    print(f"Assessment pairs: {coverage['pairs']}; runs: {coverage['unique_background_runs']}; hulls: {coverage['unique_vessel_hulls']}; clips: {coverage['unique_vessel_clips']}")
    print(policy[['policy','threshold_policy','rmse','auc','tpr_observed_positive','fpr_no_addition']].to_string(index=False))
    print('Output:',output)

def self_test():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'check';p.mkdir()
        a=pd.DataFrame({'deleted_type':['station','station','vessel_hull_id'],
          'deleted_group':['a','b','h'], 'rmse_reduction':[.01,.02,.03],
          'auc_gain':[.03,.04,.05], 'fpr_no_addition_increase':[.1,.2,.3]})
        t=summarize_deletion(a)
        assert len(t)==2 and t.rmse_nonpositive_count.eq(0).all()
        assert abs(t.loc[t.deleted_type=='station','min_rmse_reduction'].iloc[0]-.01)<1e-10
        assert not p.joinpath('new').exists()
    print('EXPERIMENT 007 DECISION GATE SELF-TEST: PASS')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--self-test-only',action='store_true')
    a=parser.parse_args()
    self_test()
    if not a.self_test_only:audit(a.input,a.output)
if __name__=='__main__':main()
