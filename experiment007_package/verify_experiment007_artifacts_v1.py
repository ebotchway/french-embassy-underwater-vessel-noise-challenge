#!/usr/bin/env python3
"""Read-only integrity verifier for Experiment 007 generated outputs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT=Path('reports/baselines/experiment007_source_disjoint_v1')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()

def validate(root):
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    if manifest.get('source_partition')!='exp004_train ONLY':raise ValueError('Unexpected source partition')
    for name,expected in manifest['output_sha256'].items():
        path=root/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError(f'Missing/modified output: {path}')
    overlap=pd.read_csv(root/'source_overlap_audit.csv')
    if len(overlap)!=9 or not overlap.overlap_count.eq(0).all():raise ValueError('Source overlap detected')
    bg=pd.read_csv(root/'background_group_allocation.csv')
    vt=pd.read_csv(root/'vessel_group_allocation.csv')
    if bg.run_id.duplicated().any() or vt.hull_id.duplicated().any():raise ValueError('Duplicate group allocation')
    for p in ['train','calibration','assessment']:
        if not bg.partition.eq(p).any() or not vt.partition.eq(p).any():
            raise ValueError(f'Missing {p} source groups')
    cal=pd.read_csv(root/'calibration_predictions.csv')
    ass=pd.read_csv(root/'assessment_predictions.csv')
    for frame,name in [(cal,'calibration'),(ass,'assessment')]:
        if set(frame.policy)!=set(('original_only','contrast_varied','hybrid')):raise ValueError(f'Missing {name} policy')
        for p, part in frame.groupby('policy'):
            if part.duplicated(subset=['pair','spectral_intervention','requested_share']).any():
                raise ValueError(f'Duplicate scenario in {name} {p}')
            expected=7*(manifest['calibration_pairs'] if name=='calibration' else manifest['assessment_pairs'])
            if len(part)!=expected:raise ValueError(f'Wrong scenario count for {name} {p}')
            if not part.predicted_share.between(0,1).all() or not part.detection_score.between(0,1).all():
                raise ValueError('Out-of-range predictions')
        baseline=frame.loc[frame.policy.eq('original_only')].sort_values(['pair','requested_share','spectral_intervention'])
        varied=frame.loc[frame.policy.eq('contrast_varied')].sort_values(['pair','requested_share','spectral_intervention'])
        hybrid=frame.loc[frame.policy.eq('hybrid')].sort_values(['pair','requested_share','spectral_intervention'])
        if not np.array_equal(baseline.detection_score.to_numpy(),hybrid.detection_score.to_numpy()):
            raise ValueError(f'Hybrid detector mismatch in {name}')
        if not np.array_equal(varied.predicted_share.to_numpy(),hybrid.predicted_share.to_numpy()):
            raise ValueError(f'Hybrid regressor mismatch in {name}')
    policies=pd.read_csv(root/'assessment_policy_metrics.csv')
    if not set(policies.policy)=={'original_only','contrast_varied','hybrid'}:raise ValueError('Policy metrics incomplete')
    if not policies.threshold_policy.eq('.5_fixed').groupby(policies.policy).any().all():
        raise ValueError('Fixed threshold baseline missing')
    print('EXPERIMENT 007 OUTPUT INTEGRITY: PASS')
    print('Source group overlap: ZERO; SHA256 checks: PASS; hybrid equality: PASS')
    print('Assessment policies:\n'+policies[['policy','threshold_policy','threshold','rmse','auc','tpr_observed_positive','fpr_no_addition']].to_string(index=False))
    print('Note: verifies structural integrity only, not acoustic realism or external generalization.')

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--input',type=Path,default=DEFAULT)
    args=a.parse_args();validate(args.input)

if __name__=='__main__':main()
