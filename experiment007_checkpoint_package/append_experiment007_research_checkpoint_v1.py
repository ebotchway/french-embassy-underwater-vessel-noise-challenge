#!/usr/bin/env python3
"""Append Experiment 007 checkpoint to current local notebook/log/checkpoint (safe, idempotent).

Reads only Experiment 007 report and optional read-only decision gate. No model training.
Dry-run by default; --apply backs up originals then replaces only these three files.
"""
from __future__ import annotations
import argparse, copy, datetime, hashlib, json, shutil, tempfile
from pathlib import Path
import pandas as pd

MARKER='EXPERIMENT007_SOURCE_DISJOINT_CHECKPOINT_2026_11_22'
DOCS={'notebook':Path('notebooks/01_south_african_acoustic_eda.ipynb'),
      'log':Path('PROJECT_LEARNING_LOG.md'), 'checkpoint':Path('CHECKPOINT.md')}
ROOT=Path('reports/baselines/experiment007_source_disjoint_v1')
GATE=Path('reports/audits/experiment007_decision_gate_v1')
MANUSCRIPT=Path('reports/research_manuscript_v1/EXPERIMENT007_RESULTS_ADDENDUM.md')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def checkpoint_text(policy,root,gate):
    def metric(model,selection):
        f=policy[(policy.policy==model)&(policy.threshold_policy==selection)]
        if len(f)!=1:raise ValueError('Missing single policy '+model+'/'+selection)
        return f.iloc[0]
    orig=metric('original_only','.5_fixed'); varied=metric('contrast_varied','calibration_selected')
    hybrid=metric('hybrid','.5_fixed'); varied_fixed=metric('contrast_varied','.5_fixed')
    if abs(float(hybrid.rmse)-float(varied.rmse))>1e-12:raise ValueError('Hybrid regression mismatch')
    if abs(float(hybrid.auc)-float(orig.auc))>1e-12:raise ValueError('Hybrid detector mismatch')
    if not (float(varied.rmse)<float(orig.rmse)):
        outcome='The contrast-varied regressor did not improve the prespecified assessment RMSE.'
    else:
        outcome='The contrast-varied regressor improved the prespecified internal assessment RMSE.'
    section=f'''## Experiment 007 — Source-disjoint training, calibration and assessment ({MARKER})

**Execution evidence:** The researcher's local execution of `scripts/run_experiment007_source_disjoint_v1.py` reported `EXPERIMENT 007: PASS` and `scripts/verify_experiment007_artifacts_v1.py` reported `OUTPUT INTEGRITY: PASS`, with zero known background-run, hull or clip overlap across the three allocations. Source allocations were performed *before* pairing. Implementation commit: `de051b3`. This is a **new internal synthetic assessment** from `exp004_train` only; its source pools were studied historically, so it is neither a pristine external holdout nor real-vessel validation.

**Assessment:** 120 paired background/source conditions, 840 cases per model, 120 genuine no-addition controls, and 714 observed positives (6 requested-positive cases became observed-zero after masking). Both model families were trained on 1,200 simulated windows; threshold selection used a separate calibration partition. Five stations were assessed; Durban was training-only due to a single eligible background run.

| Model and threshold | Fraction RMSE | Detector AUC | Observed-positive TPR | Genuine no-addition FPR |
|---|---:|---:|---:|---:|
| Original-only, fixed 0.50 | {orig.rmse:.6f} | {orig.auc:.6f} | {orig.tpr_observed_positive:.4f} | {orig.fpr_no_addition:.4f} |
| Contrast-varied, fixed 0.50 | {varied_fixed.rmse:.6f} | {varied_fixed.auc:.6f} | {varied_fixed.tpr_observed_positive:.4f} | {varied_fixed.fpr_no_addition:.4f} |
| Contrast-varied, calibration-selected {varied.threshold:.2f} | {varied.rmse:.6f} | {varied.auc:.6f} | {varied.tpr_observed_positive:.4f} | {varied.fpr_no_addition:.4f} |
| Hybrid varied regressor + original detector, fixed 0.50 | {hybrid.rmse:.6f} | {hybrid.auc:.6f} | {hybrid.tpr_observed_positive:.4f} | {hybrid.fpr_no_addition:.4f} |

{outcome} The absolute RMSE reduction against original-only was **{orig.rmse-varied.rmse:.6f}** (~**{100*(orig.rmse-varied.rmse)/orig.rmse:.1f}%**). The contrasted detector's ROC AUC was **{varied.auc:.6f}**, versus original-only **{orig.auc:.6f}**, but the **fixed 0.50 FPR** increased from **{orig.fpr_no_addition:.4f}** to **{varied_fixed.fpr_no_addition:.4f}**. The *calibration-selected* threshold {varied.threshold:.2f} yielded assessment FPR **{varied.fpr_no_addition:.4f}** and sensitivity **{varied.tpr_observed_positive:.4f}**. The 25% FPR limit was imposed only during calibration; assessment need not meet it.

**Hybrid interpretation:** The hybrid is not a third trained model: its share predictions are those of the contrast-varied regressor and its detection scores those of the original-only classifier. Fixed 0.5 hybrid FPR **{hybrid.fpr_no_addition:.4f}** and TPR **{hybrid.tpr_observed_positive:.4f}** equal the original detector by construction. The two best candidates reflect a real specificity-versus-sensitivity trade-off, not a single universal winner. ROC AUC is a ranking metric; the synthetic positive prevalence makes AP unsuitable as an estimate of natural event precision.

**Integrity and scientific limits:** The independent output verifier checked the saved results and hashes and found no known source-group overlap. Run-group bootstraps remain conditional on the split and do not make hulls or deployments independent. Relative-energy mixing has no cross-corpus physical sound-pressure calibration; VTUAD candidate vessel clips include ambient sound. Neither model has independent real South African vessel-event ground truth. Station-level position candidates do not establish hydrophone coordinates. The Experiment 004 frozen evaluation, original validation, sealed station holdouts, and hidden Zindi Test labels were not accessed in this experiment.

**Decision:** Freeze Experiment 007 predictions and calibration thresholds. Do not retune using its assessment outputs, choose a new threshold after inspecting results, or submit a second Zindi candidate solely on synthetic evidence. Preserve both the hybrid and calibrated varied-detector configurations as *research policies*, and finish the scientific manuscript, source/version declaration and location/measurement provenance checks. Additional comparison should be independently predeclared before accessing reserved data.

**Files:** `reports/baselines/experiment007_source_disjoint_v1/`; `reports/audits/experiment007_decision_gate_v1/` (if executed); scripts `run_experiment007_source_disjoint_v1.py`, `verify_experiment007_artifacts_v1.py`, `audit_experiment007_decision_gate_v1.py`. Earlier Phase 3 content is already incorporated in `CHECKPOINT.md`; there is no requirement to restore a separate Phase 3 file.
'''
    if gate.exists() and (gate/'AUDIT_MANIFEST.json').is_file():
        obj=json.loads((gate/'AUDIT_MANIFEST.json').read_text())
        section+='\n**Read-only decision-gate audit:** Output manifest present; all policy interpretations remain descriptive and no further model selection was performed. Verified gate source SHA256: `'+obj['source_manifest_sha256']+'`.\n'
    return section

def update(args):
    policy=pd.read_csv(args.results/'assessment_policy_metrics.csv')
    required=set(('policy','threshold_policy','rmse','auc','fpr_no_addition','tpr_observed_positive','threshold'))
    if not required.issubset(policy.columns):raise ValueError('Missing policy metrics fields')
    section=checkpoint_text(policy,args.results,args.gate)
    docs={name:args.project/path for name,path in DOCS.items()}
    for name,path in docs.items():
        if not path.is_file():raise FileNotFoundError(f'Required {name} file missing: {path}')
    if args.manuscript.exists():raise FileExistsError('Manuscript addendum already exists: '+str(args.manuscript))
    old_log=docs['log'].read_text();old_checkpoint=docs['checkpoint'].read_text()
    nb=json.loads(docs['notebook'].read_text())
    if any(MARKER in s for s in (old_log,old_checkpoint)) or any(MARKER in ''.join(c.get('source',[])) if isinstance(c.get('source',[]),list) else MARKER in c.get('source','') for c in nb['cells']):
        raise ValueError('Checkpoint marker exists; refusing duplicate append')
    original=copy.deepcopy(nb)
    cell={'cell_type':'markdown','id':'exp007-source-disjoint-checkpoint','metadata':{},'source':section.splitlines(keepends=True)}
    nb['cells'].append(cell)
    assert nb['cells'][:-1]==original['cells']
    old_outputs=[(c.get('execution_count'),copy.deepcopy(c.get('outputs',[]))) for c in original['cells'] if c['cell_type']=='code']
    new_outputs=[(c.get('execution_count'),copy.deepcopy(c.get('outputs',[]))) for c in nb['cells'][:-1] if c['cell_type']=='code']
    if old_outputs!=new_outputs:raise ValueError('Notebook outputs changed unexpectedly')
    new_log=old_log.rstrip()+'\n\n'+section+'\n'
    summary='''# Current study status — after Experiment 007 (22 November 2026)

Experiment 007 completed a prespecified training/calibration/assessment split **before** acoustic pairing, using only Experiment 004's training source pool. Source overlap and output hashes passed independent verification. The assessment documents improved synthetic share RMSE for the contrast-varied regressor but competing detector specificity/sensitivity trade-offs; no real-vessel generalization is established. The Experiment 007 assessment is frozen: do not select or retune a threshold using its results. Phase 3 remains preserved in the existing history of this checkpoint.

'''
    new_checkpoint=summary+old_checkpoint.rstrip()+'\n\n'+section+'\n'
    if not args.apply:
        print('EXPERIMENT 007 DOCUMENTATION DRY RUN: PASS')
        print('Will append one notebook markdown cell and one learning-log section; prepend current status and append section to CHECKPOINT.md.')
        print(f'Notebook code cells/outputs preserved: {len(old_outputs)} code cells; original total cells: {len(original['cells'])}')
        print('Will create manuscript addendum:',args.manuscript)
        return
    args.backups.mkdir(parents=True,exist_ok=True)
    # Backups are produced before changes, and the three new bodies are staged before any replacement.
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backups={}
    for name,path in docs.items():
        target=args.backups/f'{path.stem}_before_{MARKER}_{stamp}{path.suffix}'
        if target.exists():raise FileExistsError(target)
        shutil.copy2(path,target);backups[name]=target
    args.manuscript.parent.mkdir(parents=True,exist_ok=True)
    staged={}
    try:
        for name,content in [('log',new_log),('checkpoint',new_checkpoint)]:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=docs[name].parent,
                       prefix=f'.exp007_{name}_',suffix='.tmp',delete=False) as temp:
                temp.write(content);staged[name]=Path(temp.name)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=docs['notebook'].parent,
                       prefix='.exp007_notebook_',suffix='.tmp',delete=False) as temp:
            temp.write(json.dumps(nb,ensure_ascii=False,indent=1)+'\n');staged['notebook']=Path(temp.name)
        validated=json.loads(staged['notebook'].read_text())
        if validated['cells'][:-1]!=original['cells']:raise ValueError('Staged notebook cells changed')
        for name,path in docs.items():staged[name].replace(path)
        args.manuscript.write_text(section)
    finally:
        for path in staged.values():
            if path.exists():path.unlink()
    print('EXPERIMENT 007 DOCUMENTATION APPLY: PASS')
    print('Backups:', *(str(p) for p in backups.values()),sep='\n  ')
    print('Manuscript addendum:',args.manuscript)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path('.'))
    parser.add_argument('--results',type=Path,default=ROOT)
    parser.add_argument('--gate',type=Path,default=GATE)
    parser.add_argument('--manuscript',type=Path,default=MANUSCRIPT)
    parser.add_argument('--backups',type=Path,default=Path('reports/research_backups/experiment007_checkpoint'))
    parser.add_argument('--apply',action='store_true')
    a=parser.parse_args();update(a)
if __name__=='__main__':main()
