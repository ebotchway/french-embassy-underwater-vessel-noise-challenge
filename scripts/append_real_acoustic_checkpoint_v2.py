#!/usr/bin/env python3
"""Opt-in, idempotent update of the research log/checkpoint/notebook from verified local appendix.

Dry run by default. Does not execute notebook or alter existing cells/outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

TAG='REAL_ACOUSTIC_RESULTS_APPENDIX_V2_2026_11_22'
NOTEBOOK=Path('notebooks/01_south_african_acoustic_eda.ipynb')
LOG=Path('PROJECT_LEARNING_LOG.md')
CHECKPOINT=Path('CHECKPOINT.md')
OUT=Path('reports/research_manuscript_v1/real_acoustic_results_v2')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build_note(manifest):
    ck=manifest['input_checks']
    return f'''## Real-acoustic descriptive research appendix — {TAG}

Training-only existing South African station/run acoustic results have been independently inventoried and re-presented without retraining or examining hidden Zindi targets.

- {ck['stations']} stations, {ck['runs']} unique sampled run entries and {ck['spectral_bands_per_station']} frequency bands per station in the existing station spectral-shape report.
- Tables, five figures and conditional run-bootstrap intervals: `reports/research_manuscript_v1/real_acoustic_results_v2/REAL_ACOUSTIC_RESULTS_APPENDIX.md`.
- Draft PDF: `reports/research_manuscript_v1/real_acoustic_results_v2/REAL_ACOUSTIC_RESULTS_APPENDIX.pdf`.
- Provenance and input SHA-256: `reports/research_manuscript_v1/real_acoustic_results_v2/manifest.json`.
- No bootstrap interval for stations with fewer than 10 sampled recording runs; background run bootstrap is not deployment-independent.
- Undefined `station_robust_z` entries are explicitly counted in `robust_z_availability.csv`; no undefined score is replaced by zero or treated as a valid observation.
- **Limits:** Archive level units/reference and calibration are not verified. No natural-vessel annotations, hydrophone coordinates, spatial interpolation, or absolute sound-pressure claims are established.
- Separate synthetic Experiments 005–007 and the model decision gate remain distinct from these descriptive acoustic figures; the source pool is historical development data.
- Open evidence gates: exact local VTUAD release, public external-data declaration, verified sensor deployment positions, calibration, and natural-vessel attribution.

Do not describe this appendix as a completed submission-ready vessel-activity map. Preserve the earlier Experiment 007 assessment without further tuning.
'''

def apply(root, out, do_apply=False):
    root=Path(root);out=Path(out)
    paths=[root/NOTEBOOK,root/LOG,root/CHECKPOINT]
    for p in [*paths,out/'manifest.json',out/'REAL_ACOUSTIC_RESULTS_APPENDIX.md',out/'REAL_ACOUSTIC_RESULTS_APPENDIX.pdf']:
        if not p.is_file():raise FileNotFoundError(str(p))
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['analysis']!='real_acoustic_results_v2':raise ValueError('wrong appendix manifest')
    for name,digest in manifest['outputs_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('appendix artifact modified since build: '+name)
    note=build_note(manifest)
    nb=json.loads(paths[0].read_text())
    if not isinstance(nb.get('cells'),list):raise ValueError('notebook cells missing')
    oldcode=[(c.get('source'),c.get('outputs'),c.get('execution_count')) for c in nb['cells'] if c.get('cell_type')=='code']
    originals=[p.read_bytes() for p in paths]
    if any(TAG.encode() in content for content in originals):
        raise FileExistsError('Checkpoint marker already present in at least one document; refusing duplicate/partial update')
    newcell={'cell_type':'markdown','metadata':{'tags':[TAG]},'source':note.splitlines(True)}
    nb['cells'].append(newcell)
    updated_nb=json.dumps(nb,ensure_ascii=False,indent=1)+'\n'
    newer=json.loads(updated_nb)
    newcode=[(c.get('source'),c.get('outputs'),c.get('execution_count')) for c in newer['cells'] if c.get('cell_type')=='code']
    if oldcode!=newcode:raise AssertionError('notebook code/output changed')
    log=originals[1].decode('utf-8').rstrip()+'\n\n'+note+'\n'
    checkpoint='### Current real-acoustic results appendix — '+TAG+'\n\n'+note.split('\n\n',1)[1]+'\n\n'+originals[2].decode('utf-8')
    if not do_apply:
        print('REAL ACOUSTIC CHECKPOINT DRY RUN: PASS')
        print(f'Will append 1 notebook markdown cell; existing code cells/outputs preserved: {len(oldcode)} code cells')
        print('Will append learning-log section and prepend current-status section to CHECKPOINT.md')
        return
    backup=root/'reports/research_backups/real_acoustic_appendix_v2'
    if backup.exists():raise FileExistsError(f'Backup location already exists: {backup}')
    backup.mkdir(parents=True)
    for p in paths:shutil.copy2(p,backup/p.name)
    payloads=[updated_nb.encode(),log.encode(),checkpoint.encode()]
    # Stage all payloads before changing any existing file.
    staged=[]
    try:
        for p,bytes_ in zip(paths,payloads):
            with tempfile.NamedTemporaryFile(prefix='.acoustic_checkpoint_',dir=p.parent,delete=False) as fp:
                fp.write(bytes_); staged.append(Path(fp.name))
        for p,tmp in zip(paths,staged):tmp.replace(p)
    finally:
        for tmp in staged:tmp.unlink(missing_ok=True)
    print('REAL ACOUSTIC CHECKPOINT APPLY: PASS')
    print(f'Backups: {backup}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--apply',action='store_true')
    a=ap.parse_args();apply(a.root,a.output,a.apply)
if __name__=='__main__':main()
