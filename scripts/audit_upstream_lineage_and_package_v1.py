#!/usr/bin/env python3
"""Read-only source lineage audit + guarded *candidate* reproduction-code ZIP.

Never reads acoustic data rows, hidden/test/validation/eval partitions or audio.
Never runs scientific modelling. Does not certify upstream raw-data reproduction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

INPUTS = {
    'station_summary': Path('reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv'),
    'spectral_shapes': Path('reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv'),
    'relative_proxy': Path('reports/research_manuscript_v1/station_relative_anomaly_proxy.csv'),
}
ORIGINAL = Path('reports/research_manuscript_v1/real_acoustic_results_v2/manifest.json')
REPRO = Path('reports/research_manuscript_v1/real_acoustic_reproduction_phase9_v1/manifest.json')
CANDIDATE_SCRIPTS = (
    'audit_real_acoustic_development_v1.py',
    'build_real_acoustic_results_v2.py',
    'verify_real_acoustic_results_v2.py',
    'summarize_real_acoustic_development_v1.py',
)
# Never include data, notebooks with saved outputs, partitions, user secrets or results.
FORBIDDEN = re.compile(r'(?i)(^|/)(data|reports|\.venv[^/]*|test\.csv|secrets?|\.env|keys?)(/|$)')
MAX_BYTES = 10_000_000

def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def inspect(root: Path) -> dict:
    mpath=root/ORIGINAL
    if not mpath.is_file(): raise FileNotFoundError(f'Original appendix manifest missing: {mpath}')
    m=json.loads(mpath.read_text(encoding='utf-8'))
    recorded=m.get('source_sha256',{})
    if not all(k in recorded for k in INPUTS):
        raise ValueError('Original manifest missing expected input hashes')
    checks=[]
    for k,rel in INPUTS.items():
        p=root/rel
        actual=digest(p) if p.is_file() else None
        checks.append({'id':k,'path':rel.as_posix(),'status':'PASS' if actual==recorded[k] else 'FAIL',
                       'expected_sha256':recorded[k],'actual_sha256':actual})
    rm=root/REPRO
    repro_state='NOT_FOUND'
    if rm.is_file():
        repro_manifest=json.loads(rm.read_text(encoding='utf-8'))
        repro_state='MATCH' if repro_manifest.get('source_sha256')==recorded and repro_manifest.get('bootstrap_draws')==m.get('bootstrap_draws') and repro_manifest.get('random_seed')==m.get('random_seed') else 'MISMATCH'
    # A textual producer reference is *candidate evidence*, not an authenticated dependency chain.
    producer=root/'scripts/audit_real_acoustic_development_v1.py'
    snippets={}
    if producer.is_file():
        content=producer.read_text(encoding='utf-8',errors='replace')
        for k in ('station_summary','spectral_shapes'):
            snippets[k]='WRITE_REFERENCE_FOUND' if re.search(r"to_csv\([^\n]{0,120}['\"]"+re.escape(INPUTS[k].name)+r"['\"]",content) else ('FILENAME_MENTION_ONLY' if INPUTS[k].name in content else 'NO_REFERENCE')
    else:
        snippets={k:'SCRIPT_MISSING' for k in ('station_summary','spectral_shapes')}
    hits=[]
    # Only inspect source code; never walk data/ or read reports outside the two manifests.
    for d in ('scripts','notebooks'):
        base=root/d
        if not base.is_dir(): continue
        for p in sorted(base.glob('*.py' if d=='scripts' else '*.ipynb')):
            if p.stat().st_size>3_000_000: continue
            content=p.read_text(encoding='utf-8',errors='replace')
            if INPUTS['relative_proxy'].name in content:
                hits.append({'source':p.relative_to(root).as_posix(),
                             'contains_proxy_filename':True,
                             'contains_to_csv': 'to_csv' in content,
                             'role':'CANDIDATE_ONLY_REQUIRES_MANUAL_INSPECTION'})
    return {'status':'EVIDENCE_ONLY_NOT_SUBMISSION_APPROVAL','input_hash_checks':checks,
            'phase9_manifest_status':repro_state,'original_bootstrap_draws':m.get('bootstrap_draws'),
            'original_random_seed':m.get('random_seed'),
            'documented_producer_references':snippets,'proxy_producer_candidates':hits,
            'proxy_producer_verified':False,
            'raw_data_to_derived_inputs_reproduced':False,
            'protected_partitions_read':False,
            'scientific_gates':{'hydrophone_coordinates':'UNVERIFIED','natural_vessel_attribution':'UNVERIFIED',
                                'physical_pressure_calibration':'UNVERIFIED','external_source_public_declaration':'UNVERIFIED'}}

def package(root:Path, output:Path, report:dict)->dict:
    names=[]
    for name in CANDIDATE_SCRIPTS:
        p=root/'scripts'/name
        if p.is_file(): names.append((p,Path('scripts')/name))
    if not any(n.name=='build_real_acoustic_results_v2.py' for p,n in names):
        raise FileNotFoundError('Essential appendix builder unavailable; refusing incomplete code package')
    req=root/'requirements.txt'
    if not req.is_file(): raise FileNotFoundError('requirements.txt missing')
    names.append((req,Path('requirements.txt')))
    for _p,rel in names:
        if FORBIDDEN.search(rel.as_posix()): raise ValueError(f'Unsafe archive member: {rel}')
    readme='''# Candidate research code archive — NOT a full submission or raw-data reproducibility proof

This code-only archive contains selected source scripts and frozen dependencies. No audio,
South African organiser-review Parquet, competition labels, evaluation partitions or results
are redistributed. Do not submit this as proof of a raw-data-to-manuscript pipeline.

Python 3.13 is the tested interpreter. Install `pip install -r requirements.txt`
in a fresh environment. To regenerate the verified appendix **only after acquiring and
preparing the three allowed training-only derived inputs** in the paths below, run:

`python scripts/build_real_acoustic_results_v2.py --root . --output reports/research_manuscript_v1/real_acoustic_reproduction_new --bootstrap 1000 --seed 42`

`python scripts/verify_real_acoustic_results_v2.py --root . --output reports/research_manuscript_v1/real_acoustic_reproduction_new`

Required derived input paths:
- reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv
- reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv
- reports/research_manuscript_v1/station_relative_anomaly_proxy.csv

The provenance audit records source hashes and candidate upstream producers but does not
certify how the run-level proxy was generated. Any full-data claim needs independently
reproduced upstream processing from permitted sources, documented source access and
scientific review. Existing frozen evaluation splits must not be touched.
'''
    zpath=output/'CANDIDATE_REPRODUCTION_CODE.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('README.md',readme)
        for p,rel in names:z.write(p,rel.as_posix())
    if zpath.stat().st_size>MAX_BYTES:
        zpath.unlink()
        raise ValueError('Code archive exceeds 10 MB; removed excess archive')
    with zipfile.ZipFile(zpath) as z:
        entries=z.namelist()
        assert all(not FORBIDDEN.search(e) for e in entries)
        assert z.testzip() is None
    return {'filename':zpath.name,'bytes':zpath.stat().st_size,'sha256':digest(zpath),
            'entries':entries,'scope':'CANDIDATE_CODE_ONLY'}

def execute(root:Path,output:Path)->dict:
    root=root.resolve(strict=True)
    output=output.absolute()
    if output.exists():raise FileExistsError(f'Output exists; refusing overwrite: {output}')
    data=inspect(root)
    output.mkdir(parents=True,exist_ok=False)
    pkg=package(root,output,data)
    data['code_package']=pkg
    (output/'LINEAGE_AUDIT.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    lines=['# Phase 10 — upstream lineage and submission-code candidate','',
           '**Status: evidence only; not competition submission clearance.**','',
           '## Derived input fingerprints','',
           '| Source | Status | SHA-256 |','|---|---|---|']
    for c in data['input_hash_checks']:
        lines.append(f"| `{c['path']}` | {c['status']} | `{c['actual_sha256'] or 'MISSING'}` |")
    lines += ['',f"Phase 9 original-versus-reproduction source manifest: **{data['phase9_manifest_status']}**.",'',
              '## Upstream code evidence','',
              'The station summary and spectral-shape source appears in `audit_real_acoustic_development_v1.py` as candidate producer code. Source-file text evidence is not independent reconstruction.','',
              f"Producer checks: `{json.dumps(data['documented_producer_references'],sort_keys=True)}`.",'',
              'Run-level proxy producer: **NOT VERIFIED**. The following filenames are leads, not proof:']
    for h in data['proxy_producer_candidates']:lines.append(f"- `{h['source']}`")
    if not data['proxy_producer_candidates']:lines.append('- None found in scanned top-level source files.')
    lines += ['','## Candidate code package','',
              f"- `{pkg['filename']}`: {pkg['bytes']:,} bytes; SHA-256 `{pkg['sha256']}`",
              '- Code and dependency pins only; no source data, derived data, notebooks with outputs, audio, or protected partitions.','',
              '## Unresolved scientific requirements','',
              '- Raw acoustic inputs -> three derived inputs: **not independently reproduced**.',
              '- Run-level proxy calculation/provenance: **unverified**.',
              '- Hydrophone deployment coordinates, independent vessel events and physical calibration: **unverified**.',
              '- External-data public Zindi declaration: **unverified**.',
              '- Code ZIP is a **candidate**, not a validated complete standalone reproduction package.','']
    (output/'LINEAGE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PHASE 10:',data['status'])
    print('Input hashes:',','.join(f"{x['id']}={x['status']}" for x in data['input_hash_checks']))
    print('Phase 9 source manifest:',data['phase9_manifest_status'])
    print('Run-level proxy producer: NOT VERIFIED')
    print('Candidate code archive:',zpath_display(output/'CANDIDATE_REPRODUCTION_CODE.zip'))
    print('Report:',output)
    return data

def zpath_display(p:Path)->str:return str(p)

def selftest()->None:
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)/'project';root.mkdir()
        (root/'scripts').mkdir()
        for k,rel in INPUTS.items():
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f'{k}\n')
        (root/'requirements.txt').write_text('numpy==2.5.4\n')
        for n in CANDIDATE_SCRIPTS:
            (root/'scripts'/n).write_text('print("fixture")\n')
        original=root/ORIGINAL;original.parent.mkdir(parents=True,exist_ok=True)
        obj={'source_sha256':{k:digest(root/rel) for k,rel in INPUTS.items()},'bootstrap_draws':1000,'random_seed':42}
        original.write_text(json.dumps(obj))
        repro=root/REPRO;repro.parent.mkdir(parents=True,exist_ok=True);repro.write_text(json.dumps(obj))
        out=Path(t)/'output';result=execute(root,out)
        assert all(x['status']=='PASS' for x in result['input_hash_checks'])
        assert result['phase9_manifest_status']=='MATCH'
        assert not result['proxy_producer_verified']
        assert (out/'CANDIDATE_REPRODUCTION_CODE.zip').is_file()
        try:execute(root,out)
        except FileExistsError:pass
        else:raise AssertionError('Overwrite refusal failed')
        (root/INPUTS['relative_proxy']).write_text('changed\n')
        assert any(x['status']=='FAIL' for x in inspect(root)['input_hash_checks'])
        print('PHASE 10 SELF-TEST: PASS (hash tampering, safe archive, non-overwrite)')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('reports/audits/phase10_upstream_lineage_v1'))
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only:selftest()
    else:execute(a.root,a.output)
if __name__=='__main__':main()
