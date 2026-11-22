#!/usr/bin/env python3
"""Read-only, evidence-based inventory of underwater acoustic research artifacts.

No training, uploads, downloads, data deletions, or reconstruction are performed.
Reports are written only to a new --output directory; existing dirs are refused.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VESSEL_SHA256 = '96564721a917d8b4dd161fdb90c66db917136786e6f4680028be6c6e03cf8ac6'
BACKGROUND_MD5 = 'd28ad0fb6ccc98341ba4f7e953d9c68f'


def digest(path: Path, method: str='sha256') -> str:
    h = hashlib.new(method)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_parts(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
    return h.hexdigest()


def check(root: Path, *, hash_large: bool=False) -> dict:
    checks = []
    def emit(name, status, detail, evidence=None):
        checks.append({'check': name, 'status':status, 'detail':detail, 'evidence': evidence or ''})
    downloads = root/'data/external/vtuad/downloads'
    parts = [downloads/f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in range(1,4)]
    bg = downloads/'vtuad_strict_wav_background.zip'
    reconstructed = downloads/'vtuad_strict_wav_vessel.zip'
    if all(p.is_file() for p in parts):
        emit('vessel_parts_presence', 'PASS', 'Three retained split assets exist', '; '.join(p.name for p in parts))
        if hash_large:
            value = digest_parts(parts)
            emit('vessel_parts_combined_sha256', 'PASS' if value==VESSEL_SHA256 else 'FAIL', 'Combined source-part hash matches previously validated local reference' if value==VESSEL_SHA256 else 'Source parts differ from previously validated local reference',value)
        else:
            emit('vessel_parts_combined_sha256','NOT_RUN','Run with --hash-large to stream-hash all split parts')
    else:
        emit('vessel_parts_presence','FAIL','One or more split parts are missing','; '.join(str(p) for p in parts if not p.is_file()))
    if bg.is_file():
        if hash_large:
            val=digest(bg,'md5')
            emit('background_publisher_md5','PASS' if val==BACKGROUND_MD5 else 'FAIL','Compare against published background asset MD5',val)
        else:
            emit('background_publisher_md5','NOT_RUN','Run with --hash-large to verify against published checksum')
    else:
        emit('background_presence','FAIL','Background publisher ZIP is missing')
    if reconstructed.exists():
        emit('combined_vessel_zip','INFO','Reconstructed ZIP exists; can be redundant after strict parts comparison',str(reconstructed.relative_to(root)))
    else:
        emit('combined_vessel_zip','PASS','Reconstructed combined vessel ZIP intentionally absent; original split parts must be kept')
    provenance = root/'reports/eda/vtuad_source_split_v2.csv'
    if provenance.is_file():
        try:
            with provenance.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.DictReader(f)
                required={'clip_uid','path','label','directory_category','source_split','exists'}
                if not required.issubset(reader.fieldnames or []):
                    emit('vtuad_manifest','FAIL', 'Required column(s) missing',','.join(sorted(required-set(reader.fieldnames or []))))
                else:
                    rows=list(reader)
                    ids=[r['clip_uid'] for r in rows]
                    label_counts=Counter(r['label'].strip().lower() for r in rows)
                    dirs=Counter(r['directory_category'].strip().lower() for r in rows)
                    splits=Counter(r['source_split'].strip() for r in rows)
                    cross=Counter((r['directory_category'].strip().lower(),r['label'].strip().lower()=='background') for r in rows)
                    statuses=[r['exists'].strip().lower() for r in rows]
                    # Do not treat arbitrary nonempty strings as boolean True.
                    true_vals={'true','1','yes'}
                    all_exist=all(x in true_vals for x in statuses)
                    expected= (len(rows)==525 and len(set(ids))==525 and dirs==Counter({'vessel':509,'background':16}) and label_counts['background']==38 and splits==Counter({'train':410,'validation':77,'background_reference':38}) and cross[('vessel',True)]==22 and cross[('background',True)]==16 and cross[('background',False)]==0 and all_exist)
                    emit('vtuad_manifest','PASS' if expected else 'REVIEW','Expected inventory, partition, and directory/metadata cross-tab consistency',json.dumps({'rows':len(rows),'unique_ids':len(set(ids)),'directories':dict(dirs),'background_label_count':label_counts['background'],'source_splits':dict(splits),'vessel_directory_background_label_count':cross[('vessel',True)],'all_exists_flagged':all_exist},sort_keys=True))
        except (OSError, UnicodeError, csv.Error) as exc:
            emit('vtuad_manifest','FAIL',f'Unable to inspect manifest: {exc}')
    else:
        emit('vtuad_manifest','FAIL','Manifest unavailable',str(provenance.relative_to(root)))
    station=root/'reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv'
    if station.is_file():
        try:
            with station.open(newline='',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            total=sum(int(row['sampled_runs']) for row in rows)
            emit('acoustic_stations','PASS' if len(rows)==6 and total==460 else 'REVIEW',f'Stations={len(rows)}; sampled recording runs={total}',str(station.relative_to(root)))
        except (ValueError,KeyError,UnicodeError,OSError,csv.Error) as exc:
            emit('acoustic_stations','FAIL',f'Cannot parse station summary: {exc}')
    else:
        emit('acoustic_stations','FAIL','Station descriptive summary is absent')
    manuscript=root/'reports/research_manuscript_v1/submission_review_phase5_v1/RESEARCH_MANUSCRIPT_REVIEW.pdf'
    if manuscript.is_file():
        n=manuscript.stat().st_size
        with manuscript.open('rb') as f: hdr=f.read(5)
        emit('manuscript_pdf','PASS' if hdr==b'%PDF-' and n<=10_000_000 else 'FAIL',f'PDF header={hdr!r}; bytes={n}; file limit=10,000,000',str(manuscript.relative_to(root)))
    else:
        emit('manuscript_pdf','FAIL','Manuscript PDF absent')
    for rel in [
        'requirements.txt',
        'scripts/audit_submission_provenance_v1.py',
        'scripts/build_real_acoustic_results_v2.py',
        'scripts/verify_real_acoustic_results_v2.py',
        'scripts/build_scientific_submission_review_v1.py',
        'scripts/audit_final_submission_v1.py',
        'scripts/manage_underwater_storage.py',
        'reports/audits/submission_provenance_after_storage_cleanup_v1/EVIDENCE_REPORT.md',
    ]:
        p=root/rel
        if p.is_file():
            emit('file:'+rel,'PASS','File present; SHA256 recorded',digest(p))
        else:
            emit('file:'+rel,'NOT_VERIFIED','Expected working-file path not confirmed')
    # Explicitly no globbing of Test.csv or protected assessment files.
    emit('external_source_declaration','NOT_VERIFIED','Local declaration draft is not evidence of a published Zindi disclosure')
    emit('hydrophone_deployments','NOT_VERIFIED','Precise station coordinates require original deployment metadata')
    emit('vessel_attribution','NOT_VERIFIED','No independent time-aligned AIS or independently validated natural-vessel events established')
    emit('acoustic_calibration','NOT_VERIFIED','Source-export dB values are not verified absolute pressure levels')
    emit('full_scientific_reproduction','NOT_RUN','Must independently execute selected full-data analyses into new output paths and compare numerical results and figures')
    emit('separate_code_upload','NOT_VERIFIED','Runnable code upload and size must be manually verified')
    return {'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'python':sys.version.split()[0],'platform':platform.platform(),'hash_large':hash_large,'checks':checks,'status':'EVIDENCE_ONLY_NOT_SUBMISSION_APPROVAL'}


def report(root:Path, output:Path, hash_large:bool):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if output == root or output == root/'data':
        raise ValueError('Invalid output path')
    result=check(root,hash_large=hash_large)
    # create output only after the read-only checks have finished
    output.mkdir(parents=True,exist_ok=False)
    (output/'EVIDENCE_INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (output/'CHECKS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['check','status','detail','evidence'])
        writer.writeheader();writer.writerows(result['checks'])
    lines=['# Reproduction checkpoint — evidence inventory','','**NOT final submission clearance. No model evaluation was rerun.**','','| Check | Status | Evidence |','|---|---|---|']
    for row in result['checks']:
        v=str(row['detail']).replace('|','/').replace('\n',' ')
        lines.append(f"| `{row['check']}` | {row['status']} | {v} |")
    lines+=['','This inventory never reads protected evaluation samples, never trains models, never reconstructs ZIPs, never uploads data, and does not establish end-to-end computational reproducibility.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    counts=Counter(r['status'] for r in result['checks'])
    print('RESEARCH CHECKPOINT:',result['status'],'checks:',len(result['checks']),'statuses:',dict(counts))
    print('Output:',output)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        (root/'reports/eda').mkdir(parents=True)
        p=root/'reports/eda/vtuad_source_split_v2.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['clip_uid','path','label','directory_category','source_split','exists'])
            w.writeheader()
            for i in range(525):
                category='vessel' if i<509 else 'background'
                background=(487<=i<509 or i>=509)
                w.writerow({'clip_uid':str(i),'path':f'{category}/{i}.wav','label':'background' if background else 'tug','directory_category':category,'source_split':'background_reference' if background else ('train' if i<410 else 'validation'),'exists':'True'})
        results=check(root)
        manifest=next(x for x in results['checks'] if x['check']=='vtuad_manifest')
        assert manifest['status']=='PASS',manifest
        out=root/'audit'
        report(root,out,False)
        try:
            report(root,out,False)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Must not overwrite an existing report')
    print('SELF TEST: PASS — synthetic manifest, and output-overwrite refusal')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--hash-large',action='store_true',help='Stream-hash approximately 4.6 GB of original VTUAD source parts; may take time')
    p.add_argument('--self-test-only',action='store_true')
    args=p.parse_args()
    if args.self_test_only:
        self_test();return
    root=args.root.resolve(strict=True)
    if not root.is_dir():p.error('--root must be a directory')
    out=(args.output or root/'reports/audits/reproduction_checkpoint_v1').resolve()
    report(root,out,args.hash_large)

if __name__=='__main__':
    main()
