#!/usr/bin/env python3
"""Evidence-only Phase 5 provenance ledger. No model, hidden test, publishing or geo-inference.

Run from project root. Do not use this script to certify source release/deployment.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, tempfile
from datetime import datetime, timezone
from pathlib import Path

CARD = Path('data/external/vtuad/metadata/dataset_card.md')
LEDGER = Path('data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt')
STATIONS = Path('data/zindi/extracted/stations.csv')
DEPLOYMENT = Path('reports/research_manuscript_v1/STATION_COORDINATES_PROVENANCE_TEMPLATE.csv')
GATES = Path('reports/audits/research_evidence_phase2_v2/claim_evidence_gates.csv')
SUMMARY = Path('reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv')
DEFAULT_OUT = Path('reports/audits/submission_provenance_phase5_v1')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def tabular(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def check(root, out):
    if out.exists():raise FileExistsError(f'Output already exists; do not overwrite: {out}')
    inputs={'dataset_card':CARD,'strict_sha256_ledger':LEDGER,'site_candidates':STATIONS,
            'deployment_template':DEPLOYMENT,'evidence_gates':GATES,'station_summary':SUMMARY}
    missing={k:str(p) for k,p in inputs.items() if not (root/p).is_file()}
    if missing:raise FileNotFoundError('Missing source inputs: '+json.dumps(missing))
    paths={k:root/v for k,v in inputs.items()}
    text=paths['dataset_card'].read_text(encoding='utf-8',errors='replace')
    versions=sorted(set(re.findall(r'(?i)\bv?1\.\d+\.\d+\b',text)))
    ledger_raw=[line.strip() for line in paths['strict_sha256_ledger'].read_text(errors='replace').splitlines()
        if line.strip() and not line.lstrip().startswith('#')]
    ledger_entries=[]; malformed=[]; duplicate_names=set(); seen=set()
    for i,line in enumerate(ledger_raw,1):
        m=re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+?)\s*',line)
        if not m:
            malformed.append(i);continue
        name=m.group(2).replace('\\','/')
        if name in seen:duplicate_names.add(name)
        seen.add(name)
        ledger_entries.append({'sha256':m.group(1).lower(),'relative_filename':name})
    if malformed or duplicate_names:
        raise ValueError('Malformed/duplicated SHA ledger entries: '+str({'malformed_lines':malformed[:10], 'duplicate_count':len(duplicate_names)}))
    stations=tabular(paths['site_candidates'])
    station_rows=tabular(paths['station_summary'])
    if not stations or not station_rows:raise ValueError('Empty station metadata')
    if not {'station','sampled_runs'}.issubset(station_rows[0]):raise ValueError('Invalid station summary columns')
    gate_rows=tabular(paths['evidence_gates'])
    gate_map={r['gate']:r['status'] for r in gate_rows}
    # Never consider filled coordinates in a generic CSV to be verified deployments.
    # Produce checklist, not georeferenced map; contact source authority for original deployment IDs.
    site_cols=sorted(set().union(*(set(r) for r in stations)))
    candidate_names=[r.get('station',r.get('station_name',r.get('name','(unlabelled)'))) for r in stations]
    review=[{'station_from_acoustic_summary':r['station'],
             'sampled_runs':r['sampled_runs'],
             'original_sensor_id':'', 'original_deployment_id':'',
             'recording_start_utc':'', 'recording_end_utc':'',
             'source_document_url_or_archived_id':'', 'coordinate_crs':'',
             'original_source_latitude':'','original_source_longitude':'',
             'authority_verified_by':'',
             'evidence_status':'NOT_VERIFIED_DO_NOT_MAP'} for r in station_rows]
    # Need to see source asset independently, so no source claims are upgraded automatically.
    provenance={
        'mode':'READ_ONLY_EVIDENCE_INVENTORY_NOT_CERTIFICATION',
        'date_utc':datetime.now(timezone.utc).isoformat(),
        'publisher_record_to_inspect':'https://zenodo.org/records/23037895',
        'publisher_note':'v1.3.2 core audio/registry/features/split described as byte-identical to v1.3.0; verify against publisher',
        'card_version_tokens':versions,'ledger_entries':len(ledger_entries),
        'ledger_sha256_unique_values':len({x['sha256'] for x in ledger_entries}),
        'ledger_issues':{'malformed':len(malformed),'duplicate_paths':len(duplicate_names)},
        'local_release_authentication':'NOT_VERIFIED',
        'original_download_archive_present':'NOT_CHECKED_NO_RECEIPT',
        'publisher_asset_hash_match':'NOT_VERIFIED',
        'public_external_data_post':'NOT_VERIFIED',
        'site_candidate_rows':len(stations),'site_candidate_columns':site_cols,
        'station_summary_rows':len(station_rows),
        'hydrophone_deployments':'NOT_VERIFIED',
        'natural_vessel_labels':'NOT_ESTABLISHED',
        'pressure_calibration':'NOT_ESTABLISHED',
        'prior_evidence_gates':gate_map,
        'source_sha256':{k:digest(p) for k,p in paths.items()},
        'prohibited_inputs':['Test.csv','exp004_eval','validation','sealed_holdout']}
    summary=['# Phase 5 — Provenance findings and external disclosure gate','','**Not a certification or final-submission approval.**','',
        f"- Local VTUAD card version references: {', '.join(versions) or '(none found)'}.",
        f"- Local strict-interval SHA-256 ledger: {len(ledger_entries)} parsed entries, {len(malformed)} malformed entries, {len(duplicate_names)} duplicated paths.",
        '- Exact local publisher release remains **unverified**. The checksum ledger authenticates neither the downloaded archive nor its distribution origin on its own.',
        '- Public external-data declaration remains **unverified**. Posting is a separate user action.',
        f'- Station source inventory: {len(station_rows)} acoustic summary rows; {len(stations)} unverified site-candidate metadata rows.',
        '- No exact hydrophone coordinates, natural-vessel detections, calibrated sound pressure or pressure maps are certified.',
        '','## Evidence needed to resolve gates','',
        '1. Original download receipts, asset names, archives and publisher-side checksums; exact dataset card version.',
        '2. Public Zindi external-data declaration URL and exact published source/version/license wording.',
        '3. Sensor/deployment IDs and recording periods tied to the exporter, plus authoritative coordinates, datum and uncertainty.',
        '4. Independently dated vessel observations/AIS or validated real-event annotations; otherwise retain descriptive-only claims.',
        '5. Instrument sensitivity, reference pressure and processing chain if calibrated pressure is to be claimed.',
        '','## Available publication wording','',
        'Publicly available VTUAD recordings reconstructed from Ocean Networks Canada archives were used as synthetic vessel-like intervention sources. The locally processed strict-interval WAV inventory contains 525 checksum-ledger entries. The exact acquired VTUAD distribution release remains under provenance verification; the external source is not a physically isolated vessel signal and includes environmental sound. No real South African vessel-event attribution is claimed based on the synthetic labels.',
        '','**Do not post this statement unchanged if the exact version can be verified from retained download records.**','']
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.phase5_provenance_',dir=out.parent) as staging:
        stage=Path(staging)
        (stage/'SOURCE_PROVENANCE_EVIDENCE.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
        (stage/'EVIDENCE_REPORT.md').write_text('\n'.join(summary))
        with (stage/'HYDROPHONE_DEPLOYMENT_VERIFICATION_TEMPLATE.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(review[0]));wr.writeheader();wr.writerows(review)
        with (stage/'VTUAD_LEDGER_FILENAME_INDEX.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['sha256','relative_filename']);wr.writeheader();wr.writerows(ledger_entries)
        (stage/'INPUT_HASHES.json').write_text(json.dumps(provenance['source_sha256'],indent=2,sort_keys=True)+'\n')
        stage.rename(out)
    print('PHASE 5 PROVENANCE: PASS (inventory only; unresolved claims remain unresolved)')
    print('Parsed WAV hashes:',len(ledger_entries),'| station summaries:',len(station_rows))
    print('Output:',out)

def self_test():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for p in [CARD, LEDGER,STATIONS,DEPLOYMENT,GATES,SUMMARY]:(root/p).parent.mkdir(parents=True,exist_ok=True)
        (root/CARD).write_text('VTUAD v1.3.0; prior v1.2.0')
        (root/LEDGER).write_text('a'*64+'  vessel/clip.wav\n')
        (root/STATIONS).write_text('station,latitude_candidate\nA,-32\n')
        (root/DEPLOYMENT).write_text('station,latitude\n')
        (root/GATES).write_text('gate,status\nEXACT_LOCAL_VTUAD_RELEASE,BLOCKED\n')
        (root/SUMMARY).write_text('station,sampled_runs\nA,12\n')
        o=root/'out';check(root,o)
        d=json.loads((o/'SOURCE_PROVENANCE_EVIDENCE.json').read_text())
        assert d['local_release_authentication']=='NOT_VERIFIED'
        assert d['ledger_entries']==1 and d['hydrophone_deployments']=='NOT_VERIFIED'
        try:check(root,o)
        except FileExistsError:pass
        else:raise AssertionError('Must refuse overwrite')
        (root/LEDGER).write_text('invalid hash\n')
        try:check(root,root/'out2')
        except ValueError:pass
        else:raise AssertionError('Must reject invalid checksum')
    print('PHASE 5 PROVENANCE SELF-TEST: PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only:self_test()
    else:
        root=a.root.resolve();out=a.output if a.output.is_absolute() else root/a.output
        check(root,out)
