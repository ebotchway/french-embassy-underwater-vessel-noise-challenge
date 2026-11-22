#!/usr/bin/env python3
"""Read-only, fail-closed research provenance and submission-readiness audit.

Only scans named project files; never opens competition Test.csv or sealed evaluation files.
Does not download from the Internet, verify deployment coordinates automatically, or post declarations.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, re, sys
from pathlib import Path

OUTPUT_DEFAULT = 'reports/audits/research_submission_readiness_v1'
FILES = {
    'local_vtuad_card':'data/external/vtuad/metadata/dataset_card.md',
    'vtuad_checksum_ledger':'data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt',
    'stations':'data/zindi/extracted/stations.csv',
    'station_dictionary':'data/zindi/extracted/DATA_DICTIONARY.md',
    'coordinates_template':'reports/research_manuscript_v1/STATION_COORDINATES_PROVENANCE_TEMPLATE.csv',
    'exp004_protocol':'reports/splits/experiment004_frozen_v1/split_protocol.json',
    'exp005':'reports/audits/simulation_realism_v1/scenario_summary.csv',
    'exp005_uncertainty':'reports/audits/simulation_realism_uncertainty_v1/comparison_summary.csv',
    'exp006':'reports/baselines/experiment006_internal_contrast_v1/development_summary.csv',
    'exp006b':'reports/audits/experiment006b_readonly_v1/overall_uncertainty.csv',
    'exp006c':'reports/audits/experiment006c_calibration_v1/RESEARCH_NOTE.md',
    'exp007':'reports/baselines/experiment007_source_disjoint_v1/assessment_policy_metrics.csv',
    'exp007_integrity':'reports/audits/experiment007_decision_gate_v1/RESEARCH_NOTE.md',
    'real_acoustics':'reports/audits/real_acoustic_development_v1/station_summary.csv',
    'manuscript':'reports/research_manuscript_v1/SCIENTIFIC_MANUSCRIPT_DRAFT.md',
    'manuscript_exp007':'reports/research_manuscript_v1/EXPERIMENT007_RESULTS_ADDENDUM.md',
    'external_declaration':'VTUAD_EXTERNAL_DATA_DECLARATION_DRAFT.md',
    'notebook':'notebooks/01_south_african_acoustic_eda.ipynb',
    'learning_log':'PROJECT_LEARNING_LOG.md',
    'checkpoint':'CHECKPOINT.md',
}

def digest(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def analyze(root:Path):
    inventory=[]
    for key,rel in FILES.items():
        path=root/rel; exists=path.is_file()
        inventory.append({'key':key,'path':rel,'exists':exists,'bytes':path.stat().st_size if exists else None,
                          'sha256':digest(path) if exists else None})
    findings=[]; coordinates=[]
    def record(id,status,evidence,action): findings.append({'id':id,'status':status,'evidence':evidence,'next_action':action})
    p=root/FILES['local_vtuad_card']
    if p.is_file():
        content=p.read_text(errors='replace')
        versions=sorted(set(re.findall(r'(?i)\bv?1\.\d+\.\d+(?:[-_ ]?candidate)?\b',content)))
        record('VTUAD_LOCAL_RELEASE','REVIEW_REQUIRED',f'Local dataset-card version strings: {versions or "none detected"}. File SHA-256 {digest(p)}',
               'Manually compare local dataset card, archive download ledger and Zenodo release assets; do not silently rename release.')
    else:record('VTUAD_LOCAL_RELEASE','MISSING','Local card not found','Locate authentic downloaded dataset card; do not infer version from challenge discussions.')
    p=root/FILES['vtuad_checksum_ledger']
    record('VTUAD_WAV_INTEGRITY','LEDGER_PRESENT' if p.is_file() else 'MISSING',
           'Checksum ledger recorded, not all audio rehashed' if p.is_file() else 'Checksum ledger not found',
           'Retain existing 525-file verification log or run a separate explicit WAV checksum audit.')
    p=root/FILES['stations']
    if p.is_file():
        with p.open(newline='',encoding='utf-8-sig') as f:
            rdr=csv.DictReader(f)
            required={'site_name','coordinate_status','latitude_candidate','longitude_candidate'}
            if not required.issubset(rdr.fieldnames or []):
                record('STATION_COORDINATES','BLOCKED','Station CSV lacks expected candidate/provenance headers', 'Inspect original archive schema.')
            else:
                for r in rdr:
                    coordinates.append({k:r.get(k,'') for k in ('site_name','latitude_candidate','longitude_candidate','latitude_alternative','longitude_alternative','coordinate_status')})
                status_set=sorted(set(r['coordinate_status'] for r in coordinates))
                record('STATION_COORDINATES','BLOCKED',f'{len(coordinates)} candidate sites; statuses: {status_set}',
                       'Verify hydrophone deployment coordinates, date, reference system, source document and sensor-to-export assignment.')
    else:record('STATION_COORDINATES','MISSING','Candidate station CSV absent','Locate original station metadata.')
    p=root/FILES['coordinates_template']
    if p.is_file():
        with p.open(newline='',encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        populated=[r for r in rows if r.get('latitude') and r.get('longitude') and r.get('source_url_or_document') and r.get('source_page_or_row')]
        record('DEPLOYMENT_PROVENANCE','REVIEW_REQUIRED',f'{len(populated)} rows contain coordinates and citation fields; not independently verified',
               'Check cited source and exact deployment match by hand; only then authorize map, separately from automatic readiness.')
    else:record('DEPLOYMENT_PROVENANCE','MISSING','Coordinates provenance template missing','Generate blank template; do not assume candidates are verified.')
    p=root/FILES['external_declaration']
    if p.is_file():
        s=p.read_text(errors='replace').lower()
        record('EXTERNAL_DATA_DECLARATION','DRAFT_ONLY' if ('draft' in s or 'not yet published' in s) else 'UNCONFIRMED',
               'A local declaration document is present; published forum URL not independently confirmed',
               'Verify local source version and attribution, publish allowed declaration, record exact public post URL.')
    else:record('EXTERNAL_DATA_DECLARATION','MISSING','Local declaration draft not found',
                'Prepare and publicly disclose version/source/licence/access as required by Zindi.')
    for key in ('exp004_protocol','exp005','exp005_uncertainty','exp006','exp006b','exp006c','exp007','exp007_integrity','real_acoustics','manuscript','manuscript_exp007','notebook','learning_log','checkpoint'):
        row=next(r for r in inventory if r['key']==key)
        record('FILE_'+key.upper(),'PRESENT' if row['exists'] else 'MISSING',row['path'],
               'Include cited source with frozen hash and archive or build reproducible evidence.' if row['exists'] else 'Locate or regenerate only from allowed development material.')
    record('REAL_VESSEL_LABELS','UNVERIFIED','No independent timestamped vessel annotations supplied in this audit',
           'Do not attribute acoustic excursions to confirmed vessel passages without independent AIS/port/sensor verification.')
    record('ACOUSTIC_CALIBRATION','UNVERIFIED','WAV relative digital power and archive acoustic levels not independently physically calibrated together',
           'Use relative-energy terminology, not absolute underwater SPL or source energy claims.')
    record('SUBMISSION_DEADLINE','PUBLIC_INFO_RECHECK','Zindi challenge research PDF and runnable code: 2026-12-13 11:30 UTC (checked 2026-11-22)',
           'Recheck official challenge page before submission.')
    return inventory,findings,coordinates

def emit(root:Path,output:Path):
    if output.exists():raise FileExistsError(f'Refusing existing output {output}')
    if output.resolve().is_relative_to((root/'data').resolve()):raise ValueError('Output must not be inside data directory')
    inventory,findings,coords=analyze(root)
    output.mkdir(parents=True,exist_ok=False)
    def write_csv(name, rows):
        if not rows:return
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_csv('input_inventory.csv',inventory)
    write_csv('submission_readiness_gates.csv',findings)
    write_csv('unverified_site_coordinate_candidates.csv',coords)
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'root':'project-relative',
              'input_files_found':sum(x['exists'] for x in inventory),'input_files_expected':len(inventory),
              'forbidden_inputs_not_opened':['Test.csv','original validation','sealed holdouts','exp004_eval'],
              'location_candidates_unverified':True,'published_external_declaration_not_verified':True}
    (output/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    blockers=[f for f in findings if f['status'] in ('BLOCKED','MISSING','UNVERIFIED','DRAFT_ONLY','REVIEW_REQUIRED','UNCONFIRMED')]
    lines=['# Research submission readiness: evidence-gated audit','',
           'This is an inventory/provenance audit, not confirmation of physics, publication, or legal compliance.', '',
           f'Available named inputs: {manifest["input_files_found"]}/{manifest["input_files_expected"]}.',
           f'Gates requiring attention: {len(blockers)}.', '', '## Open gates', '']
    for f in blockers:lines += [f'- **{f["id"]} ({f["status"]})**: {f["evidence"]}. Next: {f["next_action"]}']
    lines += ['', '## Scope', '', 'No hidden test, protected evaluation, image-based OCR, or web downloading was used by this audit.',
              'Candidate latitude/longitude fields are intentionally NOT emitted as verified monitoring sites.',
              'No geographic vessel-attribution map is authorized by this inventory.']
    (output/'READINESS_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('RESEARCH READINESS AUDIT: PASS (inventory only, unresolved gates retained)')
    print(f'Named inputs: {manifest["input_files_found"]}/{manifest["input_files_expected"]}; unresolved gates: {len(blockers)}')
    print('Output:',output)

def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);(root/'data/zindi/extracted').mkdir(parents=True)
        (root/'data/zindi/extracted/stations.csv').write_text('site_name,latitude_candidate,longitude_candidate,coordinate_status\nA,-1,20,working candidate\n')
        out=root/'reports/readiness'
        emit(root,out)
        assert 'BLOCKED' in (out/'submission_readiness_gates.csv').read_text()
        try:emit(root,out)
        except FileExistsError:pass
        else:raise AssertionError('Refusal to overwrite failed')
        assert not (root/'Test.csv').exists()
    print('SELF-TEST: PASS')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--output',default=OUTPUT_DEFAULT)
    ap.add_argument('--self-test-only',action='store_true');a=ap.parse_args()
    if a.self_test_only:self_test()
    else:emit(Path(a.root).resolve(),Path(a.output))
if __name__=='__main__':main()
