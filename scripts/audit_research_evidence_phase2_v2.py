#!/usr/bin/env python3
"""Evidence-led, read-only Phase 2 v2 audit of existing research materials."""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, re, tempfile, os, math
from pathlib import Path

BASE_OUT = 'reports/audits/research_evidence_phase2_v2'
INPUTS = {
 'vtuad_card':'data/external/vtuad/metadata/dataset_card.md',
 'vtuad_sha_ledger':'data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt',
 'vtuad_source_manifest':'reports/eda/vtuad_source_split_v2.csv',
 'vtuad_features':'reports/eda/vtuad_minute_band_power_split_v2.csv',
 'station_candidates':'data/zindi/extracted/stations.csv',
 'data_dictionary':'data/zindi/extracted/DATA_DICTIONARY.md',
 'station_coordinates_verified':'reports/research_manuscript_v1/STATION_COORDINATES_PROVENANCE_TEMPLATE.csv',
 'real_station_summary':'reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv',
 'real_spectral_summary':'reports/audits/spectral_shape_mismatch_v1/station_shape_difference_summary.csv',
 'exp007_assessment':'reports/baselines/experiment007_source_disjoint_v1/assessment_policy_metrics.csv',
 'exp007_decision':'reports/audits/experiment007_decision_gate_v1/RESEARCH_NOTE.md',
 'external_data_draft':'VTUAD_EXTERNAL_DATA_DECLARATION_DRAFT.md',
 'manuscript':'reports/research_manuscript_v1/SCIENTIFIC_MANUSCRIPT_DRAFT.md',
 'exp007_addendum':'reports/research_manuscript_v1/EXPERIMENT007_RESULTS_ADDENDUM.md',
}
EXTERNAL_RECORD = 'https://zenodo.org/records/23037895'
EXTERNAL_RULES = 'https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36266'

def sha256(path:Path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def rows_csv(path:Path, limit=100000):
 with path.open('r',newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames: raise ValueError(f'No CSV headers: {path}')
  rows=[]
  for i,row in enumerate(reader):
   if i>=limit:raise ValueError(f'Exceeded read cap ({limit}) at {path}')
   rows.append(row)
  return reader.fieldnames,rows

def write_csv(path:Path, headers:list[str], records:list[dict]):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,headers,extrasaction='ignore');w.writeheader();w.writerows(records)

def inventory(root):
 out=[]
 for label,rel in INPUTS.items():
  p=root/rel
  out.append(dict(label=label,relative_path=rel,present=p.is_file(),size_bytes=p.stat().st_size if p.is_file() else '',sha256=sha256(p) if p.is_file() else ''))
 return out

def safe_version_text(s):
 return sorted(set(m.group(0) for m in re.finditer(r'(?i)\bv?1\.[0-9]+\.[0-9]+(?:\s*[- ]?candidate)?\b',s)))

def provenance(root,entries):
 info={}
 card=root/INPUTS['vtuad_card']
 info['card_versions']=safe_version_text(card.read_text(encoding='utf-8',errors='replace')) if card.is_file() else []
 ledger=root/INPUTS['vtuad_sha_ledger']
 if ledger.is_file():
  lines=[x.strip() for x in ledger.read_text(encoding='utf-8',errors='replace').splitlines() if x.strip() and not x.startswith('#')]
  # Count only SHA256SUMS lines. Do not confuse it with verifying the file bytes.
  valid=[x for x in lines if re.match(r'^[a-fA-F0-9]{64}\s+\*?\S+',x)]
  info['sha256_ledger_entries']=len(valid)
  info['sha256_ledger_malformed']=len(lines)-len(valid)
 else:
  info['sha256_ledger_entries']=None
  info['sha256_ledger_malformed']=None
 info['release_status']='UNVERIFIED_LOCAL_RELEASE'
 info['external_record']=EXTERNAL_RECORD
 info['published_release_note']='Publisher describes v1.3.2 WAV/audio, registry, features and frozen split as byte-identical to v1.3.0; this does not prove local source identity.'
 return info

def candidates(root):
 p=root/INPUTS['station_candidates']
 if not p.is_file():return []
 headers,rows=rows_csv(p,1000)
 required={'site_name','coordinate_status','latitude_candidate','longitude_candidate'}
 if not required.issubset(headers): raise ValueError('stations.csv missing expected candidate headers')
 out=[]
 for row in rows:
  out.append({'station_name':row.get('site_name',''),'coordinate_status':row.get('coordinate_status',''),
   'candidate_latitude':row.get('latitude_candidate',''),'candidate_longitude':row.get('longitude_candidate',''),
   'alternative_latitude':row.get('latitude_alternative',''),'alternative_longitude':row.get('longitude_alternative',''),
   'source_path':INPUTS['station_candidates'],'deployment_verified':'NO',
   'evidence_needed':'recording/recorder-to-sensor ID; matching deployment dates; authoritative lat/lon with CRS; source locator and documented instrument location'})
 return out

def measured_station_evidence(root):
 p=root/INPUTS['real_station_summary']
 if not p.is_file():return None
 headers, rows=rows_csv(p,2000)
 return headers,rows

def validate_station_summary(source):
 """Validate actual source content; presence alone is not scientific evidence."""
 if source is None:
  return {'status':'SOURCE_MISSING','detail':'Expected descriptive station CSV is absent.', 'stations':0,'runs':0}
 headers,rows=source
 required={'station','sampled_runs','recorders','mean_observed_level',
           'median_observed_level','median_within_window_range',
           'median_temporal_band_variability'}
 missing=required-set(headers)
 if missing:
  return {'status':'SOURCE_INVALID','detail':f'Missing columns: {sorted(missing)}','stations':0,'runs':0}
 if not rows:
  return {'status':'SOURCE_INVALID','detail':'No station rows','stations':0,'runs':0}
 stations=set(); total=0
 for i,r in enumerate(rows,1):
  station=(r['station'] or '').strip()
  if not station or station in stations:
   return {'status':'SOURCE_INVALID','detail':f'Empty/duplicate station at row {i}: {station!r}','stations':len(stations),'runs':total}
  stations.add(station)
  try:
   n=float(r['sampled_runs']); recorders=float(r['recorders'])
   numbers=[float(r[x]) for x in ('mean_observed_level', 'median_observed_level',
        'median_within_window_range','median_temporal_band_variability')]
  except (ValueError,TypeError) as e:
   return {'status':'SOURCE_INVALID','detail':f'Invalid numerical value at row {i}: {e}','stations':len(stations),'runs':total}
  if not (n.is_integer() and n>0 and recorders.is_integer() and recorders>0 and all(math.isfinite(x) for x in [n, recorders]+numbers)):
   return {'status':'SOURCE_INVALID','detail':f'Nonfinite or invalid count at row {i}','stations':len(stations),'runs':total}
  if numbers[2]<0 or numbers[3]<0:
   return {'status':'SOURCE_INVALID','detail':f'Negative variability/range at row {i}','stations':len(stations),'runs':total}
  total+=int(n)
 return {'status':'SOURCE_PRESENT_SCHEMA_VALIDATED',
         'detail':f'{len(stations)} distinct station rows and {total} sampled-run entries; schema and basic numerical checks pass. Units, calibration, run provenance, and vessel attribution NOT verified.',
         'stations':len(stations),'runs':total}

def gates(info,coords,inv,station_validation):
 found={x['label']:bool(x['present']) for x in inv}
 g=[
  {'gate':'EXACT_LOCAL_VTUAD_RELEASE','status':'BLOCKED','reason':f"Local card version tokens: {info['card_versions']!r}; same clip count and ledger do not establish release identity.",'required_evidence':'Original downloaded archive file hashes and dataset-card release tag; reconcile Zenodo release assets.'},
  {'gate':'VTUAD_SOURCE_DECLARATION','status':'NOT_VERIFIED_PUBLIC','reason':'Local draft (if present) is not proof of an externally published declaration.','required_evidence':'Public discussion URL, exact version wording, source/license/access; independently inspect posted text.'},
  {'gate':'HYDROPHONE_DEPLOYMENTS','status':'BLOCKED','reason':f'{len(coords)} site-level location candidates, none certified by this script.','required_evidence':'Sensor ID and original deployment source matched to exporter and recording interval; coordinate CRS and uncertainty.'},
  {'gate':'VESSEL_ATTRIBUTION','status':'NOT_ESTABLISHED','reason':'Acoustic-level excursions cannot by themselves establish vessel passage.','required_evidence':'Time-aligned AIS/port observation or independent validated natural event annotation.'},
  {'gate':'PHYSICAL_PRESSURE_CALIBRATION','status':'NOT_ESTABLISHED','reason':'Relative VTUAD digital powers and archive dB measurements lack a validated common physical reference.','required_evidence':'Instrument sensitivity, calibration reference, processing chain and compatible acoustic units.'},
  {'gate':'REAL_ACOUSTIC_DESCRIPTIVE_EVIDENCE','status':station_validation['status'],'reason':station_validation['detail'],'required_evidence':'Check sampling/run coverage, timestamps, instrument effects and presentation caveats.'},
  {'gate':'ASSESSMENT_MODEL_CLAIMS','status':'SOURCE_PRESENT' if found['exp007_assessment'] else 'SOURCE_MISSING','reason':'Experiment 007 is source-disjoint internally but not an independent real-domain validation.','required_evidence':'Use frozen manifests; cite run-cluster uncertainty and no further threshold optimization.'},
  {'gate':'FINAL_RESEARCH_SUBMISSION','status':'NOT_READY','reason':'PDF plus runnable code and defensible spatial evidence still require verification.','required_evidence':'Final review with actual local data provenance and public external-data disclosure.'}
 ]
 return g

def report(info,inv,cands,station_columns,g,station_validation):
 lines=['# Research evidence phase 2 — provenance and claim audit','',
  'Scope: inventory and evidence grading only. **No coordinate or natural vessel inference is certified.**','',
  f"Named source files present: {sum(x['present'] for x in inv)}/{len(inv)}.",
  f"VTUAD card version strings: {', '.join(info['card_versions']) or 'none found'}.",
  f"VTUAD checksum ledger parsed entries: {info['sha256_ledger_entries']}; malformed lines: {info['sha256_ledger_malformed']}.",
  f"Unverified site candidates: {len(cands)}; alternative-position entries: {sum(bool(x['alternative_latitude'] and x['alternative_longitude']) for x in cands)}.",
  '','## Source verification decisions','',
  'The current publisher record describes the v1.3.2 supplementary additions as leaving WAV/audio, registry, feature library and frozen split byte-identical to v1.3.0. This is a **publisher release statement**, not proof which downloaded files were used locally.',
  f'Publisher record: {EXTERNAL_RECORD}',
  '', '## Evidence gates','']
 for a in g:
  lines += [f"- **{a['gate']} — {a['status']}**. {a['reason']} Needed: {a['required_evidence']}"]
 lines+=['','## Real acoustic presentation','',
 'The development station summaries may support *site-level descriptive distributions* after verifying their measurement convention and coverage. A non-geographic station plot is scientifically different from a georeferenced pressure/activity map.',
 f"Local descriptive station-summary columns: {station_columns if station_columns is not None else 'unavailable'}.",
 f"Station-source validation: {station_validation['detail']}",
 '','## Publication safeguards','',
 '- The candidate positions are never promoted to hydrophone locations automatically.',
 '- No geographically interpolated vessel heatmap is generated.',
 '- No hidden Zindi test, Experiment 004 evaluation, original validation or sealed holdout file is read.',
 '- No online post or external data declaration is created or submitted by this audit.',
 '- The researcher must independently verify the date-sensitive challenge requirements before final upload.',
 '']
 return '\n'.join(lines)

def emit(root:Path,out:Path):
 root=root.resolve();out=out.resolve()
 if out.exists():raise FileExistsError(f'Output exists; will not overwrite: {out}')
 if out.is_relative_to((root/'data').resolve()):raise ValueError('Do not place output inside source data directory')
 inv=inventory(root);info=provenance(root,inv);cands=candidates(root)
 station=measured_station_evidence(root)
 st_cols=station[0] if station else None
 validation=validate_station_summary(station)
 g=gates(info,cands,inv,validation)
 out.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.research_phase2_',dir=str(out.parent)) as staging:
  tmp=Path(staging)
  write_csv(tmp/'file_inventory.csv',['label','relative_path','present','size_bytes','sha256'],inv)
  write_csv(tmp/'unverified_location_registry.csv',['station_name','coordinate_status','candidate_latitude','candidate_longitude','alternative_latitude','alternative_longitude','source_path','deployment_verified','evidence_needed'],cands)
  write_csv(tmp/'claim_evidence_gates.csv',['gate','status','reason','required_evidence'],g)
  (tmp/'local_vtuad_version_evidence.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
  (tmp/'RESEARCH_EVIDENCE_REPORT.md').write_text(report(info,inv,cands,st_cols,g,validation))
  manifest={'audit':'research_evidence_phase2_v2','timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
   'input_hashes':{x['relative_path']:x['sha256'] for x in inv if x['present']},
   'prohibited_inputs':['Test.csv','exp004_eval','original validation','sealed station holdouts'],
   'station_summary_validation':validation,
   'declared_limitations':['no physical calibration verification','no exact hydrophone coordinate verification','no public declaration verification','no real vessel labels']}
  (tmp/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  # Atomic destination publication. Existing directory is always an error, preserving prior results.
  os.rename(tmp,out)
 print('RESEARCH EVIDENCE PHASE 2: PASS (inventory/claims, NOT submission approval)')
 print(f"Files: {sum(x['present'] for x in inv)}/{len(inv)}; coordinate candidates: {len(cands)}; exact release: UNVERIFIED")
 print('Real acoustic descriptive evidence:',validation['status'],f"({validation['stations']} stations; {validation['runs']} run entries)")
 print('Output:',out)

def self_test():
 with tempfile.TemporaryDirectory() as t:
  r=Path(t)
  (r/'data/zindi/extracted').mkdir(parents=True)
  (r/'data/external/vtuad/metadata').mkdir(parents=True)
  (r/'data/zindi/extracted/stations.csv').write_text('site_name,coordinate_status,latitude_candidate,longitude_candidate,latitude_alternative,longitude_alternative\nSaldanha,unresolved,-33,18,-34,19\n')
  (r/'data/external/vtuad/metadata/dataset_card.md').write_text('VTUAD v1.3.0 candidate\n')
  (r/'data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt').write_text('a'*64+'  sample.wav\n')
  real=r/'reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv'
  real.parent.mkdir(parents=True)
  real.write_text('station,sampled_runs,recorders,mean_observed_level,median_observed_level,median_within_window_range,median_temporal_band_variability\nA,120,2,63.1,63.2,4.0,2.1\nB,22,1,90.0,89.0,5.0,1.9\n')
  out=r/'reports/audits/phase2'
  emit(r,out)
  assert len(rows_csv(out/'unverified_location_registry.csv')[1])==1
  assert 'UNVERIFIED_LOCAL_RELEASE' in (out/'local_vtuad_version_evidence.json').read_text()
  assert 'NOT_ESTABLISHED' in (out/'claim_evidence_gates.csv').read_text()
  assert 'SOURCE_PRESENT_SCHEMA_VALIDATED' in (out/'claim_evidence_gates.csv').read_text()
  real.write_text('station,sampled_runs,recorders,mean_observed_level,median_observed_level,median_within_window_range,median_temporal_band_variability\nA,missing,2,63.1,63.2,4.0,2.1\n')
  emit(r,r/'reports/audits/phase2_invalid')
  assert 'SOURCE_INVALID' in (r/'reports/audits/phase2_invalid/claim_evidence_gates.csv').read_text()
  assert 'deployment_verified' in (out/'unverified_location_registry.csv').read_text()
  try:emit(r,out)
  except FileExistsError:pass
  else: raise AssertionError('Overwrite not blocked')
 print('RESEARCH EVIDENCE PHASE 2 SELF-TEST: PASS')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',default='.')
 p.add_argument('--output',default=BASE_OUT)
 p.add_argument('--self-test-only',action='store_true')
 a=p.parse_args()
 if a.self_test_only:self_test()
 else:emit(Path(a.root),Path(a.output))
if __name__=='__main__':main()
