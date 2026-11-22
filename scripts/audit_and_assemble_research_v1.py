#!/usr/bin/env python3
"""Read-only research evidence reconciliation and evidence-bounded draft assembly.

Runs only over existing research summaries. Does not fetch URLs, inspect Zindi test,
train models, create pressure maps, or publish disclosures.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, tempfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve()
APPENDIX = Path('reports/research_manuscript_v1/real_acoustic_results_v2')
METRICS = Path('reports/baselines/experiment007_source_disjoint_v1/assessment_policy_metrics.csv')
GATES = Path('reports/audits/research_evidence_phase2_v2/claim_evidence_gates.csv')
RELEASE_CARD = Path('data/external/vtuad/metadata/dataset_card.md')
LEDGER = Path('data/external/vtuad/metadata/strict_interval_wav_SHA256SUMS.txt')
DRAFT = Path('reports/research_manuscript_v1/SCIENTIFIC_MANUSCRIPT_DRAFT.md')
ADDENDUM = Path('reports/research_manuscript_v1/EXPERIMENT007_RESULTS_ADDENDUM.md')
OUTPUT = Path('reports/research_manuscript_v1/submission_evidence_phase4_v1')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def check_local(root):
    needed={'acoustic_manifest':APPENDIX/'manifest.json',
       'acoustic_appendix':APPENDIX/'REAL_ACOUSTIC_RESULTS_APPENDIX.md',
       'acoustic_pdf':APPENDIX/'REAL_ACOUSTIC_RESULTS_APPENDIX.pdf',
       'acoustic_station_summary':APPENDIX/'validated_station_summary.csv',
       'acoustic_bootstrap':APPENDIX/'run_bootstrap_intervals.csv',
       'experiment007_metrics':METRICS,'evidence_gates':GATES,
       'vtuad_card':RELEASE_CARD,'vtuad_ledger':LEDGER,
       'manuscript':DRAFT,'experiment007_addendum':ADDENDUM}
    paths={k:root/v for k,v in needed.items()}
    missing=[k for k,p in paths.items() if not p.is_file()]
    if missing: raise FileNotFoundError('Missing inputs: '+', '.join(f'{k} ({needed[k]})' for k in missing))
    manifest=json.loads(paths['acoustic_manifest'].read_text())
    for rel,expected in manifest.get('outputs_sha256',{}).items():
        fp=root/APPENDIX/rel
        if not fp.is_file() or sha(fp)!=expected: raise ValueError(f'Acoustic output mismatch: {fp}')
    # Validate source inputs against the manifest when present, not just outputs.
    for key,expected in manifest.get('source_sha256',{}).items():
        rp=manifest.get('source_paths',{}).get(key)
        if rp is None: raise ValueError(f'Missing acoustic source-path provenance for {key}')
        fp=root/rp
        if not fp.is_file() or sha(fp)!=expected: raise ValueError(f'Acoustic input mismatch: {fp}')
    gates={r['gate']:r['status'] for r in rows(paths['evidence_gates'])}
    if gates.get('REAL_ACOUSTIC_DESCRIPTIVE_EVIDENCE') != 'SOURCE_PRESENT_SCHEMA_VALIDATED':
        raise ValueError('Real acoustic evidence gate is not schema validated')
    st=rows(paths['acoustic_station_summary'])
    if len(st)!=6 or sum(int(float(r['sampled_runs'])) for r in st)!=460:
        raise ValueError('Six-station/460 sampled-run evidence count mismatch')
    mm=rows(paths['experiment007_metrics'])
    # Verify uniquely reported policies; do not select or tune based on assessment.
    key={(r['policy'],r['threshold_policy']):r for r in mm}
    for pair in [('original_only','.5_fixed'),('contrast_varied','calibration_selected'),('hybrid','.5_fixed')]:
        if pair not in key: raise ValueError(f'Missing Experiment007 policy {pair}')
    if len(mm)<6: raise ValueError('Incomplete Experiment007 policy metrics')
    card=paths['vtuad_card'].read_text(errors='replace')
    version_tokens=sorted(set(re.findall(r'(?i)\bv?1\.[0-9]+\.[0-9]+\b',card)))
    # Integrity of a local ledger is distinct from authenticity against publisher assets.
    with paths['vtuad_ledger'].open(errors='replace') as f: ledger_lines=[l.strip() for l in f if l.strip() and not l.lstrip().startswith('#')]
    hashes=[re.search(r'\b[0-9a-fA-F]{64}\b',line) for line in ledger_lines]
    ledger_ok=bool(hashes) and all(hashes)
    if not ledger_ok: raise ValueError('Malformed local source checksum ledger')
    return paths,manifest,gates,st,key,dict(version_tokens=version_tokens,ledger_entries=len(hashes),
                  release_identity='NOT INDEPENDENTLY VERIFIED',published_declaration='NOT INDEPENDENTLY VERIFIED')


def summarize(gates,st,key,release):
    def fmt(pair,col): return f"{float(key[pair][col]):.6f}"
    s=['# Evidence-gated research integration — working submission manuscript',
       '', '**DRAFT ONLY — not cleared for submission. No instrument-calibrated pressure map or natural vessel validation is claimed.**',
       '', '## Data provenance and research status',
       '',f"Existing training-only South African station summaries cover **{len(st)} stations and 460 indexed sampled runs** across 25 challenge bands. Run identifiers are not guaranteed independent deployments.",
       '', 'The independently checked acoustic appendix uses descriptive archive-reported levels and relative band-shape statistics. It does not demonstrate vessel identities, acoustic source separation, or absolute acoustic pressure.',
       '', f"The local VTUAD card includes version tokens {', '.join(release['version_tokens']) or '(none)'}; the local checksum ledger parses {release['ledger_entries']} entries. Local release identity is **unverified** until upstream release assets are matched to original acquisition records. WAV checksums alone do not authenticate the dataset release.",
       '', '## Internal synthetic Experiment 007 — fixed, previously assessed policies',
       '', '| Policy | Fraction RMSE | Detector AUC | Sensitivity | Genuine background FPR |',
       '|---|---:|---:|---:|---:|']
    for pair,label in [(('original_only','.5_fixed'),'Original, 0.50'),(('contrast_varied','.5_fixed'),'Contrast-varied, 0.50'),(('contrast_varied','calibration_selected'),'Contrast-varied, calibration-selected'),(('hybrid','.5_fixed'),'Hybrid, 0.50')]:
        s.append('| '+label+' | '+' | '.join(fmt(pair,c) for c in ['rmse','auc','tpr_observed_positive','fpr_no_addition'])+' |')
    s+=['','These numbers are from an internally source-disjoint *synthetic* assessment. They are not natural-vessel detection accuracy. The assessment must not be used for additional threshold or hyperparameter tuning.',
         '', 'The source-group run-bootstrap interval for the RMSE reduction was reported as −0.00041 to 0.04230 in one fixed-threshold comparison; it includes zero. The observed point-estimate reduction is not a guaranteed external-domain improvement.',
         '', '## Verified descriptive acoustic evidence',
         '', '| Station | Sampled runs | Recorders | Reported median level |',
         '|---|---:|---:|---:|']
    for r in st:
        s.append(f"| {r['station']} | {int(float(r['sampled_runs']))} | {r['recorders']} | {r['median_observed_level']} |")
    s+=['','The level column is the archive-reported numerical convention, **not independently verified calibrated underwater dB re 1 μPa**. Durban has one sampled run and no reliable station-level bootstrap interval or robust z-score.',
         '', '## Open scientific and competition gates', '']
    for k in ['EXACT_LOCAL_VTUAD_RELEASE','VTUAD_SOURCE_DECLARATION','HYDROPHONE_DEPLOYMENTS','VESSEL_ATTRIBUTION','PHYSICAL_PRESSURE_CALIBRATION','FINAL_RESEARCH_SUBMISSION']:
        s.append(f'- **{k}: {gates.get(k,"NOT FOUND")}**')
    s+=['', 'The station coordinates are working site-level candidates, not sensor-deployment coordinates. No interpolation or hydrophone activity heatmap is generated. A vessel-related spatial claim requires matching instrument and time metadata and independent vessel activity records.',
       '', '## Reproducibility and limitations',
       '', 'This integration verifies input and figure artifact hashes, copies the pre-existing acoustic results, and uses frozen Experiment 007 assessment rows only for reporting. It does not read the Zindi hidden test, recompute model decisions, modify the notebook, or silently publish an external-data declaration.',
       '', '### Sources to verify and cite in the final submission',
       '', '- Zindi competition: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge',
       '- Official data page: https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data',
       '- VTUAD Zenodo record: https://zenodo.org/records/23037895',
       '- Ocean Networks Canada: https://data.oceannetworks.ca/',
       '', '## Existing detailed manuscript, Experiment 007 addendum and acoustic appendix',
       '', 'The complete prior manuscript and actual-data acoustic appendix are packaged separately below; no model-result figures are reconstructed from unavailable raw data.',
       '']
    return '\n'.join(s)+'\n'


def render_pdf(note, images, out):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    st=getSampleStyleSheet();st.add(ParagraphStyle(name='TinyBody',parent=st['BodyText'],fontSize=8.8,leading=12.4,spaceAfter=6))
    story=[]
    for line in note.splitlines():
        if not line.strip(): continue
        if line.startswith('# '): style=st['Title'];line=line[2:]
        elif line.startswith('## '): style=st['Heading2'];line=line[3:]
        elif line.startswith('### '):style=st['Heading3'];line=line[4:]
        elif line.startswith('|---'):continue
        elif line.startswith('|'):style=st['TinyBody'];line=line.strip('| ').replace('|','  ·  ')
        else:style=st['TinyBody'];line=line.replace('**','').replace('`','')
        story.append(Paragraph(escape(line),style))
    story.append(PageBreak());story.append(Paragraph('Measured archive figures — descriptive only',st['Heading1']))
    for img in images:
        story.append(Paragraph(escape(img.stem.replace('_',' ')),st['Heading3']))
        el=Image(str(img));scale=min(470/el.drawWidth,320/el.drawHeight,1)
        el.drawWidth*=scale;el.drawHeight*=scale;story.append(el);story.append(Spacer(1,15))
    SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48,
                      title='Evidence-gated underwater acoustic research working report').build(story)


def build(root,out):
    if out.exists(): raise FileExistsError(f'Output exists; preserve it: {out}')
    paths,manifest,gates,stations,metrics,release=check_local(root)
    note=summarize(gates,stations,metrics,release)
    figures=sorted((root/APPENDIX/'figures').glob('*.png'))
    if len(figures)!=5: raise ValueError(f'Expected five prior verified acoustic figures; found {len(figures)}')
    for fig in figures:
        rel=str(fig.relative_to(root/APPENDIX))
        if manifest['outputs_sha256'].get(rel)!=sha(fig):raise ValueError(f'Unverified figure: {fig}')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.research_phase4_stage_',dir=out.parent) as d:
        stage=Path(d);(stage/'figures').mkdir()
        (stage/'RESEARCH_INTEGRATION_WORKING.md').write_text(note)
        (stage/'EXISTING_MANUSCRIPT.md').write_bytes(paths['manuscript'].read_bytes())
        (stage/'EXPERIMENT007_RESULTS_ADDENDUM.md').write_bytes(paths['experiment007_addendum'].read_bytes())
        (stage/'REAL_ACOUSTIC_RESULTS_APPENDIX.md').write_bytes(paths['acoustic_appendix'].read_bytes())
        for p in figures:(stage/'figures'/p.name).write_bytes(p.read_bytes())
        render_pdf(note,sorted((stage/'figures').glob('*.png')),stage/'RESEARCH_INTEGRATION_WORKING.pdf')
        inventory={'status':'DRAFT_ONLY','source_sha256':{k:sha(v) for k,v in paths.items()},
                   'acoustic_input_source_hashes':manifest.get('source_sha256',{}),'evidence_gates':gates,
                   'vtuad_version_evidence':release,'source_figures_sha256':{p.name:sha(p) for p in figures},
                   'output_sha256':{str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()},
                   'prohibited_claims':['verified_sensor_locations','absolute_calibrated_pressure','natural_vessel_labels','external_accuracy']}
        (stage/'AUDIT_MANIFEST.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
        stage.rename(out)
    print('RESEARCH INTEGRATION: PASS — draft evidence assembly, NOT submission approval')
    print(f'Output: {out}\nStation summaries: 6; run entries: 460; verified figures: 5')
    print('VTUAD exact release: UNVERIFIED; deployment and real-vessel attribution: UNVERIFIED')


def selftest():
    # smoke-test pure rendering and scoring without impersonating real project evidence
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d);s={x:'BLOCKED' for x in ['EXACT_LOCAL_VTUAD_RELEASE','HYDROPHONE_DEPLOYMENTS']}
        key={}
        for x,y in [('original_only','.5_fixed'),('contrast_varied','.5_fixed'),('contrast_varied','calibration_selected'),('hybrid','.5_fixed')]:
            key[(x,y)]={k:'0.5' for k in ['rmse','auc','tpr_observed_positive','fpr_no_addition']}
        st=[{'station':f's{i}','sampled_runs':'1','recorders':'1','median_observed_level':'60'} for i in range(6)]
        note=summarize(s,st,key,{'version_tokens':['v1.3.0'],'ledger_entries':525})
        assert 'not independently verified' in note.lower() or 'unverified' in note.lower()
        render_pdf(note,[],folder/'test.pdf')
        assert (folder/'test.pdf').read_bytes().startswith(b'%PDF')
    print('RESEARCH INTEGRATION SELF-TEST: PASS (synthetic text/PDF smoke test only)')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only:selftest()
    else:build(a.root.resolve(),a.output if a.output.is_absolute() else a.root.resolve()/a.output)
