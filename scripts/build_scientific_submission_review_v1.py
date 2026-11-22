#!/usr/bin/env python3
"""Consolidate existing verified research reports into a readable review draft.

No retraining, use of protected evaluations, coordinate inference, or automatic certification.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,tempfile,shutil
from pathlib import Path
from xml.sax.saxutils import escape

ROOT_INPUT=Path('reports/research_manuscript_v1/submission_evidence_phase4_v1')
ACOUSTIC=Path('reports/research_manuscript_v1/real_acoustic_results_v2')
RESULTS=Path('reports/baselines/experiment007_source_disjoint_v1/assessment_policy_metrics.csv')
PROVENANCE=Path('reports/audits/submission_provenance_phase5_v1/SOURCE_PROVENANCE_EVIDENCE.json')
DEST=Path('reports/research_manuscript_v1/submission_review_phase5_v1')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def collect(root):
    needed={
        'phase4_manifest':ROOT_INPUT/'AUDIT_MANIFEST.json',
        'phase4_draft':ROOT_INPUT/'RESEARCH_INTEGRATION_WORKING.md',
        'original_manuscript':ROOT_INPUT/'EXISTING_MANUSCRIPT.md',
        'exp007_addendum':ROOT_INPUT/'EXPERIMENT007_RESULTS_ADDENDUM.md',
        'acoustic_appendix':ROOT_INPUT/'REAL_ACOUSTIC_RESULTS_APPENDIX.md',
        'acoustic_station':ACOUSTIC/'validated_station_summary.csv',
        'acoustic_bootstrap':ACOUSTIC/'run_bootstrap_intervals.csv',
        'exp007_policy_metrics':RESULTS,
        'phase5_provenance':PROVENANCE}
    missing=[f'{k}:{p}' for k,p in needed.items() if not (root/p).is_file()]
    if missing:raise FileNotFoundError('Required source artifact(s) missing: '+'; '.join(missing))
    paths={k:root/p for k,p in needed.items()}
    phase4=json.loads(paths['phase4_manifest'].read_text())
    if phase4.get('status')!='DRAFT_ONLY':raise ValueError('Unexpected phase4 evidence status')
    # Recheck each item used from the assembled report against its own manifest.
    for rel,want in phase4['output_sha256'].items():
        p=root/ROOT_INPUT/rel
        if not p.is_file() or digest(p)!=want:raise ValueError('Phase4 assembly hash mismatch: '+str(p))
    gate=phase4['evidence_gates']
    if gate.get('REAL_ACOUSTIC_DESCRIPTIVE_EVIDENCE')!='SOURCE_PRESENT_SCHEMA_VALIDATED':
        raise ValueError('Acoustic evidence not source-schema validated')
    if gate.get('EXACT_LOCAL_VTUAD_RELEASE') not in ('BLOCKED','NOT_VERIFIED','UNVERIFIED'):
        raise ValueError('Unexpected release status; needs explicit manual review')
    if gate.get('VESSEL_ATTRIBUTION') not in ('NOT_ESTABLISHED','BLOCKED'):
        raise ValueError('Unexpected attribution status; needs explicit manual review')
    prov=json.loads(paths['phase5_provenance'].read_text())
    if prov.get('local_release_authentication')!='NOT_VERIFIED':raise ValueError('Release status needs new review')
    st=rows(paths['acoustic_station']);metrics=rows(paths['exp007_policy_metrics'])
    if len(st)!=6 or sum(int(float(x['sampled_runs'])) for x in st)!=460:
        raise ValueError('Expected validated six-station 460-run sample')
    key={(r['policy'],r['threshold_policy']):r for r in metrics}
    policies=[(('original_only','.5_fixed'),'Original detector and regressor · fixed 0.50'),
              (('contrast_varied','.5_fixed'),'Contrast-varied · fixed 0.50'),
              (('contrast_varied','calibration_selected'),'Contrast-varied · calibration-selected'),
              (('hybrid','.5_fixed'),'Hybrid · fixed 0.50')]
    if any(i not in key for i,_ in policies):raise ValueError('Required frozen assessment policies missing')
    figures=sorted((root/ROOT_INPUT/'figures').glob('*.png'))
    if len(figures)!=5:raise ValueError(f'Expected 5 verified acoustic figures, got {len(figures)}')
    for fig in figures:
        rel=str(fig.relative_to(root/ROOT_INPUT))
        if phase4['output_sha256'].get(rel)!=digest(fig):raise ValueError('Figure hash not preserved: '+rel)
    return paths,gate,prov,st,key,policies,figures

def compose(gates,prov,stations,metric,policies,figures):
    out=[
    '# Underwater acoustic monitoring and source-disjoint synthetic vessel-noise modelling in South Africa',
    '', '**Research manuscript — review draft, not cleared for competition submission**',
    '', '## Abstract',
    'We evaluate a controlled synthetic vessel-addition task against descriptive underwater acoustic archive measurements from six named South African monitoring sites. A training-only development sample contains 460 indexed recording runs and 25 third-octave frequency bands. Acoustic differences across sites are described without assigning natural vessel-event labels or claiming calibrated sound pressure. A source-disjoint internal assessment of 120 synthetic paired scenarios compares original-source and contrast-varied models. The contrast-varied fraction estimate gives RMSE 0.094566 versus 0.114236 for the original-only estimator, while the fixed-threshold contrast-varied detector increases genuine background false positives. These are internal synthetic results, not evidence of validated vessel detection or external transfer.',
    '', '## 1. Research question and data separation',
    'The challenge requests an estimate of an added vessel acoustic-energy fraction and a detection score from 15-minute, 25-band records. This manuscript distinguishes three domains: (i) observed South African archive statistics, (ii) external vessel-associated VTUAD clips, and (iii) synthetic mixtures with known injected energy ratios. These are not interchangeable sources of ground truth.',
    '',f"The local VTUAD checksum manifest contains **{prov['ledger_entries']} entries**, but the exact locally acquired release is **not authenticated**. Published versions and publisher statements require independent reconciliation with the original download artifacts. VTUAD clips contain ambient acoustic energy; they must not be called pure isolated ship signals.",
    '', '## 2. Methods and reproducibility',
    'The existing research pipeline constructs fixed 15-minute multiband windows, protects its sequestered Experiment 004 evaluation and the Zindi hidden test, and trains on allocated development sources. Relative-energy mixing scales external clip band powers toward a requested injected-energy proportion; it is not a calibrated acoustic-pressure mixing procedure. Missing observation masks can make the realized observed fraction differ from the requested fraction.',
    'Experiment 007 allocates recording runs and vessel hull/clip groups to train, calibration and assessment before generating mixtures. Detector thresholds are selected only on calibration data with a calibration background false-positive ceiling of 25%. The single internal assessment is not reused for model tuning.',
    '', '## 3. Real South African acoustic archive: descriptive evidence',
    'Six station summary rows represent **460 sampled run entries** across 25 challenge bands. These units are indexed runs rather than independently authenticated deployment campaigns. Site comparisons may reflect recorder settings, local background and sampling differences. Reported numeric levels retain the archive convention; they are **not** independently established as absolute dB re 1 μPa.',
    '', '| Site | Sampled runs | Recorders | Reported median archive level |',
    '|---|---:|---:|---:|']
    for r in stations:
        out.append('| '+str(r['station'])+' | '+str(int(float(r['sampled_runs'])))+' | '+str(r['recorders'])+' | '+f"{float(r['median_observed_level']):.2f}"+' |')
    out += [
    '', 'Durban contributes only one sampled run; it cannot support a reliable within-station run-bootstrap uncertainty interval. The single undefined robust z-score is deliberately not imputed. Station-level figures describe distributions and spectral shape differences, not identified natural-vessel passages.',
    '', '## 4. Internal synthetic fraction and detection results',
    '', '| Prespecified policy | RMSE (lower better) | ROC AUC | Observed-positive sensitivity | Genuine background FPR |',
    '|---|---:|---:|---:|---:|']
    for k,label in policies:
        r=metric[k]
        out.append('| '+label+' | '+' | '.join(f"{float(r[col]):.4f}" for col in ['rmse','auc','tpr_observed_positive','fpr_no_addition'])+' |')
    out += [
    '', 'The varied regressor improves the observed RMSE point estimate by about 17.2%, but a conditional run-bootstrap confidence interval for the fixed-threshold comparison (−0.00041 to 0.04230) includes zero. The fixed 0.50 detector comparison improves sensitivity but increases false positives; the calibration-selected varied threshold trades some sensitivity for fewer false alarms. The hybrid combines the varied regressor and original detector and does not constitute a separately fitted independent model.',
    '', 'These detector labels concern *synthetically injected vessel-associated energy*. They must not be equated with naturally present vessels, calibrated physical energy fractions, or instrument-independent real-world false-alarm rates.',
    '', '## 5. Figures and station-level uncertainty',
    'The five figures below are incorporated unchanged from the verified acoustic-results appendix. Their visual content is descriptive. No interpolated georeferenced pressure map is presented because exact hydrophone-deployment coordinates and pressure calibration are not established.',
    '']
    for i,fig in enumerate(figures,1):
        label=fig.stem.replace('_',' ').capitalize()
        out.extend([f'**Figure {i}. {label}.** Descriptive archive measurement or relative spectral statistic; not a vessel-attribution or calibrated-pressure plot.',f'![Figure {i}](figures/{fig.name})',''])
    out += [
    '## 6. Spatial interpretation and research limitations',
    'Station names and candidate geographic coordinates identify monitoring-site areas, not verified hydrophone deployment positions. In particular, multiple candidate coordinates may exist for a site. It would be misleading to interpolate an acoustic pressure surface or assert vessel traffic from these candidates. A future qualified georeferenced assessment requires sensor identifier, dated deployment record, location datum, coordinate uncertainty and a compatible calibrated measurement convention. Independently timestamped AIS/port observations or validated annotations would be required to attribute changes to vessels.',
    '', '## 7. Data, code and responsible reproducibility',
    'The authoritative scripts, frozen split manifests, input SHA-256 records and research learning log are retained in the local project. This review manuscript is assembled from validated local results and sealed assessment summaries, not from new tuning. Protected competition test labels are neither available nor inferred. The provenance evidence auditor does not publish any external-data declaration.',
    '', '## 8. Evidence gates and submission readiness',
    '']
    for k in ['EXACT_LOCAL_VTUAD_RELEASE','VTUAD_SOURCE_DECLARATION','HYDROPHONE_DEPLOYMENTS','VESSEL_ATTRIBUTION','PHYSICAL_PRESSURE_CALIBRATION','FINAL_RESEARCH_SUBMISSION']:
        out.append(f'- **{k}: {gates.get(k,"NOT_RECORDED")}**')
    out += ['', 'Scientific interpretation remains conditional on these unresolved gates. A completed descriptive and synthetic paper is not equivalent to the competition-requested validated vessel-related pressure map.',
    '', '## References and records for independent review',
    '1. Zindi, *Underwater Vessel Noise Challenge by the French Embassy in South Africa*. https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge',
    '2. Zindi, *Challenge Data and Research Write-up Requirements*. https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data',
    '3. VTUAD publisher record, Zenodo. https://zenodo.org/records/23037895',
    '4. Ocean Networks Canada data portal. https://data.oceannetworks.ca/',
    '', '## Supporting traceability artifacts',
    'The full previous scientific manuscript draft, Experiment 007 addendum, and separately verified real-acoustic appendix remain available as unchanged supporting research artifacts. This manuscript integrates their defensible summary rather than replacing them or asserting that all evidence gates have passed.',
    '']
    return '\n'.join(out)

def pdf(text,figures,path):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,KeepTogether,Table,TableStyle
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ResearchBody',parent=styles['BodyText'],fontName='Helvetica',fontSize=9.1,leading=13.6,spaceAfter=7))
    styles.add(ParagraphStyle(name='CaptionResearch',parent=styles['BodyText'],fontSize=8,leading=11,spaceAfter=7,textColor=colors.HexColor('#475569')))
    styles.add(ParagraphStyle(name='TableResearch',parent=styles['BodyText'],fontSize=7.25,leading=9.6,spaceAfter=0))
    styles['Title'].fontSize=15;styles['Title'].leading=19;styles['Heading2'].spaceBefore=13
    story=[];fig_by_name={f.name:f for f in figures}
    lines=text.splitlines(); i=0
    while i<len(lines):
        t=lines[i].strip();i+=1
        if not t:continue
        if t.startswith('|'):
            table=[]
            while True:
                if not t.startswith('|'):break
                if not t.startswith('|---'):
                    table.append([Paragraph(escape(x.strip()),styles['TableResearch']) for x in t.strip('|').split('|')])
                if i>=len(lines):break
                t=lines[i].strip()
                if not t.startswith('|'):break
                i+=1
            if table:
                cols=len(table[0]);available=174*mm
                item=Table(table,colWidths=[available/cols]*cols,repeatRows=1,hAlign='LEFT')
                item.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf3')),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')]),
                    ('LINEBELOW',(0,0),(-1,0),0.6,colors.HexColor('#94a3b8')),
                    ('BOTTOMPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),5),
                    ('LEFTPADDING',(0,0),(-1,-1),5),
                    ('RIGHTPADDING',(0,0),(-1,-1),5)]))
                story.append(item);story.append(Spacer(1,9))
            continue
        if t.startswith('**Figure '):
            caption=t.strip('*')
            nxt=lines[i].strip() if i<len(lines) else ''
            if nxt.startswith('![') and nxt.endswith(')'):
                i+=1
                filename=nxt.rsplit('/',1)[-1][:-1]
                file=fig_by_name.get(filename)
                if file:
                    im=Image(str(file));scale=min(166*mm/im.drawWidth, 84*mm/im.drawHeight,1)
                    im.drawWidth*=scale;im.drawHeight*=scale
                    story.append(KeepTogether([Paragraph(escape(caption),styles['CaptionResearch']),im,Spacer(1,9)]))
                    continue
            story.append(Paragraph(escape(caption),styles['CaptionResearch']))
            continue
        if t.startswith('!['):continue
        if t.startswith('# '):style=styles['Title'];t=t[2:]
        elif t.startswith('## '):style=styles['Heading2'];t=t[3:]
        else:style=styles['ResearchBody'];t=t.replace('**','').replace('`','')
        story.append(Paragraph(escape(t),style))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
        canvas.drawString(18*mm,12*mm,'Working research draft — not submission-cleared')
        canvas.drawRightString(192*mm,12*mm,f'Page {doc.page}')
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=19*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=20*mm,
        title='Underwater acoustic monitoring and synthetic vessel noise modelling - scientific review').build(story,onFirstPage=footer,onLaterPages=footer)

def build(root,out):
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    sources,gates,prov,stations,metrics,policies,figures=collect(root)
    text=compose(gates,prov,stations,metrics,policies,figures)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.submission_review_phase5_',dir=out.parent) as tmp:
        s=Path(tmp);(s/'figures').mkdir()
        (s/'RESEARCH_MANUSCRIPT_REVIEW.md').write_text(text)
        for f in figures:shutil.copy2(f,s/'figures'/f.name)
        pdf(text,[s/'figures'/f.name for f in figures],s/'RESEARCH_MANUSCRIPT_REVIEW.pdf')
        source_hashes={k:digest(v) for k,v in sources.items()}
        hashes={str(p.relative_to(s)):digest(p) for p in sorted(s.rglob('*')) if p.is_file()}
        inventory={'review_status':'DRAFT_NOT_SUBMISSION_CLEARED','source_sha256':source_hashes,
            'output_sha256':hashes,'copied_figures':len(figures),'station_run_entries':460,
            'assessment_is_synthetic':True,'no_new_model_training':True,'evidence_gates':gates,
            'manual_approvals_required':['VTUAD release','public source declaration','hydrophone source records',
                'natural-vessel attribution','pressure calibration','competition submission review']}
        (s/'MANUSCRIPT_REVIEW_MANIFEST.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
        s.rename(out)
    print('PHASE 5 MANUSCRIPT REVIEW: PASS — draft assembled, NOT submission approval')
    print('Acoustic figures:',len(figures),'| station runs:',460,'| PDF:',out/'RESEARCH_MANUSCRIPT_REVIEW.pdf')

def verify(root,out):
    manifest=json.loads((out/'MANUSCRIPT_REVIEW_MANIFEST.json').read_text())
    assert manifest['review_status']=='DRAFT_NOT_SUBMISSION_CLEARED'
    for key,sha in manifest['output_sha256'].items():
        p=out/key
        if not p.is_file() or digest(p)!=sha:raise ValueError('Output mismatch: '+str(p))
    if not (out/'RESEARCH_MANUSCRIPT_REVIEW.pdf').read_bytes().startswith(b'%PDF'):
        raise ValueError('PDF signature invalid')
    print('PHASE 5 MANUSCRIPT OUTPUT VERIFICATION: PASS', len(manifest['output_sha256']),'artifacts')

def self_test():
    with tempfile.TemporaryDirectory() as folder:
        base=Path(folder)
        stations=[{'station':f'S{i}','sampled_runs':str(n),'recorders':'1','median_observed_level':'60.123456'}
            for i,n in enumerate([120,1,22,77,120,120])]
        d={('original_only','.5_fixed'):{'rmse':'0.114236','auc':'0.848328','tpr_observed_positive':'0.739496','fpr_no_addition':'0.166667'},
            ('contrast_varied','.5_fixed'):{'rmse':'0.094566','auc':'0.886421','tpr_observed_positive':'0.876751','fpr_no_addition':'0.300000'},
            ('contrast_varied','calibration_selected'):{'rmse':'0.094566','auc':'0.886421','tpr_observed_positive':'0.852941','fpr_no_addition':'0.233333'},
            ('hybrid','.5_fixed'):{'rmse':'0.094566','auc':'0.848328','tpr_observed_positive':'0.739496','fpr_no_addition':'0.166667'}}
        pol=[(k,str(k)) for k in d]
        text=compose({'EXACT_LOCAL_VTUAD_RELEASE':'BLOCKED'},{'ledger_entries':525},stations,d,pol,[])
        assert '60.12' in text and 'pressure' in text and 'BLOCKED' in text
        p=base/'test.pdf';pdf(text,[],p)
        assert p.read_bytes().startswith(b'%PDF')
    print('PHASE 5 MANUSCRIPT SELF-TEST: PASS (synthetic manuscript fixture)')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--output',type=Path,default=DEST)
    a.add_argument('--self-test-only',action='store_true');a.add_argument('--verify-only',action='store_true')
    p=a.parse_args()
    if p.self_test_only:self_test()
    else:
        root=p.root.resolve();out=p.output if p.output.is_absolute() else root/p.output
        if p.verify_only:verify(root,out)
        else:build(root,out);verify(root,out)
