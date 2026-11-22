#!/usr/bin/env python3
"""Read-only evidence-gated audit of Phase 5 scientific manuscript and research artifacts.

Never clears submission automatically; does not train, fetch, or read hidden test data.
Uses only the Phase 5 review output, existing manifests, and selected code/docs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, tempfile
from pathlib import Path
from datetime import datetime, timezone

PHASE5 = Path('reports/research_manuscript_v1/submission_review_phase5_v1')
PHASE4 = Path('reports/research_manuscript_v1/submission_evidence_phase4_v1')
ACOUSTIC = Path('reports/research_manuscript_v1/real_acoustic_results_v2')
PROV = Path('reports/audits/submission_provenance_phase5_v1')
DEFAULT_OUT = Path('reports/audits/final_submission_review_phase6_v1')
MAX_BYTES = 10_000_000
REQUIRED_GATES = ['EXACT_LOCAL_VTUAD_RELEASE','VTUAD_SOURCE_DECLARATION', 'HYDROPHONE_DEPLOYMENTS',
                  'VESSEL_ATTRIBUTION','PHYSICAL_PRESSURE_CALIBRATION','FINAL_RESEARCH_SUBMISSION']

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check(condition, code, severity, status, evidence, recommendation, rows):
    rows.append({'code':code,'severity':severity,'status':status if condition else 'FAIL',
                 'evidence':str(evidence),'required_action':recommendation})

def scan(root: Path):
    root=root.resolve(strict=True)
    mdir=root/PHASE5
    md=mdir/'RESEARCH_MANUSCRIPT_REVIEW.md'
    pdf=mdir/'RESEARCH_MANUSCRIPT_REVIEW.pdf'
    manifest_file=mdir/'MANUSCRIPT_REVIEW_MANIFEST.json'
    rows=[]
    for code,p in [('MANUSCRIPT_MARKDOWN',md),('MANUSCRIPT_PDF',pdf),('MANUSCRIPT_MANIFEST',manifest_file)]:
        check(p.is_file(),code,'BLOCKER','PASS',str(p.relative_to(root)), 'Generate Phase 5 review artifacts; do not substitute fixture output.',rows)
    if not all(p.is_file() for p in (md,pdf,manifest_file)):
        return rows,{}, {'files_available':False}
    content=md.read_text(encoding='utf-8')
    text=content.lower()
    try: manifest=json.loads(manifest_file.read_text())
    except (ValueError,OSError) as exc:
        check(False,'MANIFEST_PARSE','BLOCKER','PASS',str(exc),'Regenerate verified Phase 5 manifest.',rows)
        return rows,{}, {'files_available':True}
    gates=manifest.get('evidence_gates')
    if not isinstance(gates,dict):gates={}
    expected=manifest.get('output_sha256') or {}
    mismatched=[]
    for rel,want in expected.items():
        # No absolute paths, traversal, symlink escape
        p=(mdir/rel).resolve()
        if not p.is_relative_to(mdir.resolve()) or not p.is_file() or sha256(p)!=want:
            mismatched.append(rel)
    check(not mismatched and len(expected)>=2,'MANUSCRIPT_OUTPUT_SHA256','BLOCKER','PASS',
          f'{len(expected)} recorded output hashes; mismatches {mismatched}',
          'Reconcile or rebuild verified Phase 5 outputs without overwriting existing evidence.',rows)
    check(pdf.read_bytes()[:5]==b'%PDF-','PDF_HEADER','BLOCKER','PASS',f'{pdf.stat().st_size} bytes',
          'Regenerate the PDF from reviewed source.',rows)
    check(pdf.stat().st_size<=MAX_BYTES,'PDF_10MB','BLOCKER','PASS', f'{pdf.stat().st_size} / {MAX_BYTES} bytes',
          'Reduce PDF size under 10 MB while preserving evidence.',rows)
    # pdf text extraction optional; do not fail manuscript because parser missing
    try:
        from pypdf import PdfReader
        reader=PdfReader(str(pdf)); pages=len(reader.pages)
        pdftext=' '.join((p.extract_text() or '') for p in reader.pages).lower()
        pdfcheck=all(x in pdftext for x in ['experiment', 'acoustic', 'vessel'])
        check(pdfcheck and pages>=2, 'PDF_TEXT_CONTENT','BLOCKER','PASS',
              f'{pages} pages, key manuscript topics present={pdfcheck}',
              'Visually inspect PDF and correct truncated or missing sections.',rows)
    except ImportError:
        rows.append({'code':'PDF_TEXT_CONTENT','severity':'REVIEW','status':'NOT_CHECKED',
                     'evidence':'pypdf not installed; PDF structure/size still checked',
                     'required_action':'Install pypdf and rerun; visually review rendered PDF pages.'})
    except Exception as exc:
        check(False,'PDF_TEXT_CONTENT','BLOCKER','PASS',f'PDF parsing failed: {exc}',
              'Inspect and rebuild PDF.',rows)
    # figure references and input integrity
    figs=re.findall(r'!\[[^]]*\]\((figures/[^)]+)\)',content)
    missing=[f for f in figs if not (mdir/f).is_file()]
    check(len(figs)==5 and not missing,'FIVE_FIGURES_PRESENT','BLOCKER','PASS',
          f'{len(figs)} referenced; missing={missing}',
          'Include all five verified descriptive figures with correct file references.',rows)
    check('assessment_is_synthetic' in manifest and manifest.get('assessment_is_synthetic') is True,
          'SYNTHETIC_ASSESSMENT_DISTINCTION','BLOCKER','PASS','Synthetic assessment flag in manifest',
          'Keep synthetic and observed domains separate.',rows)
    check('calibrated' in text and ('not' in text or 'unverified' in text),
          'PHYSICAL_CALIBRATION_DISCLOSURE','REVIEW','PASS', 'Mention of calibration/uncertainty',
          'Explicitly state archive levels are not demonstrated as calibrated dB re 1 µPa.',rows)
    check('durban' in text and ('one sampled run' in text or 'one run' in text),
          'SINGLE_RUN_LIMITATION','REVIEW','PASS','Durban run-count caveat',
          'Explain station N=1 and absent meaningful run-bootstrap CI.',rows)
    check('confidence interval' in text or 'bootstrap' in text,
          'UNCERTAINTY_EXPLANATION','REVIEW','PASS','Uncertainty language present',
          'Report interval definitions, grouping limitations and full numbers.',rows)
    check('reproduc' in text and 'reference' in text,
          'METHODS_REFERENCES','REVIEW','PASS','Methods/references section indicators',
          'Add full executable commands, exact dependencies, and access/citation details.',rows)
    # Check real-data attribution from explicit gate, do not infer from rhetorical wording
    for g in REQUIRED_GATES:
        value=str(gates.get(g,'NOT_RECORDED'))
        ok= value in ('VERIFIED','CONFIRMED','CLEARED','PASS')
        code='EVIDENCE_'+g
        check(ok,code,'BLOCKER','VERIFIED',value,
              'Provide authoritative independently checkable evidence, or keep this requirement explicitly unfulfilled.',rows)
    # Competition requests scientific pressure map and natural attribution; no unsupported map generation.
    qualifying_map = gates.get('HYDROPHONE_DEPLOYMENTS') in ('VERIFIED','CONFIRMED') and gates.get('VESSEL_ATTRIBUTION') in ('VERIFIED','CONFIRMED') and gates.get('PHYSICAL_PRESSURE_CALIBRATION') in ('VERIFIED','CONFIRMED')
    check(qualifying_map,'VESSEL_SPATIAL_MAP_EVIDENCE','BLOCKER','VERIFIED',
          'Requires verified deployments, independent attribution and physical-pressure reference',
          'Do not invent/interpolate pressure or vessel maps. Obtain linked metadata or present honest limitation and seek guidance from organizers.',rows)
    # External-data published proof cannot be checked from a local draft alone
    links=re.findall(r'https?://[^\s)<>]+',content)
    declaration_urls=[u for u in links if 'zindi' in u.lower() and '/discussions/' in u.lower()]
    check(bool(declaration_urls) and gates.get('VTUAD_SOURCE_DECLARATION') in ('VERIFIED','CONFIRMED'),
          'PUBLIC_EXTERNAL_DATA_DECLARATION','BLOCKER','VERIFIED',
          f'candidate public declaration links in manuscript: {len(declaration_urls)}; gate={gates.get("VTUAD_SOURCE_DECLARATION")}',
          'Publish reviewed source/version/license disclosure on Zindi and record exact URL; manually confirm posted contents.',rows)
    # Separate runnable deliverable exists? No automatic claim of reproducibility just because scripts exist.
    code_paths=[root/'scripts/build_real_acoustic_results_v2.py',root/'scripts/verify_real_acoustic_results_v2.py',
                root/'scripts/run_experiment007_source_disjoint_v1.py',root/'scripts/build_scientific_submission_review_v1.py']
    absent=[str(p.relative_to(root)) for p in code_paths if not p.is_file()]
    check(not absent,'CORE_REPRO_SCRIPTS_EXIST','REVIEW','PASS',f'Missing {absent}',
          'Include a clean, separate runnable research analysis artifact with dependencies and inputs.',rows)
    rows.append({'code':'CODE_SEPARATE_UPLOAD','severity':'BLOCKER','status':'NOT_VERIFIED',
                 'evidence':'No submission upload or separately packaged code verified',
                 'required_action':'Prepare reproducible code/notebook as separate <=10 MB file, run in fresh environment, and manually verify submission form.'})
    rows.append({'code':'REPRODUCIBILITY_FRESH_ENV','severity':'BLOCKER','status':'NOT_VERIFIED',
                 'evidence':'Local script existence does not demonstrate clean-environment reproduction',
                 'required_action':'Run documented code on clean env with accessible/redacted input data and compare figure, table and hashes.'})
    rows.append({'code':'PUBLIC_DATA_LICENSE_REVIEW','severity':'BLOCKER','status':'NOT_VERIFIED',
                 'evidence':'Local source card/licence inventory not legal or competition clearance',
                 'required_action':'Record source version, licence and redistribution terms; avoid bundling third-party audio where rights are unclear.'})
    rows.append({'code':'FINAL_MANUAL_VISUAL_REVIEW','severity':'BLOCKER','status':'NOT_VERIFIED',
                 'evidence':'PDF must be rendered and inspected page by page',
                 'required_action':'Review tables, page breaks, captions, maps, declarations and references before upload.'})
    summary={'files_available':True,'manuscript_pdf_size_bytes':pdf.stat().st_size,'figure_count':len(figs),
             'source_output_hash_count':len(expected),'source_pdf_sha256':sha256(pdf),
             'source_markdown_sha256':sha256(md),'source_manifest_sha256':sha256(manifest_file),
             'read_only':True,'approved_for_submission':False}
    return rows,gates,summary

def publish(root:Path,out:Path):
    if out.exists():raise FileExistsError(f'Refusing to overwrite existing audit: {out}')
    rows,gates,summary=scan(root)
    # Fail closed: even if all checks were to pass, no automated approval.
    blocked=[r for r in rows if r['severity']=='BLOCKER' and r['status'] not in ('PASS','VERIFIED')]
    summary.update({'blocker_count':len(blocked),'finding_count':len(rows),
                    'audit_time_utc':datetime.now(timezone.utc).isoformat(),
                    'upstream_evidence_gates':gates,
                    'submission_status':'NOT_CLEARED_AUTOMATICALLY'})
    out.parent.mkdir(parents=True,exist_ok=True)
    # Write into temp directory, then atomic rename after successful completion.
    with tempfile.TemporaryDirectory(prefix='.phase6_audit_',dir=str(out.parent)) as temp:
        t=Path(temp)
        with (t/'FINDINGS.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['code','severity','status','evidence','required_action']);writer.writeheader();writer.writerows(rows)
        (t/'AUDIT_MANIFEST.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        parts=['# Phase 6 — Final submission scientific audit','',
               '**STATUS: NOT CLEARED FOR SUBMISSION. This is an evidence inventory, not competition approval.**','',
               f"- Findings: {len(rows)}; unresolved blockers: {len(blocked)}.",
               f"- Figures in Phase 5 manuscript: {summary.get('figure_count','unavailable')}",
               '- This audit does not train models, reconstruct hidden labels, publish disclosures, or generate maps.','',
               '## Blockers and required actions','']
        for r in blocked:
            parts += [f"### {r['code']}: {r['status']}",f"Evidence: {r['evidence']}",
                      f"Required: {r['required_action']}",'']
        parts += ['## Passing / review-only checks','']
        for r in rows:
            if r not in blocked:parts += [f"- {r['code']}: {r['status']} — {r['evidence']}"]
        parts += ['','## Research interpretation','',
                  'The manuscript currently supports archive-level descriptive results and internal synthetic assessment only.',
                  'It does not establish instrument-calibrated vessel acoustic pressure, authentic sensor positions, or independent vessel-event labels.',
                  'The organizers request a vessel-related spatial pressure map and a separate reproducible code file; these remain substantive gates.',
                  'Do not use the model assessment results again for threshold tuning.','']
        (t/'SCIENTIFIC_AUDIT_REPORT.md').write_text('\n'.join(parts))
        t.rename(out)
    print('PHASE 6 AUDIT: COMPLETED, NOT SUBMISSION APPROVAL')
    print(f'Findings: {len(rows)}; unresolved blockers: {len(blocked)}; report: {out}')

def self_test():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        r=Path(td);m=r/PHASE5;m.mkdir(parents=True)
        (m/'figures').mkdir()
        for i in range(5):(m/'figures'/f'f{i}.png').write_bytes(b'fixture_png')
        text='\n'.join(['# Underwater acoustic report','## References','experiment acoustic vessel calibrated not pressure',
                        'Durban one sampled run confidence interval bootstrap reproducibility']+
                       [f'![figure](figures/f{i}.png)' for i in range(5)])
        (m/'RESEARCH_MANUSCRIPT_REVIEW.md').write_text(text)
        (m/'RESEARCH_MANUSCRIPT_REVIEW.pdf').write_bytes(b'%PDF-1.4\nfixture-not-real-pdf')
        hashes={'figures/'+f'f{i}.png':sha256(m/'figures'/f'f{i}.png') for i in range(5)}
        (m/'MANUSCRIPT_REVIEW_MANIFEST.json').write_text(json.dumps({'assessment_is_synthetic':True,
              'output_sha256':hashes,'evidence_gates':{k:'BLOCKED' for k in REQUIRED_GATES}}))
        rows,gates,summary=scan(r)
        assert any(x['code']=='VESSEL_SPATIAL_MAP_EVIDENCE' and x['status']=='FAIL' for x in rows)
        assert any(x['code']=='PUBLIC_EXTERNAL_DATA_DECLARATION' and x['status']=='FAIL' for x in rows)
        o=r/'audit';publish(r,o)
        assert (o/'FINDINGS.csv').is_file()
        try:publish(r,o)
        except FileExistsError:pass
        else:raise AssertionError('Overwrite unexpectedly allowed')
        assert json.loads((o/'AUDIT_MANIFEST.json').read_text())['approved_for_submission'] is False
    print('PHASE 6 SELF-TEST: PASS (synthetic fixture only, no real manuscript reviewed)')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.self_test_only:self_test()
    else:publish(a.root.resolve(),(a.root/a.output).resolve())
if __name__=='__main__':main()
