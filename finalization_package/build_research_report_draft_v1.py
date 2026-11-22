#!/usr/bin/env python3
"""Build a candid, citable WORKING DRAFT PDF/Markdown from observed experiment checkpoints.

The numeric values below are transcribed from the user's 2026-11-22 local terminal
reports; they are not recalculated here. This deliberately does NOT promote the
document to final submission status or invent pressure maps/ground truth.
"""
from __future__ import annotations
import argparse, datetime as dt, html, json, re
from pathlib import Path

PUBLIC_SOURCES = [
 ('Official Zindi competition and evaluation','https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge'),
 ('Official competition data page','https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data'),
 ('Zindi research write-up clarification','https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/36266'),
 ('Zenodo VTUAD release page (verify exact locally used version)','https://zenodo.org/records/23037895'),
 ('Ocean Networks Canada archive','https://data.oceannetworks.ca/'),
 ('South African coastal soundscape study (check deployment identity before matching)','https://www.mdpi.com/2077-1312/10/6/746'),
]
SECTIONS = [
 ('Abstract', [
 'This working paper evaluates source-isolated synthetic vessel-noise additions to 15-minute underwater acoustic profiles, using South African archive backgrounds and relative digital-energy templates derived from Ocean Networks Canada VTUAD recordings. The objective is to evaluate methodological robustness while distinguishing experimentally imposed vessel energy from independently observed vessel activity.',
 'Prespecified Experiment 007 internal assessment produced lower fraction RMSE for contrast-varied source training (0.094566) than original-only source training (0.114236). The varied detector improved ROC AUC (0.886421 versus 0.848328) but increased background false positives at threshold 0.5 (30.0% versus 16.7%). A calibration-selected varied-detector threshold of 0.55 yielded 23.3% background false positives and 85.3% sensitivity on assessment. These are synthetic results, not evidence of performance on naturally labelled vessel passages.',
 ]),
 ('1. Research questions and data', [
 'The principal scientific questions are: how controllable source spectral contrast changes synthetic acoustic signatures; whether model performance is stable across source groups; and whether regression gains can be separated from detection-specificity costs.',
 'Archive-derived background profiles have 15 minutes and 25 deci-decade spectral bands in the challenge format. Source clips come from VTUAD (rebuilt), with 525 checksum-checked WAV files, 487 non-background-labelled clips and 38 background-labelled reference clips. The model-development subset and source separation follow the recorded experiment manifests. The exact local VTUAD version is a v1.3.0 candidate awaiting release verification, and should not be misrepresented as v1.3.2.',
 'Source WAV digital band power and South African acoustic levels have not been independently calibrated to a common pressure reference. Synthetic mixture share is a controlled relative-energy quantity, not a measurement of true isolated vessel sound.',
 ]),
 ('2. Data separation and reproducibility', [
 'Experiment 004 froze background-run and vessel-hull separation within the existing development pool. Experiments 005 and 006 focused on training-only development. Experiment 006C correctly stopped when a connected graph of reused background/source identities left 112 of 120 pairs inside one connected component, making strict calibration/assessment partitioning infeasible.',
 'Experiment 007 then allocated training, calibration and assessment sources before creating mixtures. Its independent integrity verifier reported no overlap by known background-run, vessel-hull or clip IDs. The final assessment used 120 background/source pairs, 840 model predictions per policy, 68 background runs, 14 hulls and 30 clips. These source groups were drawn from a broader pool already inspected historically, limiting claims of fully pristine independence.',
 ]),
 ('3. Measured acoustic contrast and synthetic realism', [
 'Experiment 005 used 460 background runs and 5,520 controlled scenario rows, varying 3%/15% requested relative-energy share, one-/three-minute duration, and original/halfway/background-shaped VTUAD source spectra. These treatments are deliberately counterfactual: aligning a source spectrum to background is not evidence of physically correct vessel acoustics.',
 'The median paired increase in maximum-step spectral-change statistic for original versus background-shaped spectra was 3.4952 (3%, one minute), 1.4173 (3%, three minutes), 6.5732 (15%, one minute), and 4.9443 (15%, three minutes). All four background-run bootstrap intervals were positive. St Francis had zero median difference at 3% in both durations. Durban provided only one training background run, preventing station-generalized inference.',
 'Experiment 005 deletion checks did not reverse the original-versus-background difference after removal of a single run, station, clip or hull. However, shared source/hydrophone dependencies remain, and this study does not prove the origin of the public-leaderboard generalization gap.',
 ]),
 ('4. Experiment 006 and 006B development findings', [
 'In Experiment 006 internal development (120 pairs, 840 rows), contrast-varied training lowered RMSE from 0.133224 to 0.114353 and improved ROC AUC from 0.847977 to 0.877137. At fixed threshold 0.5, genuine no-addition false-positive rate rose from 23.33% to 37.50%.',
 'Experiment 006B paired bootstrap reported RMSE reduction 0.018871 with 95% pair-bootstrap interval 0.006436–0.031532, ROC AUC gain 0.029160 with interval 0.016376–0.041305, and genuine no-addition false-positive rate increase 0.141667 with interval 0.075000–0.208333. Thirty-six requested-positive examples had zero observed share after masking and were treated distinctly from true no-addition controls.',
 'These paired intervals were conditional on the development sample and did not fully capture dependence from reused hulls, clips or deployments. No calibration was claimed from Experiment 006C because the strict split was infeasible.',
 ]),
 ('5. Experiment 007 independent internal assessment', [
 'Three strategies were compared: original-only regression/detection; contrast-varied regression/detection; and a hybrid consisting of the varied regressor and original detector. Calibration thresholds were selected on disjoint calibration groups under a 25% genuine-background false-positive ceiling; assessment observations were used once for reporting, not threshold tuning.',
 'The assessment found that contrast-varied regression and the hybrid share exactly the same RMSE of 0.094566, compared with 0.114236 for the original-only model (17.2% relative reduction). The original detector AUC was 0.848328 and the contrast-varied detector AUC was 0.886421. At a fixed 0.5 threshold, sensitivity/FPR were 73.95%/16.67% for original, versus 87.68%/30.00% for varied.',
 'On assessment the calibration-selected varied detector threshold was 0.55, giving sensitivity 85.29% and genuine no-addition FPR 23.33%. The calibration-selected original detector threshold of 0.35 yielded 82.35% sensitivity and 29.17% FPR on assessment; it breached the nominal 25% limit despite satisfying it during calibration.',
 'Run-cluster bootstrap estimated an RMSE reduction 0.01967 with one recorded 95% interval −0.00041 to 0.04230 (crossing zero). The AUC gain was 0.03809 with interval 0.02421–0.05520. Source-removal sensitivity retained positive point estimates. Bootstrap uncertainty is conditional on this development pool and does not fully cover hull/deployment dependence.',
 ]),
 ('6. South African real-acoustic evidence and geographic limits', [
 'Real-acoustic descriptive analysis sampled 460 background training runs across Richards Bay (120), Durban (1), Bird Island (22), St Croix Island (77), St Francis (120), and Saldanha Bay (120). The data reveal station-dependent levels, temporal variability and band-shape differences. None of these observational contrasts alone attributes acoustic activity to vessels.',
 'Ten station locations are presently recorded only as working site-level coordinate candidates. Saldanha Bay has two alternative positions with unresolved deployment assignment. No point may be presented as an exact underwater sensor deployment without matching instrument identity, deployment interval, documented coordinate system and authoritative source.',
 'A geographic pressure or vessel-related activity map is therefore NOT yet substantiated by these exported files. Relative, station-labelled acoustic figures may be used if sampling coverage and archive-level units are stated. Any spatial activity map must await verified deployments, and any vessel attribution needs independent timestamped evidence such as permitted AIS/port records with temporal matching and coverage limitations.',
 ]),
 ('7. Competition result and interpretation', [
 'Submission 001 achieved a captured provisional public score 0.761535762 (share RMSE about 0.238464238) and diagnostic AUC 0.517533333. Its evidence score was an uncalibrated duplicate of estimated share, not a separately trained detector. Public and synthetic performance use different populations, so a causal explanation for the gap cannot be inferred from the comparison alone.',
 'No further submission or hidden-test threshold/label fitting is claimed. The preserved Experiment 004 evaluation and sealed holdouts must not be repeatedly inspected for model selection. A second candidate should require a frozen inference recipe, source/version declaration, and explicit scientific limitations.',
 ]),
 ('8. Remaining research gates', [
 'Before treating this as a final research PDF: verify the exact locally used VTUAD version and release assets against Zenodo, publish the required external-data declaration, document original South African archive units and instrument calibrations, resolve hydrophone deployment coordinates, independently corroborate vessel activity for any vessel-related map, and include licensed figures/attribution and executable data-free reproduction instructions.',
 'The available evidence supports controlled relative-energy modelling and descriptive South African acoustic comparisons, not exact acoustic source separation, calibrated absolute sound-pressure mapping or independent natural-vessel detection accuracy. This document deliberately preserves those distinctions.',
 ]),
]

def markdown():
    top=['# Underwater vessel-noise estimation — research working draft',
         '', '**WORKING DRAFT — NOT YET A RESEARCH SUBMISSION**',
         '', 'Research checkpoint: 22 November 2026. Numeric findings transcribed from the researcher’s local terminal outputs and study reports; not recomputed in this build.',
         '', 'Research write-up deadline reported by official Zindi clarification: 13 December 2026 at 11:30 UTC. Reverify before submitting.', '']
    for heading,paras in SECTIONS:
        top+=['## '+heading,'']
        for p in paras:top += [p,'']
    top+=['## References and provenance links','']
    for title,url in PUBLIC_SOURCES:top += [f'- {title}: {url}']
    top+=['','## Required reproducibility artifacts (local project)', '',
          '- `reports/baselines/experiment007_source_disjoint_v1/` — assessment metrics, paired predictions, frozen allocation, manifest',
          '- `reports/audits/experiment007_decision_gate_v1/` — source deletion and run bootstrap summaries',
          '- `reports/audits/simulation_realism_uncertainty_v1/` — matched simulation contrast study',
          '- `notebooks/01_south_african_acoustic_eda.ipynb` — research record; preserve prior outputs',
          '- `reports/research_manuscript_v1/` — existing figures and prior manuscript',
          '- `VTUAD_EXTERNAL_DATA_DECLARATION_DRAFT.md` — must be verified and published, not simply attached',
          '', '---', '', 'Do not distribute organizer-restricted data or source WAV archives in a public deliverable without authorization.']
    return '\n'.join(top)+'\n'

def build_pdf(path:Path):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'],fontSize=19,leading=24,spaceAfter=15,textColor=colors.HexColor('#153450')))
    styles.add(ParagraphStyle(name='ReportSect', parent=styles['Heading2'],fontSize=11.5,leading=15,spaceBefore=14,spaceAfter=7,textColor=colors.HexColor('#153450')))
    styles.add(ParagraphStyle(name='ReportBody', parent=styles['BodyText'],fontSize=9,leading=13.7,spaceAfter=8))
    styles.add(ParagraphStyle(name='ReportSmall', parent=styles['BodyText'],fontSize=8,leading=12,spaceAfter=6))
    def esc(s):return html.escape(s).replace('−','-').replace('–','-').replace('—','-')
    story=[Spacer(1,0.2*inch),Paragraph('Underwater Vessel-Noise Estimation',styles['ReportTitle']),
           Paragraph('Source-separated synthetic experiments and South African acoustic evidence',styles['Heading2']),
           Spacer(1,0.18*inch),Paragraph('<b>WORKING RESEARCH DRAFT - NOT READY FOR SUBMISSION</b>',styles['ReportBody']),
           Paragraph('Evidence checkpoint: 22 November 2026. Results transcribed from local terminal reports; not independently recomputed for this PDF.',styles['ReportSmall']),
           HRFlowable(width='100%',thickness=1,color=colors.HexColor('#8197AA')),Spacer(1,0.15*inch)]
    for heading,paras in SECTIONS:
        story.append(Paragraph(esc(heading),styles['ReportSect']))
        for p in paras:story.append(Paragraph(esc(p),styles['ReportBody']))
    story.append(Paragraph('References and documentation',styles['ReportSect']))
    for title,url in PUBLIC_SOURCES:
        # long URLs wrap only if embedded as links; show labels and clickable links.
        story.append(Paragraph(f'{esc(title)}: <link href="{html.escape(url,quote=True)}" color="#276A91">source link</link>',styles['ReportSmall']))
    story.append(Paragraph('Reproducibility and distribution',styles['ReportSect']))
    story.append(Paragraph('Saved evidence: reports/baselines/experiment007_source_disjoint_v1; reports/audits/experiment007_decision_gate_v1; reports/audits/simulation_realism_uncertainty_v1. Rebuild against permitted development sources and frozen manifests. Do not package restricted competition data or copyrighted source audio.',styles['ReportBody']))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=47,leftMargin=47,topMargin=49,bottomMargin=45,
                          title='Underwater Vessel Noise - Working Research Draft',author='Research working paper')
    def foot(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#5D6570'))
        canvas.drawString(47,27,'Working draft - physical calibration and source attribution unverified')
        canvas.drawRightString(A4[0]-47,27,f'{doc.page}')
    doc.build(story,onFirstPage=foot,onLaterPages=foot)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='reports/research_manuscript_working_v1')
    ap.add_argument('--self-test-only',action='store_true');a=ap.parse_args()
    if a.self_test_only:
        assert '0.094566' in markdown() and 'NOT YET A RESEARCH SUBMISSION' in markdown()
        assert len(SECTIONS)>=8
        print('REPORT ASSEMBLER SELF-TEST: PASS');return
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    (out/'RESEARCH_WORKING_DRAFT.md').write_text(markdown())
    try:build_pdf(out/'RESEARCH_WORKING_DRAFT.pdf')
    except ImportError:
        (out/'PDF_DEPENDENCY_MISSING.txt').write_text('Install reportlab in the activated environment; Markdown is already produced.\n')
        print('REPORT ASSEMBLED: Markdown only; reportlab unavailable')
    else:print('REPORT ASSEMBLED: Markdown + PDF')
    (out/'BUILD_MANIFEST.json').write_text(json.dumps({'source':'user-provided terminal results, 2026-11-22; external links verified by web search', 'claim':'working draft only','real_vessel_attribution_verified':False,'deployment_coordinates_verified':False,'external_data_declaration_confirmed_published':False},indent=2)+'\n')
    print('Output:',out)
if __name__=='__main__':main()
