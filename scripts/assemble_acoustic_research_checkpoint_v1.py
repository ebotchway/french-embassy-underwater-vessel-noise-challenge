#!/usr/bin/env python3
"""Assemble an evidence-based research manuscript draft from local audited reports."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_OUT=Path('reports/research_manuscript_v1')
DATA=Path('reports/audits')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def mdtable(frame,columns,round_digits=3):
    """Plain Markdown without an optional tabulate dependency."""
    rows=[]
    for _,r in frame[list(columns)].iterrows():
        row=[]
        for k in columns:
            v=r[k]
            if pd.isna(v):row.append('Not estimable')
            elif isinstance(v,(float,np.floating)):row.append(f'{v:.{round_digits}f}')
            else:row.append(str(v).replace('|','/'))
        rows.append('| '+' | '.join(row)+' |')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*rows])


def write(out,source,simulation):
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    files={
      'run_intervals':source/'real_acoustic_development_figures_v1'/'run_bootstrap_intervals.csv',
      'station_shape':source/'spectral_shape_mismatch_v1'/'station_shape_difference_summary.csv',
      'vessel_shape':source/'spectral_shape_mismatch_v1'/'vessel_class_shape_difference_summary.csv',
      'run_samples':source/'real_acoustic_development_v1'/'background_sampled_runs.csv',
      'source_inventory':source/'real_acoustic_development_v1'/'vessel_clip_inventory.csv',
    }
    if simulation.is_dir():
        files['simulation_summary']=simulation/'scenario_summary.csv'
        files['simulation_manifest']=simulation/'audit_manifest.json'
    for key,p in files.items():
        if not p.is_file():raise FileNotFoundError(f'{key}: {p}')
    station=pd.read_csv(files['run_intervals'])
    shape=pd.read_csv(files['station_shape'])
    vessel=pd.read_csv(files['vessel_shape'])
    runs=pd.read_csv(files['run_samples'])
    clips=pd.read_csv(files['source_inventory'])
    required_station={'station','sampled_runs','median_run_window_level','median_level_bootstrap_lo','median_level_bootstrap_hi','median_within_minute_variability'}
    required_shape={'station','median_shape_rms_db'}
    if not required_station<=set(station) or not required_shape<=set(shape):raise ValueError('Real-acoustic report schema changed')
    if len(station)!=6 or len(shape)!=6 or len(vessel)!=11 or runs.run_id.nunique()!=len(runs):
        raise ValueError('Unexpected audit scope; check source manifests')
    if len(clips)!=303:raise ValueError('Expected 303 audited training-source clips')
    sim=None
    if 'simulation_summary' in files:
        sim=pd.read_csv(files['simulation_summary'])
        assert {'spectral_intervention','requested_share','duration_min','median_change_step','background_runs'}<=set(sim)
        if len(sim)!=12:raise ValueError('Expected twelve fixed scenario summaries')
    station['median_level_bootstrap_lo']=pd.to_numeric(station.median_level_bootstrap_lo,errors='coerce')
    station['median_level_bootstrap_hi']=pd.to_numeric(station.median_level_bootstrap_hi,errors='coerce')
    stage=out.parent
    stage.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.acoustic_manuscript_',dir=stage) as temp:
        work=Path(temp)/'final';work.mkdir()
        figdir=work/'figures';figdir.mkdir()
        image_sources={
          'station_levels.png':source/'real_acoustic_development_figures_v1'/'station_levels.png',
          'background_spectral_shapes.png':source/'real_acoustic_development_figures_v1'/'background_spectral_shapes.png',
          'shape_distance_heatmap.png':source/'real_acoustic_development_figures_v1'/'shape_distance_heatmap.png',
          'spectral_shape_distance_matrix.png':source/'spectral_shape_mismatch_v1'/'spectral_shape_distance_matrix.png'
        }
        for name,path in image_sources.items():
            if path.is_file():shutil.copy2(path,figdir/name)
        # A station-relative anomaly is *not* a vessel activity estimate.
        tmp=runs[['station','run_id','window_mean_level']].copy()
        med=tmp.groupby('station').window_mean_level.transform('median')
        mad=tmp.groupby('station').window_mean_level.transform(lambda x:float(np.median(np.abs(x-np.median(x)))))
        tmp['station_relative_level_deviation']=tmp.window_mean_level-med
        tmp['station_robust_z']=(tmp.window_mean_level-med)/(1.4826*mad.where(mad>1e-9,np.nan))
        tmp.to_csv(work/'station_relative_anomaly_proxy.csv',index=False)
        text=[
          '# Relative acoustic structure across South African monitoring locations',
          '',
          '**Research manuscript — working draft (not yet a competition-ready scientific PDF)**',
          '',
          '## Abstract',
          '',
          'We analysed archived South African underwater acoustic measurements using 25 nominal one-third-octave/deci-decade band positions across 15-minute windows, together with AIS-associated vessel-source recordings from a separate hydrophone archive. The analysis characterizes recording coverage, station-dependent spectral profiles, temporal variability, and uncertainty across sampled recording runs. Source spectra are compared only after within-source peak normalization because calibration and physical comparability have not been established. In controlled synthetic simulations, group-separated random-forest models predicted relative injected energy well but did not reproduce similar detection performance on the provisional public competition evaluation. We report methodological limitations rather than claiming vessel attribution from unlabelled acoustic measurements.',
          '',
          '## 1. Research questions',
          '',
          '1. How do recorded band-level and temporal structures vary across South African acoustic monitoring stations?',
          '2. How do the relative spectral shapes of South African background recordings differ descriptively from VTUAD vessel-associated recordings?',
          '3. Do counterfactual synthetic insertion controls create minute-to-minute changes distinguishable from variations already present in the same backgrounds?',
          '4. Which conclusions remain unsupported without recorder calibration, independent vessel annotations, and verified station coordinates?',
          '',
          '## 2. Data and reproducibility',
          '',
          'The archive initially comprised 1,715,779 one-minute measurements across ten monitoring sites. The quality-controlled index provides 88,143 non-overlapping, fully observed 15-minute windows at eight sites. The frozen Experiment 004 source allocation contains 54,823 training and 12,640 reserved evaluation windows; the existing validation and two sealed station holdouts remain separate. The present real-data study reads **only exp004_train** and selects one 15-minute window from each of 460 sampled recording runs across six development stations. Vessel-source comparisons use 303 training clips from 109 hulls. The original vessel-source recordings can contain ambient energy and are not pure isolated vessel signals.',
          '',
          'The 25 nominal frequency-band centres run from 31.6 Hz to 7,943.3 Hz. Recorded archive levels are treated as level-like observations, **not independently calibrated dB re 1 μPa**. VTUAD source values are relative digital band powers. Conversion to normalized within-source shapes allows a descriptive comparison of frequency structure, **not an absolute pressure or sound-power comparison**.',
          '',
          '## 3. Methods',
          '',
          '### 3.1 Sampling, coverage and uncertainty',
          '',
          'One randomly selected 15-minute window per chosen development recording run was used, subject to a cap of 120 runs per station. Station medians were summarized across sampled runs. Run-resampling bootstrap intervals describe uncertainty within this sampled collection; they are not independent deployment-wide or population-wide confidence intervals. Durban had only one selected training run, so a run-bootstrap interval was not calculated.',
          '',
          '### 3.2 Spectral shape',
          '',
          'For each sampled background window and vessel clip, band power was aggregated in compatible **within-source** linear-energy units, converted to relative log levels, and referenced to its own highest band. Median 25-band profiles were constructed at station or vessel-class level. Root-mean-square differences across the 25 median band profiles were then calculated as **descriptive shape-distance values**; these distances are not vessel detection accuracy or a calibrated cross-dataset level difference.',
          '',
          '### 3.3 Synthetic benchmarks (supporting methods only)',
          '',
          'In Experiment 004, newly trained models used frozen, non-overlapping background runs and vessel hulls across train/evaluation. The original synthetic representation achieved fraction RMSE 0.114430 and detector ROC AUC 0.981296 on 300 frozen synthetic evaluation examples; the additionally maximum-shifted variant reached RMSE 0.150893 and ROC AUC 0.978653. These are synthetic evaluation findings; neither guarantees real-vessel detection.',
          '',
          'Our first Zindi submission achieved provisional public score 0.761535762 (share RMSE approximately 0.238464) and public diagnostic ROC AUC 0.517533. Public scoring and rank are not the final private evaluation, and these outputs are not used here to tune a parameter or infer hidden labels.',
          '',
          '## 4. Results',
          '',
          '### 4.1 Observed station-level differences',
          '',
          mdtable(station,['station','sampled_runs','median_run_window_level','median_level_bootstrap_lo','median_level_bootstrap_hi','median_within_minute_variability']),
          '',
          'Recorded station medians differ substantially, but the differences may include hydrophone characteristics, recording schedule, geographic conditions, deployments, and unresolved calibration effects. They cannot be interpreted as proven differences in vessel activity.',
          '',
          '### 4.2 Relative spectrum differences',
          '',
          mdtable(shape,['station','median_shape_rms_db']),
          '',
          'The largest median station-level shape distance occurs at Durban, but that station contributes only one sampled training run. Vessel-class comparisons include small-hull classes; dredger, pilot vessel, rescue, and tanker have fewer than three independent hulls. Source diversity is part of the result, not a nuisance to hide.',
          '',
          '### 4.3 Vessel-class distance and diversity',
          '',
          mdtable(vessel,['vessel_class','mean_shape_rms_db','vessel_clips','vessel_hulls']),
          '',
        ]
        if sim is not None:
            text+=['### 4.4 Controlled simulation-realism sensitivity (training-only)','',
                   'This prespecified study holds each background, source clip, insertion, duration and energy share fixed while shifting the source band shape toward that background. The three spectral conditions are original VTUAD, halfway-shaped and background-shaped controls. These interventions isolate the role of spectral contrast in **simulated minute-to-minute discontinuities**, not physical vessel validity. Natural background changes are not established as vessel-free.',
                   '',mdtable(sim,['spectral_intervention','requested_share','duration_min','background_runs','median_change_step','fraction_increase']), '']
        else:
            text+=['### 4.4 Controlled simulation-realism sensitivity — pending', '',
                   'The training-only counterfactual audit has been implemented but its actual South African acoustic results have not yet been executed or supplied. This section must not be described as completed.', '']
        text += [
          '## 5. Relative acoustic-activity proxy and geographic mapping restrictions',
          '',
          '`station_relative_anomaly_proxy.csv` contains each sampled run-window mean level minus the median for **that same station**, plus a robust within-station standardized value where feasible. This is a descriptive acoustic-level anomaly only, not a modelled vessel occurrence, causal vessel activity, absolute sound-pressure map, or comparison of calibrated energy between stations.',
          '',
          '**Geographic mapping is not yet authorized by the available evidence.** A publishable map needs trustworthy monitoring coordinates, datum/location definitions, monitoring-time coverage, deployment metadata, independent vessel or AIS evidence where vessel attribution is claimed, and appropriate uncertainty. Do not invent station coordinates or label an anomaly map “vessel traffic”. We will only map supported *relative acoustic observations*, clearly showing spatial/temporal coverage, when coordinate provenance is established.',
          '',
          '## 6. Limitations',
          '',
          '- Levels are not independently calibrated across sensors or to underwater reference pressure. Station differences must not be assigned physical pressure ratios.',
          '- Background labels are observational, not confirmed vessel-absent; natural vessels or other transients may be present.',
          '- One sampled window per run reduces long-run domination but does not yield independent deployments or seasons.',
          '- One Durban training run means its between-run uncertainty is not measurable.',
          '- Class counts and distinct hulls are highly imbalanced; vessel source recordings include ambient energy.',
          '- Spectral-shape distances are descriptive and cannot establish vessel presence or causality.',
          '- The frozen synthetic evaluation is grouped for new models, but its original source pool had been touched in historical development.',
          '- Public leaderboard statistics are provisional and are not independent evidence of the causes of domain shift.',
          '',
          '## 7. Discussion and next defensible steps',
          '',
          'The observed source and station differences justify controlled stress-testing of simulator assumptions, but do not prove why public detection was weak. Future work should document recorder/instrument calibration, develop independent real annotations or trusted AIS cross-checks, validate geospatial coordinates and sampling coverage, and perform an evaluation that is genuinely independent of all development inspection before drawing vessel-activity conclusions.',
          '',
          '## 8. Reproducibility and declarations',
          '',
          'Reproduction requires the specific frozen Experiment 004 manifests, South African acoustic archive, source-feature extraction code, relevant Python dependencies, and VTUAD release/checksums. The existing VTUAD material is documented locally as **v1.3.0 candidate**; confirm actual release identity before publishing the Zindi external-source declaration. A draft declaration is not a published discussion declaration.',
          '',
          '## Figure list',
          '',
          '- `figures/station_levels.png` — run-sampled archive-level station medians and descriptive bootstrap intervals.',
          '- `figures/background_spectral_shapes.png` — station-relative 25-band spectral profiles.',
          '- `figures/shape_distance_heatmap.png` — purely descriptive station/class spectral-shape distances.',
          '- `figures/spectral_shape_distance_matrix.png` — second audit visualization; contains the same descriptive information and should not be treated as independent evidence.',
          '',
          '## References and disclosure items to finalize',
          '',
          '- [ ] Verify and cite original South African archive documentation, monitoring instruments and any available coordinate metadata.',
          '- [ ] Verify and cite the exact VTUAD release/DOI, license, AIS labeling specification and access instructions.',
          '- [ ] Review Zindi discussion clarifications and current official submission/report requirements immediately before final filing.',
          '- [ ] Publish the required external-data declaration; record its direct discussion URL.',
          '- [ ] Include verified sampling diagrams, coverage/uncertainty figures and georeferenced relative activity maps only when their assumptions are supported.',
          '- [ ] Export the reviewed final manuscript to PDF and include runnable code/notebook and a public reproducibility manifest.',
          '',
        ]
        (work/'SCIENTIFIC_MANUSCRIPT_DRAFT.md').write_text('\n'.join(text))
        checklist='''# Submission-readiness and research integrity checklist

## Completed
- [x] Preserve Experiment 004 frozen allocation in a versioned archive.
- [x] Generate group-isolated synthetic training/evaluation data and initial model baselines.
- [x] Collect training-only station acoustic samples and bootstrap station summaries.
- [x] Produce 25-band station/class shape comparisons with source diversity flags.
- [x] Describe public Zindi score as provisional, not final private ranking.

## Pending verification or work
- [ ] Confirm exact VTUAD source release (local metadata says v1.3.0 candidate; public v1.3.2 references are not automatically equivalent).
- [ ] Post the exact public-data declaration on Zindi and store its URL.
- [ ] Validate input metadata provenance, hydrophone levels, recording periods, and calibration.
- [ ] Resolve station coordinates from original defensible records; never infer or invent positions.
- [ ] Prepare relative acoustic-activity maps with coverage/uncertainty, and separate observed anomalies from vessel inference.
- [ ] Obtain independent event/vessel annotations if vessel-detection claims are to be made.
- [ ] Avoid optimizing synthetic training or hidden test through repeated public leaderboard submissions.
- [ ] Run the training-only simulation-realism sensitivity script and assess source/run-level limitations.
- [ ] Update source notebook and learning log at this milestone without dropping executed outputs.
- [ ] Add citations, review figures and methods, finish PDF, package runnable notebooks/scripts.
'''
        if sim is not None:checklist=checklist.replace('- [ ] Run the training-only simulation-realism sensitivity script','- [x] Run the training-only simulation-realism sensitivity script')
        (work/'SUBMISSION_READINESS_CHECKLIST.md').write_text(checklist)
        (work/'STATION_COORDINATES_PROVENANCE_TEMPLATE.csv').write_text('station,latitude,longitude,coordinate_reference_system,source_url_or_document,source_page_or_row,verified_at,notes\n'+''.join(f'{s},,,,,,,\n' for s in sorted(station.station)))
        manifest={'source_scope':'Derived training-only summaries; frozen evaluation metrics reported as historical benchmarks only',
                  'inputs_sha256':{k:sha(p) for k,p in files.items()},'generated_files':{},
                  'simulation_results_included':sim is not None,
                  'warnings':['NOT a finalized report; no geospatial coordinates or inferred vessel activity','No independent physical sound pressure calibration']}
        for p in work.rglob('*'):
            if p.is_file() and p.name!='MANUSCRIPT_MANIFEST.json':manifest['generated_files'][str(p.relative_to(work))]=sha(p)
        (work/'MANUSCRIPT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        out.mkdir(exist_ok=False)
        # Temp staging prevents incomplete results from a missing dependency.
        for p in work.iterdir():shutil.move(str(p),str(out/p.name))
    print('RESEARCH MANUSCRIPT ASSEMBLY: PASS')
    print('Station summaries:',len(station),'vessel classes:',len(vessel),'training-source clips:',len(clips))
    print('Simulation realism:', 'INCLUDED' if sim is not None else 'PENDING')
    print('Output:',out)
    print('WARNING: Draft only; maps, exact data release and citations require verification.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audits',type=Path,default=DATA)
    p.add_argument('--simulation',type=Path,default=DATA/'simulation_realism_v1')
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=p.parse_args();write(args.output,args.audits,args.simulation)
if __name__=='__main__':main()
