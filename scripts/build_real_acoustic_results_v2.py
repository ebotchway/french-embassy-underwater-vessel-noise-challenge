#!/usr/bin/env python3
"""Create evidence-limited South African acoustic descriptive results. No test data or geospatial attribu...."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = 'reports/audits/real_acoustic_development_v1/station_descriptive_summary.csv'
SPECTRA = 'reports/audits/real_acoustic_development_v1/background_station_spectral_shapes.csv'
PROXY = 'reports/research_manuscript_v1/station_relative_anomaly_proxy.csv'
DEFAULT_OUTPUT = 'reports/research_manuscript_v1/real_acoustic_results_v2'
NEEDED_STATION = ['station','sampled_runs','recorders','mean_observed_level','median_observed_level','median_within_window_range','median_temporal_band_variability']
NEEDED_SPECTRA = ['station','band_hz','median_relative_shape_db','p10_relative_shape_db','p90_relative_shape_db','sampled_distinct_runs']
NEEDED_PROXY = ['station','run_id','window_mean_level','station_relative_level_deviation','station_robust_z']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def validate(st: pd.DataFrame, sp: pd.DataFrame, pr: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    for name, df, cols in [('station summary',st,NEEDED_STATION),('spectral profiles',sp,NEEDED_SPECTRA),('run-level proxy',pr,NEEDED_PROXY)]:
        absent = sorted(set(cols)-set(df.columns))
        if absent:
            raise ValueError(f'{name}: missing columns {absent}')
        if df.empty:
            raise ValueError(f'{name}: no rows')
    st,sp,pr = st.copy(),sp.copy(),pr.copy()
    for df in (st,sp,pr):
        df['station'] = df['station'].astype(str)
        if df.station.isna().any() or (df.station.str.len()==0).any(): raise ValueError('empty station')
    if st.station.duplicated().any(): raise ValueError('station summary contains duplicate stations')
    if set(st.station)!=set(sp.station) or set(st.station)!=set(pr.station):
        raise ValueError('station sets differ across three inputs; refusing to silently discard stations')
    for name,df,cols in [
        ('station summary',st,NEEDED_STATION[1:]),('spectral',sp,NEEDED_SPECTRA[1:]),('proxy',pr,['window_mean_level','station_relative_level_deviation'])]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if not np.isfinite(df[c].to_numpy(dtype=float)).all(): raise ValueError(f'{name}: missing/nonfinite {c}')
    # Robust z is a DERIVED optional diagnostic: undefined when the station-scale
    # estimator is zero, absent, or nonfinite. Never impute an artificial zero.
    # Base run-level levels/deviations remain strictly required and finite.
    pr['station_robust_z'] = pd.to_numeric(pr['station_robust_z'], errors='coerce')
    pr['station_robust_z'] = pr['station_robust_z'].replace([np.inf, -np.inf], np.nan)
    if (st.sampled_runs<1).any() or (st.recorders<1).any() or not np.all(st.sampled_runs==np.floor(st.sampled_runs)):
        raise ValueError('invalid station run or recorder counts')
    if (sp.band_hz<=0).any() or (sp.sampled_distinct_runs<1).any():
        raise ValueError('invalid spectral band or count')
    if sp.duplicated(['station','band_hz']).any(): raise ValueError('duplicated spectral bands within station')
    if (sp.p10_relative_shape_db>sp.p90_relative_shape_db).any(): raise ValueError('spectral p10 > p90')
    if pr.run_id.isna().any() or (pr.run_id.astype(str).str.len()==0).any(): raise ValueError('empty run ID')
    if pr.duplicated(['station','run_id']).any(): raise ValueError('run proxy not one row per station/run; deduplication prohibited')
    ct=pr.groupby('station',sort=False).run_id.nunique()
    observed=st.set_index('station').sampled_runs
    mismatched={s:{'summary_runs':int(observed.loc[s]),'proxy_runs':int(ct[s])} for s in observed.index if int(observed.loc[s])!=int(ct[s])}
    if mismatched: raise ValueError(f'summary/proxy run-count mismatch: {mismatched}')
    distinct_bands=sp.groupby('station').band_hz.nunique().to_dict()
    if len(set(distinct_bands.values()))!=1: raise ValueError(f'inconsistent spectral band counts: {distinct_bands}')
    st=st.sort_values('station').reset_index(drop=True)
    sp=sp.sort_values(['station','band_hz']).reset_index(drop=True)
    pr=pr.sort_values(['station','run_id']).reset_index(drop=True)
    checks={'stations':int(len(st)),'runs':int(st.sampled_runs.sum()),'spectral_bands_per_station':int(next(iter(distinct_bands.values()))), 'run_ids_unique':True,'source_run_counts_match':True}
    return st,sp,pr,checks


def percentile_intervals(pr:pd.DataFrame, bootstrap:int, seed:int) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    records=[]
    for station,g in pr.groupby('station',sort=True):
        for metric in ['window_mean_level','station_relative_level_deviation','station_robust_z']:
            raw=g[metric].to_numpy(dtype=float)
            valid=raw[np.isfinite(raw)]
            total=len(raw)
            n=len(valid)
            row={'station':station,'metric':metric,'n_distinct_runs':total,
                 'n_valid_runs':n,'n_undefined_runs':total-n,
                 'median':np.nan,'p10':np.nan,'p90':np.nan,
                 'run_bootstrap_low':np.nan,'run_bootstrap_high':np.nan,
                 'bootstrap_valid_draws':0}
            if n:
                row.update(median=float(np.median(valid)),
                           p10=float(np.percentile(valid,10)),
                           p90=float(np.percentile(valid,90)))
            # Conditional interval only if at least ten valid run values.
            if n>=10:
                boot=np.empty(bootstrap)
                for j in range(bootstrap):
                    boot[j]=np.median(valid[rng.integers(n,size=n)])
                row.update(run_bootstrap_low=float(np.percentile(boot,2.5)),
                           run_bootstrap_high=float(np.percentile(boot,97.5)),
                           bootstrap_valid_draws=bootstrap)
            records.append(row)
    return pd.DataFrame.from_records(records)


def make_figures(st:pd.DataFrame,sp:pd.DataFrame,pr:pd.DataFrame,ci:pd.DataFrame,out:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    labels=st.station.str.replace(r'^\d+_', '', regex=True).str.replace('_',' ')
    fig,ax=plt.subplots(figsize=(10,5.1))
    yy=np.arange(len(st))
    ax.barh(yy,st.sampled_runs)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis(); ax.set_xlabel('Distinct sampled background recording runs')
    ax.set_title('Training-only acoustic sample coverage by station')
    for k,n in enumerate(st.sampled_runs): ax.text(n+max(st.sampled_runs)*.012,k,str(int(n)),va='center',fontsize=9)
    ax.set_xlim(0, max(st.sampled_runs)*1.16)
    fig.tight_layout(); fig.savefig(out/'station_run_coverage.png',dpi=175);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.2))
    yy=np.arange(len(st))
    lo=[];hi=[]
    for station,med in zip(st.station,st.median_observed_level):
        vals=ci[(ci.station==station)&(ci.metric=='window_mean_level')]
        # The CI describes median of run-level mean, NOT the station summary's differently defined median.
        v=vals.iloc[0]; lo.append(v.run_bootstrap_low);hi.append(v.run_bootstrap_high)
    medval=ci[ci.metric.eq('window_mean_level')].set_index('station').loc[st.station,'median'].to_numpy()
    ax.plot(medval,yy,'o',label='Median of sampled run-window means')
    for i,(a,b) in enumerate(zip(lo,hi)):
        if np.isfinite(a) and np.isfinite(b): ax.plot([a,b],[i,i],lw=2)
        else: ax.text(medval[i],i+.20,'single run: interval unavailable',fontsize=8)
    ax.set_yticks(yy,labels);ax.invert_yaxis()
    ax.set_xlabel('Archive-reported level statistic (unit/reference unverified)')
    ax.set_title('Run-level acoustic levels with conditional 95% bootstrap intervals')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'station_run_levels_intervals.png',dpi=175);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.1))
    ax.barh(yy,st.median_temporal_band_variability)
    ax.set_yticks(yy,labels);ax.invert_yaxis();ax.set_xlabel('Archived spectral-temporal variation statistic (not calibrated SPL)')
    ax.set_title('Within-window temporal-band variability: station descriptive summary')
    fig.tight_layout();fig.savefig(out/'station_temporal_variability.png',dpi=175);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.5))
    for s,g in sp.groupby('station',sort=True):
        label=s.split('_',1)[-1].replace('_',' ')
        ax.plot(g.band_hz,g.median_relative_shape_db,label=label,lw=1.6)
    ax.set_xscale('log');ax.set_xlabel('Band centre frequency (Hz)')
    ax.set_ylabel('Peak-referenced median spectral shape (dB, relative)')
    ax.set_title('Station spectral shapes; independent peak references')
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'station_relative_spectral_shapes.png',dpi=175);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.2))
    vals=[pr.loc[pr.station.eq(s),'station_relative_level_deviation'].to_numpy() for s in st.station]
    ax.boxplot(vals,orientation='horizontal',tick_labels=labels,showfliers=False)
    ax.set_xlabel('Station-relative level deviation (archive level units)')
    ax.set_title('Run-level station-relative deviations; no vessel attribution')
    fig.tight_layout();fig.savefig(out/'station_relative_deviations.png',dpi=175);plt.close(fig)


def fmt(v,dec=3):
    return 'not estimated' if not np.isfinite(v) else f'{v:.{dec}f}'


def make_markdown(st:pd.DataFrame, ci:pd.DataFrame,checks:dict, srcs:dict,bootstrap:int, seed:int) -> str:
    rows=[]
    m=ci[ci.metric=='window_mean_level'].set_index('station')
    for _,r in st.iterrows():
        c=m.loc[r.station]
        rows.append(f"| {r.station.replace('_',' ')} | {int(r.sampled_runs)} | {int(r.recorders)} | {fmt(r.median_observed_level)} | {fmt(r.median_temporal_band_variability)} | {fmt(c['median'])} | {fmt(c['run_bootstrap_low'])} to {fmt(c['run_bootstrap_high'])} |")
    return f'''# Real South African acoustic evidence — descriptive research appendix (DRAFT)

**Evidence classification:** Training-only descriptive soundscape study. The figures are **not calibrated sound-pressure maps**, and station-relative fluctuations are **not confirmed vessel passages**.

## Scope and scientific question

We characterize level, temporal variability, and spectral shape in previously sampled, distinct South African background recording runs. This appendix does not use the competition hidden test file, preserved synthetic evaluation allocations, or newly trained models. It does not assert that the input recordings are vessel-free; the term “background” identifies the archive pool used by the synthetic-mixture program.

## Input inventory and provenance

| Source | SHA-256 |
|---|---|
| `station_descriptive_summary.csv` | `{srcs['station_summary']}` |
| `background_station_spectral_shapes.csv` | `{srcs['spectral_shapes']}` |
| `station_relative_anomaly_proxy.csv` | `{srcs['relative_proxy']}` |

Run and station identifiers must be consistent across all three files. Exact sensor deployment, clock alignment, physical reference units, sample selection, and calibration are **not independently verified by this appendix**.

## Design and statistical methods

* We use one preserved proxy row per unique `(station, run_id)`; duplicates or inconsistent run counts cause an error rather than arbitrary deduplication.
* Station descriptive median-level and median temporal variability are taken as reported in the prior training-only station summary. They are not recomputed or labelled as hydrophone-calibrated SPL.
* The run-level uncertainty visualization uses the **median of `window_mean_level` over distinct sampled run entries**. This quantity differs from the original station-summary `median_observed_level`, so neither is substituted for the other.
* Conditional 95% percentile bootstrap intervals use **{bootstrap} run-level draws**, random seed **{seed}**, separately within each station. No interval is issued for stations with fewer than ten sampled runs. Resampling runs does not remove recorder/deployment dependence or sampling-selection bias.
* Spectral shapes are frequency-by-frequency archived relative median shapes, independently peak-referenced by station. No shared physical pressure reference is assumed.
* We do not compute inter-station vessel rates, spatial interpolation, acoustic exposure limits, or causal vessel attribution.

## Observed station-level results

| Station | Sampled runs | Recorders | Archived median level | Archived median temporal variability | Median run-window mean | Bootstrap 95% interval for run median |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Derived robust-z diagnostic availability

The `station_robust_z` field is a **derived, optional diagnostic**, not a prerequisite for the archived run-level mean or station-relative deviation. A missing or nonfinite robust z-score is recorded as *undefined*, never replaced with zero. A station with no defined robust-z values has no estimated robust-z distribution; at least ten valid values are required for its run bootstrap interval. We cannot determine the cause of undefined values from these derived files alone (e.g. zero robust scale or upstream missingness).

| Station | Run entries | Defined robust z | Undefined robust z |
|---|---:|---:|---:|
{chr(10).join(f"| {r.station.replace('_',' ')} | {int(r.n_distinct_runs)} | {int(r.n_valid_runs)} | {int(r.n_undefined_runs)} |" for _,r in ci.loc[ci.metric.eq('station_robust_z')].iterrows())}

The complete finite/undefined accounting is in `run_bootstrap_intervals.csv` and `robust_z_availability.csv`. Statistics for finite robust-z values, if present, are *conditional on availability* and should not be generalized to missing rows.

The station count is **{checks['stations']}**, with **{checks['runs']}** sampled run entries and **{checks['spectral_bands_per_station']}** recorded frequency bands per station in the existing spectral-shape report. Uneven station sample sizes, the single-run Durban limitation if present, and possible shared recorder deployments affect inferential precision.

## Figures produced

1. `figures/station_run_coverage.png`: run-count coverage, **not recording duration or geographic coverage**.
2. `figures/station_run_levels_intervals.png`: conditional run-bootstrap intervals for *median of run-window means*, without a calibrated SPL claim.
3. `figures/station_temporal_variability.png`: descriptive variation statistic from the station summary.
4. `figures/station_relative_spectral_shapes.png`: independently peak-referenced spectral shapes, not absolute frequency-band energies.
5. `figures/station_relative_deviations.png`: run-level station-relative deviations, not vessel-event detections.

## Links to simulation/model evidence

Experiment 005 showed synthetic insertion sensitivity to source–background spectral contrast and station. Experiments 006–007 tested original-only, contrast-varied and hybrid modelling under synthetic labels. The separately reported Experiment 007 internal assessment showed original-only fraction RMSE 0.114236 and contrast-varied fraction RMSE 0.094566, with detector ROC AUC 0.848328 versus 0.886421. These modelling statistics are **not derived from the real acoustic rows plotted here** and are not evidence of natural-vessel detection accuracy. The run-bootstrap interval for original-to-varied RMSE improvement under the fixed policy crossed zero in that study. No further tuning is authorized from those assessment rows.

## Outstanding verification before a final submission

- Identify the exact local VTUAD release/archive asset and publish the permissible external-data declaration with traceable public URL.
- Verify original recorder/sensor identifiers, deployment coordinates, CRS, and recording-date linkage before producing **any** precise hydrophone map.
- Establish instrument sensitivity and pressure reference/processing convention before labelling results as calibrated underwater SPL.
- Obtain independent time-aligned vessel evidence (e.g., trustworthy AIS linkage plus known coverage or annotations) before identifying acoustic excursions as vessel traffic.
- Verify the original sample selection, station coverage periods, station-level exposure and any season/recorder imbalances.
- Finish citations, methodology, figure captions, code execution and the final research PDF using the completed evidence gates.

**Conclusion:** The source-checked summaries support a useful, geographically unlocated *descriptive* acoustic analysis. They do not yet support vessel attribution, causal domain-shift explanations, or a pressure/activity map at sensor coordinates.
'''


def make_pdf(text:str, figures:Path, output:Path):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from xml.sax.saxutils import escape
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallResearch',parent=styles['BodyText'],fontSize=9,leading=13,spaceAfter=6))
    styles.add(ParagraphStyle(name='SectionResearch',parent=styles['Heading2'],fontSize=12,leading=15,spaceBefore=12,spaceAfter=7))
    story=[]
    # Purpose-built compact PDF with source-derived table lines and actual figures. Markdown remains the full editable appendix.
    story.append(Paragraph('Real South African Acoustic Evidence',styles['Title']))
    story.append(Paragraph('Research appendix — working draft, not a calibrated vessel-activity map',styles['SmallResearch']))
    story.append(Spacer(1,8))
    section=''
    for line in text.splitlines():
        if line.startswith('## '):
            section=line[3:]
            story.append(Paragraph(escape(section),styles['SectionResearch']))
        elif line.startswith('|'):
            # full numeric table is already available as CSV / Markdown; include only limited rows for pdf legibility
            if section=='Observed station-level results' and not line.startswith('|---'):
                s=line.strip('| ').replace('|','; ')
                if len(s)>170:s=s[:170]+'...'
                story.append(Paragraph(escape(s),styles['SmallResearch']))
        elif line.startswith('#') or not line.strip() or line.startswith('**Conclusion:**'):
            continue
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph('- '+escape(line[2:].replace('**','').replace('`','')), styles['SmallResearch']))
        elif len(line)<500 and not line.startswith('1.') and not line.startswith('2.'):
            story.append(Paragraph(escape(line.replace('**','').replace('`','')),styles['SmallResearch']))
    story.append(Paragraph('Scientific figures',styles['SectionResearch']))
    for path in sorted(figures.glob('*.png')):
        im=Image(str(path))
        if im.drawWidth>480:
            factor=480/im.drawWidth; im.drawWidth*=factor;im.drawHeight*=factor
        if im.drawHeight>325:
            factor=325/im.drawHeight;im.drawWidth*=factor;im.drawHeight*=factor
        story.append(KeepTogether([Paragraph(escape(path.stem.replace('_',' ').capitalize()),styles['SmallResearch']),im,Spacer(1,10)]))
    doc=SimpleDocTemplate(str(output),pagesize=(595.3,841.9),leftMargin=49,rightMargin=49,topMargin=48,bottomMargin=48,title='Training-only South African acoustic descriptive research appendix')
    doc.build(story)


def build(root:Path, output:Path, bootstrap=1000,seed=42):
    if output.exists(): raise FileExistsError(f'Preserving existing output: {output}')
    paths={'station_summary':root/STATIONS,'spectral_shapes':root/SPECTRA,'relative_proxy':root/PROXY}
    for label,p in paths.items():
        if not p.is_file(): raise FileNotFoundError(f'{label}: {p}. Reuse existing research outputs; do not regenerate data blindly.')
    st,sp,pr,checks=validate(*(pd.read_csv(paths[k]) for k in paths))
    source_hashes={k:sha(v) for k,v in paths.items()}
    ci=percentile_intervals(pr,bootstrap,seed)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.real_acoustic_stage_',dir=output.parent) as tmp:
        stage=Path(tmp)
        make_figures(st,sp,pr,ci,stage/'figures')
        st.to_csv(stage/'validated_station_summary.csv',index=False)
        ci.to_csv(stage/'run_bootstrap_intervals.csv',index=False,na_rep='')
        ci.loc[ci.metric.eq('station_robust_z'),
               ['station','n_distinct_runs','n_valid_runs','n_undefined_runs']].to_csv(
                   stage/'robust_z_availability.csv',index=False)

        markdown=make_markdown(st,ci,checks,source_hashes,bootstrap,seed)
        (stage/'REAL_ACOUSTIC_RESULTS_APPENDIX.md').write_text(markdown,encoding='utf-8')
        make_pdf(markdown,stage/'figures',stage/'REAL_ACOUSTIC_RESULTS_APPENDIX.pdf')
        outputs={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
        manifest={'analysis':'real_acoustic_results_v2','source_paths':{k:str(v.relative_to(root)) for k,v in paths.items()},'source_sha256':source_hashes,'input_checks':checks,'bootstrap_draws':bootstrap,'random_seed':seed,'claims_excluded':['calibrated_SPL','verified_hydrophone_coordinates','vessel_attribution','spatial_heatmap'],'outputs_sha256':outputs}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        stage.rename(output)
    print('REAL ACOUSTIC RESULTS APPENDIX: PASS')
    print(f"Station summaries: {checks['stations']}; distinct sampled run rows: {checks['runs']}; spectral bands: {checks['spectral_bands_per_station']}")
    z = ci.loc[ci.metric.eq('station_robust_z')]
    print(f"Derived robust-z availability: {int(z.n_valid_runs.sum())} defined; {int(z.n_undefined_runs.sum())} undefined (never imputed)")
    print(f'Output: {output}')
    print('LIMIT: Descriptive training-only measurements; no hydrophone coordinate or vessel attribution claims.')


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for sub in [Path(STATIONS).parent,Path(SPECTRA).parent,Path(PROXY).parent]: (root/sub).mkdir(parents=True,exist_ok=True)
        rows=[];spectra=[];prox=[]
        for station,count in [('01_A',12),('02_B',14)]:
            rows.append(dict(station=station,sampled_runs=count,recorders=2,mean_observed_level=61.0,median_observed_level=60.0,median_within_window_range=2.2,median_temporal_band_variability=1.8))
            for j in range(count): prox.append(dict(station=station,run_id=f'{station}_{j}',window_mean_level=60+j/6,station_relative_level_deviation=j/6-1,station_robust_z=(j/6-1)/.9))
            for i in range(25):spectra.append(dict(station=station,band_hz=31.6*10**(i/10),median_relative_shape_db=-i*.2,p10_relative_shape_db=-i*.2-1,p90_relative_shape_db=-i*.2+1,sampled_distinct_runs=count))
        pd.DataFrame(rows).to_csv(root/STATIONS,index=False);pd.DataFrame(spectra).to_csv(root/SPECTRA,index=False);pd.DataFrame(prox).to_csv(root/PROXY,index=False)
        proxy = pd.read_csv(root/PROXY)
        proxy.loc[proxy.station.eq('01_A'),'station_robust_z'] = np.nan
        proxy.loc[proxy.index[proxy.station.eq('02_B')][:3],'station_robust_z'] = np.nan
        proxy.to_csv(root/PROXY,index=False)
        build(root,root/'results',40,43)
        intervals = pd.read_csv(root/'results/run_bootstrap_intervals.csv')
        undefined = intervals[intervals.metric.eq('station_robust_z')].set_index('station')
        assert int(undefined.loc['01_A','n_valid_runs'])==0
        assert pd.isna(undefined.loc['01_A','median'])
        assert int(undefined.loc['02_B','n_valid_runs'])==11
        assert int(undefined.loc['02_B','n_undefined_runs'])==3
        assert (root/'results/REAL_ACOUSTIC_RESULTS_APPENDIX.pdf').stat().st_size>1000
        try:
            build(root,root/'results',40,43)
            raise AssertionError('Overwrite guard failed')
        except FileExistsError:pass
        invalid=pd.read_csv(root/STATIONS);invalid.loc[0,'sampled_runs']=0;invalid.to_csv(root/STATIONS,index=False)
        try:
            build(root,root/'invalid_out',40,43)
            raise AssertionError('Invalid run count guard failed')
        except ValueError:pass
    print('REAL ACOUSTIC RESULTS APPENDIX SELF-TEST: PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path(DEFAULT_OUTPUT))
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--self-test-only',action='store_true')
    a=p.parse_args()
    if a.bootstrap<20: p.error('--bootstrap must be >=20')
    if a.self_test_only:self_test()
    else:build(a.root,a.output,a.bootstrap,a.seed)

if __name__=='__main__':main()
