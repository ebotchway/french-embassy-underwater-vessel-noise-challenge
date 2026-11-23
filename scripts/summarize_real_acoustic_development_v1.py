#!/usr/bin/env python3
"""Read-only postprocessing for the Experiment 004 training-only real-acoustic audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQ = ['background_sampled_runs.csv', 'background_station_spectral_shapes.csv',
       'vessel_clip_inventory.csv', 'vessel_class_spectral_shapes.csv',
       'station_descriptive_summary.csv', 'shape_distance_descriptive.csv',
       'audit_summary.json']

def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def clean_name(s):
    return str(s).split('_', 1)[-1].replace('_', ' ')

def intervals(frame, seed, boots):
    rng = np.random.default_rng(seed)
    rows=[]
    for station, grp in frame.groupby('station',sort=True):
        if grp['run_id'].duplicated().any():
            raise ValueError(f'Duplicate run IDs in one-window-per-run sample: {station}')
        x=grp.window_mean_level.to_numpy(dtype=float)
        y=grp.within_minute_variability.to_numpy(dtype=float)
        n=len(x)
        if n<2:
            low=high=float('nan')
            vl=vh=float('nan')
        else:
            # A descriptive run-resampling interval, not a probability sample
            # of the entire station's time history.
            inds=rng.integers(0,n,size=(boots,n))
            meds=np.median(x[inds],axis=1)
            vars_=np.median(y[inds],axis=1)
            low,high=np.quantile(meds,[.025,.975]);vl,vh=np.quantile(vars_,[.025,.975])
        rows.append({'station':station,'sampled_runs':n,
                     'median_run_window_level':float(np.median(x)),
                     'median_level_bootstrap_lo':low,'median_level_bootstrap_hi':high,
                     'median_within_minute_variability':float(np.median(y)),
                     'median_variability_bootstrap_lo':vl,'median_variability_bootstrap_hi':vh,
                     'independent_run_uncertainty_supported':n>=2})
    return pd.DataFrame(rows)

def figures(st, bg, vt, dist, out):
    plt.rcParams.update({'font.size':10})
    stations=st.station.tolist()
    xx=np.arange(len(stations))
    fig,ax=plt.subplots(figsize=(11,5))
    vals=st.median_run_window_level.to_numpy()
    lower=st.median_level_bootstrap_lo.to_numpy()
    upper=st.median_level_bootstrap_hi.to_numpy()
    # Error bars only where n>=2; n=1 is not a legitimate interval.
    ok=np.isfinite(lower)&np.isfinite(upper)
    ax.bar(xx,vals)
    ax.errorbar(xx[ok],vals[ok],yerr=[vals[ok]-lower[ok],upper[ok]-vals[ok]],fmt='none',capsize=4,color='black')
    ax.set_xticks(xx,[clean_name(s) for s in stations],rotation=25,ha='right')
    ax.set_ylabel('Median window mean level (archive units; uncalibrated)')
    ax.set_title('Development acoustic levels: one sampled window per recording run')
    ax.text(.01,.98,'Durban: one run; no uncertainty interval',transform=ax.transAxes,va='top',fontsize=9)
    fig.tight_layout();fig.savefig(out/'station_levels.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6))
    for station,g in bg.groupby('station',sort=True):
        g=g.sort_values('band_hz')
        ax.plot(g.band_hz,g.median_relative_shape_db,label=clean_name(station))
    ax.set_xscale('log');ax.set_xlabel('Nominal band centre (Hz)')
    ax.set_ylabel('Relative band shape (dB below station peak)')
    ax.set_title('South African background spectral shapes (development only)')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'background_spectral_shapes.png',dpi=160);plt.close(fig)

    # Descriptive cross-source distance heatmap, not an estimate of likelihood.
    piv=dist.pivot(index='station',columns='vessel_class',values='median_shape_difference_rms_db')
    fig,ax=plt.subplots(figsize=(max(9,.9*len(piv.columns)),5))
    im=ax.imshow(piv.values,aspect='auto')
    ax.set_xticks(np.arange(len(piv.columns)),piv.columns,rotation=45,ha='right')
    ax.set_yticks(np.arange(len(piv.index)),[clean_name(s) for s in piv.index])
    ax.set_title('Shape-only RMS distances: background vs vessel-associated clips')
    cb=fig.colorbar(im,ax=ax);cb.set_label('RMS difference in relative dB shape')
    fig.tight_layout();fig.savefig(out/'shape_distance_heatmap.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('reports/audits/real_acoustic_development_v1'))
    p.add_argument('--output',type=Path,default=Path('reports/audits/real_acoustic_development_figures_v1'))
    p.add_argument('--seed',type=int,default=5204)
    p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args()
    if a.bootstrap<100:raise ValueError('At least 100 bootstrap replicates')
    for f in REQ:
        if not (a.input/f).is_file():raise FileNotFoundError(a.input/f)
    if a.output.exists():raise FileExistsError(f'Refusing to overwrite: {a.output}')
    audit=json.loads((a.input/'audit_summary.json').read_text())
    if 'Training-only' not in audit.get('scope',''):
        raise ValueError('Expected training-only provenance')
    sample=pd.read_csv(a.input/'background_sampled_runs.csv')
    bg=pd.read_csv(a.input/'background_station_spectral_shapes.csv')
    vf=pd.read_csv(a.input/'vessel_clip_inventory.csv')
    vt=pd.read_csv(a.input/'vessel_class_spectral_shapes.csv')
    dist=pd.read_csv(a.input/'shape_distance_descriptive.csv')
    for cols,frame in [({'station','run_id','window_mean_level','within_minute_variability'},sample),
                       ({'station','band_hz','median_relative_shape_db'},bg),
                       ({'vessel_class','band_hz','median_relative_shape_db'},vt),
                       ({'station','vessel_class','median_shape_difference_rms_db'},dist)]:
        if not cols.issubset(frame.columns):raise ValueError(f'Missing columns: {cols-set(frame.columns)}')
    if sample.run_id.duplicated().any():raise ValueError('Repeated runs in background_sampled_runs')
    if not np.isfinite(sample[['window_mean_level','within_minute_variability']].to_numpy(float)).all():
        raise ValueError('Non-finite descriptive levels')
    if len(bg)!=25*sample.station.nunique():raise ValueError('Expected 25 bands per station')
    st=intervals(sample,a.seed,a.bootstrap)
    # Compute everything before creating output directory.
    a.output.mkdir(parents=True,exist_ok=False)
    st.to_csv(a.output/'run_bootstrap_intervals.csv',index=False)
    figures(st,bg,vt,dist,a.output)
    lines=['# Real-acoustic development evidence — descriptive checkpoint','',
           'The plots use only `exp004_train` summary outputs, with one sampled 15-minute window per selected background recording run. No evaluation/validation/test measurements or hidden labels are used.','',
           '## Station summaries','',
           '| Station | Runs | Median run-window level | Descriptive 95% run bootstrap interval |',
           '|---|---:|---:|---|']
    for r in st.itertuples():
        ci=f'{r.median_level_bootstrap_lo:.3f} to {r.median_level_bootstrap_hi:.3f}' if r.independent_run_uncertainty_supported else 'Not estimable (one run)'
        lines.append(f'| {clean_name(r.station)} | {r.sampled_runs} | {r.median_run_window_level:.3f} | {ci} |')
    lines+=['','## Interpretation and limitations','',
            'These are archive level units, NOT calibrated dB re 1 µPa. Numeric cross-station differences cannot be assigned to vessel activity or physical pressure without instrument calibration and environmental context.',
            'Resampling intervals describe variation among the selected recording runs and do not correct for spatial/temporal sampling bias or recording-session dependence.',
            'VTUAD acoustic values are relative digital spectral powers. The spectral-shape comparison subtracts each source’s strongest band and is NOT a calibrated sound-pressure comparison.',
            'Similarity or distance between spectra cannot establish vessel presence, source identity, causal origin, or competition performance.',
            'Durban has one sampled training run; no run-bootstrap interval is calculated. Fish Hoek and Sea Point remain sealed.',
            'No model was trained or tuned and no leaderboard submission was made.','',
            '## Figures','',
            '- `station_levels.png`: sampled recorded levels and run-resampling descriptive intervals',
            '- `background_spectral_shapes.png`: median relative frequency shapes by station',
            '- `shape_distance_heatmap.png`: relative shape distances to VTUAD vessel-source classes',
            '','## Input SHA-256 checksums','']
    checks={k:hash_file(a.input/k) for k in REQ}
    lines += [f'- `{k}`: `{v}`' for k,v in checks.items()]
    (a.output/'RESEARCH_NOTES.md').write_text('\n'.join(lines)+'\n')
    (a.output/'analysis_manifest.json').write_text(json.dumps({'source':'training-only derived reports',
          'seed':a.seed,'bootstrap':a.bootstrap,'input_sha256':checks,
          'sampled_windows':len(sample),'vessel_clips':len(vf),
          'warning':'Descriptive development evidence, not physical calibration or independent external validation.'},indent=2)+'\n')
    print('REAL ACOUSTIC SCIENTIFIC FIGURES: PASS')
    print(st.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Output:',a.output)
    print('No evaluation, sealed holdout, or Zindi Test.csv accessed.')
if __name__=='__main__':main()
