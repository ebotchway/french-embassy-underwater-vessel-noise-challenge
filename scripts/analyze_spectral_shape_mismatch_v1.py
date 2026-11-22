#!/usr/bin/env python3
"""Read-only Phase 4 descriptive spectrum alignment report; development summaries only.

Run in repository root:
 python scripts/analyze_spectral_shape_mismatch_v1.py
No model fitting, hidden-test access, evaluation/holdout access, or source recalibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_IN = Path('reports/audits/real_acoustic_development_v1')
DEFAULT_OUT = Path('reports/audits/spectral_shape_mismatch_v1')
FILES = ('shape_distance_descriptive.csv','background_station_spectral_shapes.csv','vessel_class_spectral_shapes.csv')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def analyse(root, output, make_figure=True):
    if output.exists(): raise FileExistsError(f'Refusing to overwrite: {output}')
    src = {n:pd.read_csv(root/n) for n in FILES}
    d,b,v = (src[n] for n in FILES)
    required=[{'station','vessel_class','median_shape_difference_rms_db','background_runs','vessel_clips'},
              {'station','band_hz','median_relative_shape_db','sampled_distinct_runs'},
              {'vessel_class','band_hz','median_relative_shape_db','clips','hulls'}]
    for (name,frame),fields in zip(src.items(),required):
        if not fields <= set(frame): raise ValueError(f'Schema mismatch in {name}')
    for frame in (d,b,v):
        if frame.empty: raise ValueError('Empty source table')
    stations=sorted(d.station.unique()); classes=sorted(d.vessel_class.unique())
    if d.duplicated(['station','vessel_class']).any() or len(d)!=len(stations)*len(classes):
        raise ValueError('Missing or duplicated station/class pairs')
    if b.duplicated(['station','band_hz']).any() or v.duplicated(['vessel_class','band_hz']).any():
        raise ValueError('Duplicate spectral band/group keys')
    if set(stations)!=set(b.station) or set(classes)!=set(v.vessel_class):
        raise ValueError('Inconsistent station or class grouping')
    frequencies=sorted(b.band_hz.unique())
    if len(frequencies)!=25 or set(frequencies)!=set(v.band_hz): raise ValueError('Expected matching 25 band centres')
    bwide=b.pivot(index='station',columns='band_hz',values='median_relative_shape_db').reindex(index=stations,columns=frequencies)
    vwide=v.pivot(index='vessel_class',columns='band_hz',values='median_relative_shape_db').reindex(index=classes,columns=frequencies)
    if bwide.isna().any().any() or vwide.isna().any().any():raise ValueError('Incomplete band profiles')
    if not np.isfinite(bwide.to_numpy()).all() or not np.isfinite(vwide.to_numpy()).all():raise ValueError('Nonfinite profile')
    if not np.allclose(bwide.max(axis=1),0,atol=5) or not np.allclose(vwide.max(axis=1),0,atol=5):
        # Median of individually normalized source profiles need not have a 0 dB maximum.
        raise ValueError('Unexpectedly unnormalized median profiles; inspect source summaries')
    recomputed=[]
    for row in d.itertuples(index=False):
        bb=bwide.loc[row.station].to_numpy(float); vv=vwide.loc[row.vessel_class].to_numpy(float)
        distance=float(np.sqrt(np.mean(np.square(bb-vv))))
        if not np.isclose(distance,float(row.median_shape_difference_rms_db),rtol=1e-6,atol=1e-6):
            raise ValueError(f'Distance inconsistent with 25-band profiles: {row.station} {row.vessel_class}: {distance}')
        br=int(row.background_runs); clips=int(row.vessel_clips)
        hulls=int(v.loc[v.vessel_class.eq(row.vessel_class),'hulls'].iloc[0])
        recomputed.append({'station':row.station,'vessel_class':row.vessel_class,
           'shape_rms_db':distance,'background_runs':br,'vessel_clips':clips,'vessel_hulls':hulls,
           'limited_background_runs':br<10,'limited_vessel_hulls':hulls<3,
           'background_peak_band_hz':float(frequencies[np.argmax(bb)]),
           'vessel_peak_band_hz':float(frequencies[np.argmax(vv)]),
           'low_band_shape_difference_db':float(bb[0]-vv[0]),
           'peak_band_separation_log10':float(abs(np.log10(frequencies[np.argmax(bb)]/frequencies[np.argmax(vv)])))})
    table=pd.DataFrame(recomputed)
    by_station=table.groupby('station',as_index=False).agg(mean_shape_rms_db=('shape_rms_db','mean'),
              median_shape_rms_db=('shape_rms_db','median'),minimum_shape_rms_db=('shape_rms_db','min'),
              maximum_shape_rms_db=('shape_rms_db','max'),background_runs=('background_runs','first'),
              flagged_comparisons=('limited_vessel_hulls','sum'))
    by_class=table.groupby('vessel_class',as_index=False).agg(mean_shape_rms_db=('shape_rms_db','mean'),
              median_shape_rms_db=('shape_rms_db','median'),vessel_clips=('vessel_clips','first'),
              vessel_hulls=('vessel_hulls','first'),limited_hull_diversity=('limited_vessel_hulls','first'))
    output.mkdir(parents=True)
    table.to_csv(output/'all_station_class_differences.csv',index=False)
    by_station.to_csv(output/'station_shape_difference_summary.csv',index=False)
    by_class.to_csv(output/'vessel_class_shape_difference_summary.csv',index=False)
    if make_figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        heat=table.pivot(index='station',columns='vessel_class',values='shape_rms_db').loc[stations,classes]
        fig,ax=plt.subplots(figsize=(13,6))
        im=ax.imshow(heat.to_numpy(),aspect='auto')
        ax.set_xticks(range(len(classes)),classes,rotation=50,ha='right')
        ax.set_yticks(range(len(stations)),[s.split('_',1)[-1].replace('_',' ') for s in stations])
        ax.set_xlabel('VTUAD labelled vessel class'); ax.set_ylabel('South African monitoring station')
        ax.set_title('Median relative spectral-shape RMS difference (dB), descriptive only')
        fig.colorbar(im,ax=ax,label='Shape RMS difference (dB)')
        fig.tight_layout(); fig.savefig(output/'spectral_shape_distance_matrix.png',dpi=180);plt.close(fig)
    info={'type':'training-only descriptive report','station_count':len(stations),'vessel_class_count':len(classes),
          'comparisons':len(table),'bands':len(frequencies),
          'limited_background_station_comparisons':int(table.limited_background_runs.sum()),
          'limited_vessel_hull_comparisons':int(table.limited_vessel_hulls.sum()),
          'input_sha256':{n:digest(root/n) for n in FILES},
          'scope':'No evaluation/holdout or hidden test data; no model fitting.',
          'limits':['Profiles are medians of individually peak-referenced 25-band spectra.',
            'Distances compare shape only, not signal-to-noise, calibrated pressure, vessel presence, or causal effect.',
            'Stations and vessel classes are different source populations with potentially different hardware and ambient sound.',
            'Values are descriptive; there is no independent-sample confidence interval here.',
            'Small class hull counts and the single sampled Durban run limit generalization.']}
    (output/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    lines=['# Spectral-shape mismatch — training-only descriptive checkpoint','',
     'The 25-band median spectral profiles are peak-referenced within each original source record before calculating median station/class profiles. Cross-source distances are not calibrated level differences or vessel detection scores.','',
     '## Station-level distance summary','',by_station.to_markdown(index=False,floatfmt='.3f'),'',
     '## Vessel-class source diversity and distance summary','',by_class.to_markdown(index=False,floatfmt='.3f'),'',
     '## Limits and decisions','',
     '- Durban is represented by one sampled training run; no defensible independent-run interval.',
     '- Classes with fewer than three distinct hulls must not support strong class-level claims.',
     '- Large distances do not demonstrate that the synthetic injection can be detected in real data.',
     '- Freeze this as a descriptive checkpoint; test simulation assumptions on a future source-independent protocol, not by probing the competition test data.','']
    # No dependency on optional tabulate in case older pandas environments do not include it.
    try: (output/'research_note.md').write_text('\n'.join(lines))
    except ImportError:
        lines[6]=by_station.to_string(index=False)
        lines[10]=by_class.to_string(index=False)
        (output/'research_note.md').write_text('\n'.join(lines))
    print('SPECTRAL SHAPE MISMATCH AUDIT: PASS')
    print('Station/class comparisons:',len(table),'bands:',len(frequencies))
    print('Flagged: background samples <10 runs:',int(table.limited_background_runs.sum()),
          '; vessel classes <3 hulls:',int(table.limited_vessel_hulls.sum()))
    print('\nSTATIONS\n'+by_station.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nCLASSES\n'+by_class.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Output:',output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_IN)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--no-figure',action='store_true')
    args=p.parse_args()
    analyse(args.input,args.output,not args.no_figure)
if __name__=='__main__':main()
