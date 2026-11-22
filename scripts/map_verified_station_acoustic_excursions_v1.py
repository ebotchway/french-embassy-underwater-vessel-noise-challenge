#!/usr/bin/env python3
"""Build a verified-coordinate station excursion map, never a vessel-density map.

Coordinates MUST be manually verified from original reliable monitoring site
metadata. The manuscript assembler creates a blank template to complete.
If any coordinate/evidence is missing, fail before drawing. No basemap, station
interpolation, or physical sound pressure calibration is asserted.

  python scripts/map_verified_station_acoustic_excursions_v1.py --self-test-only
  python scripts/map_verified_station_acoustic_excursions_v1.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_INPUT=Path('reports/research_manuscript_v1/station_relative_anomaly_proxy.csv')
DEFAULT_COORD=Path('reports/research_manuscript_v1/STATION_COORDINATES_PROVENANCE_TEMPLATE.csv')
DEFAULT_OUTPUT=Path('reports/research_manuscript_v1/verified_station_map')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(proxy,coords):
    required={'station','run_id','window_mean_level'}
    if not required<=set(proxy):raise ValueError('Proxy schema incomplete')
    required_c={'station','latitude','longitude','coordinate_reference_system','source_url_or_document','source_page_or_row'}
    if not required_c<=set(coords):raise ValueError('Coordinate provenance schema incomplete')
    if proxy.station.isna().any() or coords.station.isna().any():raise ValueError('Missing station names')
    expected=set(proxy.station)
    if set(coords.station)!=expected or len(coords)!=len(expected) or coords.station.duplicated().any():
        raise ValueError('Coordinate table must contain exactly the studied stations')
    if coords[list(required_c-{'station'})].isna().any().any():
        raise ValueError('Coordinates or authoritative provenance missing: do not draw an unverified map')
    if not coords.coordinate_reference_system.astype(str).str.upper().eq('EPSG:4326').all():
        raise ValueError('This plot only accepts WGS84 EPSG:4326 coordinates')
    lat=pd.to_numeric(coords.latitude,errors='raise');lon=pd.to_numeric(coords.longitude,errors='raise')
    if not (lat.between(-90,90).all() and lon.between(-180,180).all()):raise ValueError('Invalid coordinates')
    if coords.source_url_or_document.astype(str).str.strip().eq('').any() or coords.source_page_or_row.astype(str).str.strip().eq('').any():
        raise ValueError('A nonempty authoritative source and locator are required for every station')
    rows=[]
    for station,g in proxy.groupby('station',sort=True):
        if g.run_id.duplicated().any():raise ValueError('Duplicate background runs')
        level=g.window_mean_level.to_numpy(float)
        if len(level)==0 or not np.isfinite(level).all():raise ValueError('Invalid station levels')
        median=np.median(level);mad=np.median(np.abs(level-median))
        threshold=float(median+1.4826*mad)
        frac=float(np.mean(level>threshold)) if mad>1e-9 else np.nan
        rows.append({'station':station,'sampled_runs':len(level),'station_level_median':median,
                     'station_mad':mad,'within_station_excursion_threshold':threshold,
                     'fraction_windows_above_own_station_threshold':frac})
    data=pd.DataFrame(rows).merge(coords,on='station',validate='one_to_one',how='left')
    return data


def self_test():
    p=pd.DataFrame([{'station':'A','run_id':f'r{i}','window_mean_level':float(v)} for i,v in enumerate([1.,2.,3.,10.])])
    c=pd.DataFrame([{'station':'A','latitude':-33.1,'longitude':19.2,'coordinate_reference_system':'EPSG:4326',
       'source_url_or_document':'fixture-source','source_page_or_row':'record 1'}])
    result=summarize(p,c)
    assert result.sampled_runs.iloc[0]==4 and 0<=result.fraction_windows_above_own_station_threshold.iloc[0]<=1
    c.loc[0,'source_page_or_row']=''
    try:summarize(p,c)
    except ValueError:pass
    else:raise AssertionError('Missing location provenance was accepted')
    print('VERIFIED-COORDINATE MAP SELF-TEST: PASS')


def make_map(proxy_path,coords_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():raise FileExistsError(f'Refusing overwrite: {output}')
    proxy=pd.read_csv(proxy_path)
    coords=pd.read_csv(coords_path)
    data=summarize(proxy,coords)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.verified_map_',dir=output.parent) as scratch:
        stage=Path(scratch);data.to_csv(stage/'station_excursion_geolocation.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,7))
        valid=data.fraction_windows_above_own_station_threshold.notna()
        plotted=data.loc[valid]
        if len(plotted):
            sc=ax.scatter(plotted.longitude,plotted.latitude,c=plotted.fraction_windows_above_own_station_threshold,
              cmap='viridis',vmin=0,vmax=1,s=85+12*np.sqrt(plotted.sampled_runs),edgecolors='black')
            fig.colorbar(sc,ax=ax,label='Fraction of run windows above OWN station median + 1 MAD scale')
        if (~valid).any():
            ax.scatter(data.loc[~valid,'longitude'],data.loc[~valid,'latitude'],marker='x',s=90,label='MAD = 0; not estimable')
            ax.legend()
        for row in data.itertuples():
            ax.annotate(f'{row.station} (n={row.sampled_runs})',(row.longitude,row.latitude),xytext=(4,5),
                        textcoords='offset points',fontsize=7)
        ax.set_xlabel('Longitude (WGS84, degrees)');ax.set_ylabel('Latitude (WGS84, degrees)')
        centre=float(np.median(data.latitude))
        ax.set_aspect(1/max(.2,np.cos(np.deg2rad(centre))))
        ax.set_title('Verified-site relative acoustic excursions — development recordings only')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(stage/'verified_station_excursion_points.png',dpi=180)
        plt.close(fig)
        info={'scope':'Georeferenced points ONLY; training sample; NOT a pressure/vessel activity/interpolated field map',
              'threshold':'Within each station: median sampled run-window level + 1.4826 × MAD',
              'limitations':['Dates/deployments differ across sites and are not adjusted',
                 'Unlabelled high sound levels may reflect anything; do not label as vessel presence',
                 'Not a calibrated between-station acoustic level or pressure comparison',
                 'Points are not a spatial interpolation'],
              'input_hashes':{'proxy':sha(proxy_path),'coordinates':sha(coords_path)}}
        (stage/'map_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
        output.mkdir(exist_ok=False)
        for p in stage.iterdir():p.replace(output/p.name)
    print('VERIFIED STATION ACOUSTIC MAP: PASS; output:',output)
    print('CAUTION: A station-relative acoustic anomaly is not confirmed vessel activity.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--coordinates',type=Path,default=DEFAULT_COORD)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--self-test-only',action='store_true')
    args=p.parse_args();self_test()
    if not args.self_test_only:make_map(args.input,args.coordinates,args.output)
if __name__=='__main__':main()
