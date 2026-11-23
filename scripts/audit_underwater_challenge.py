"""Reproducible audit of Zindi Underwater Vessel Noise Challenge files."""
import argparse, io, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('acoustic_audit'))
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    root=args.data_dir
    test=pd.read_csv(root.parent/'raw'/'Test.csv',dtype={'ID':'string'})
    sub=pd.read_csv(root.parent/'raw'/'sample_submission.csv',dtype={'ID':'string'})
    feature=[c for c in test if c!='ID']
    assert len(feature)==375 and test.shape[0]==1500
    assert test.ID.is_unique and sub.ID.is_unique and set(test.ID)==set(sub.ID)
    arr=test[feature].to_numpy(dtype=np.float64).reshape(-1,15,25)
    missing=np.isnan(arr)
    minute_missing=missing.all(axis=2)
    partial=(missing.any(axis=2)&~minute_missing)
    print('Test shapes:',arr.shape,'Missing cells:',missing.sum(),'Partial-band missing minute rows:',partial.sum())
    summary=pd.DataFrame({'ID':test.ID,'missing_cells':missing.sum(axis=(1,2)),'missing_minutes':minute_missing.sum(axis=1),
          'observed_dB_mean':np.nanmean(arr,axis=(1,2)),'observed_dB_median':np.nanmedian(arr,axis=(1,2)),
          'observed_dB_std':np.nanstd(arr,axis=(1,2))})
    summary.to_csv(args.output_dir/'test_inference_only_diagnostics.csv',index=False)
    archive=root.parent/'raw'/'South_African_Real_Acoustics.zip'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        stations=pd.read_csv(io.BytesIO(z.read('stations.csv')))
        recorders=pd.read_csv(io.BytesIO(z.read('recorders.csv')))
        stations.to_csv(args.output_dir/'stations_inventory.csv',index=False)
        recorders.to_csv(args.output_dir/'recorder_inventory.csv',index=False)
        rows=[]
        for name in names:
            if not name.endswith('.parquet'): continue
            try: d=pd.read_parquet(io.BytesIO(z.read(name)))
            except ImportError as e: raise RuntimeError('Install pyarrow: python -m pip install pyarrow') from e
            bands=[c for c in d if c.startswith('TOL_')]
            durations=pd.to_numeric(d['dur'],errors='coerce')
            rows.append(dict(file=name,rows=len(d),columns=len(d.columns),band_columns=len(bands),
                first_timestamp=str(d['t'].min()),last_timestamp=str(d['t'].max()),
                pct_missing_bands=round(100*d[bands].isna().to_numpy().mean(),3),
                rows_duration_under55=int((durations<55).sum()),rows_duration_not60=int((durations!=60).sum()),
                recorders=d['recorder'].nunique()))
        pd.DataFrame(rows).to_csv(args.output_dir/'station_acoustic_audit.csv',index=False)
        print('Total acoustic records:',sum(x['rows'] for x in rows))
    print('Created audit files in',args.output_dir.resolve())

if __name__=='__main__': main()
