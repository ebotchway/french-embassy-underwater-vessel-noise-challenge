#!/usr/bin/env python3
"""Read-only spot validation of indexed 15x25 acoustic windows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def retrieve(path, positions, columns):
    pf = pq.ParquetFile(path)
    if any(p < 0 or p >= pf.metadata.num_rows for p in positions):
        raise ValueError('Index contains out-of-range row positions')
    groups = []
    offset = 0
    for i in range(pf.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        selected = [(j, p - offset) for j, p in enumerate(positions) if offset <= p < offset+n]
        if selected:
            frame = pf.read_row_group(i, columns=columns).to_pandas()
            for j, local in selected:
                groups.append((j, frame.iloc[local]))
        offset += n
    return pd.DataFrame([v for _, v in sorted(groups)]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='reports/eda/background_window_index.csv')
    ap.add_argument('--per-station', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    index = pd.read_csv(args.index, dtype={'run_id':'string'})
    if index.empty:
        raise SystemExit('Index is empty')
    choices = pd.concat([
        g.sample(n=min(args.per_station, len(g)), random_state=args.seed)
        for _, g in index.groupby('station', sort=True)
    ], ignore_index=True)
    results=[]
    for _, r in choices.iterrows():
        path = Path(r.source_parquet)
        if not path.exists():
            raise FileNotFoundError(f'Indexed source missing: {path}')
        positions = json.loads(r.row_positions_json)
        pf = pq.ParquetFile(path)
        band_cols = [c for c in pf.schema_arrow.names if c.startswith('TOL_')]
        # Prefer the exact competition band mapping, read only header, never test values.
        header = pd.read_csv('data/zindi/raw/Test.csv', nrows=0).columns
        ordered = []
        for col in header:
            if '_TOL_' in col:
                band = 'TOL_' + col.split('_TOL_',1)[1]
                if band not in ordered:
                    ordered.append(band)
        if len(ordered) != 25 or not set(ordered).issubset(band_cols):
            raise ValueError('Could not match exactly 25 test-schema bands to archive')
        cols = ['recorder','t','dur']+ordered
        frame = retrieve(path,positions,cols)
        t = pd.to_datetime(frame['t'],errors='coerce')
        durations=pd.to_numeric(frame['dur'],errors='coerce')
        values=frame[ordered].to_numpy(dtype=float)
        diffs=t.diff().dt.total_seconds().dropna()
        checks={
            '15_rows':len(frame)==15,
            'recorder_matches':frame['recorder'].astype(str).eq(str(r.recorder)).all(),
            'start_matches':str(t.iloc[0])==str(pd.Timestamp(r.start_time)),
            'last_matches':str(t.iloc[-1])==str(pd.Timestamp(r.last_start_time)),
            'minute_steps_58_62':diffs.between(58,62).all(),
            'durations_55_65':durations.between(55,65).all(),
            'finite_25_bands':np.isfinite(values).all(),
        }
        results.append({
            'station':r.station, 'run_id':r.run_id,
            'passed':all(checks.values()),
            'min_db':float(np.nanmin(values)), 'max_db':float(np.nanmax(values)),
            'median_db':float(np.nanmedian(values)),
            **checks,
        })
    output=pd.DataFrame(results)
    print(output[['station','passed','min_db','median_db','max_db']].to_string(index=False))
    print(f'Passed: {output.passed.sum()}/{len(output)}')
    if not output.passed.all():
        print('\nFAILED DETAILS:\n',output.loc[~output.passed].to_string(index=False))
        raise SystemExit(1)
    print('Spot-check passed. This does NOT establish calibration or QC validity.')

if __name__=='__main__':
    main()
