#!/usr/bin/env python3
"""Index conservative 15-observation acoustic windows; never reads test data or acoustic levels.

Rows within a Parquet file are referenced by their zero-based physical row positions.
This is a candidate index, not a calibration or waveform-quality certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('data/zindi/extracted'))
    ap.add_argument('--output', type=Path, default=Path('reports/eda/background_window_index.csv'))
    args = ap.parse_args()
    files = sorted(args.data_dir.rglob('*.parquet'))
    if not files:
        raise SystemExit(f'No Parquet files under {args.data_dir}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_windows = []
    for file in files:
        table = pq.read_table(file, columns=['recorder', 't', 'dur'])
        df = table.to_pandas()
        df['row_position'] = np.arange(len(df), dtype=np.int64)
        df['t'] = pd.to_datetime(df['t'], errors='coerce')
        df['dur'] = pd.to_numeric(df['dur'], errors='coerce')
        count_before = len(all_windows)
        for recorder, g in df.groupby('recorder', dropna=False, sort=False):
            g = g.sort_values(['t', 'row_position'], kind='mergesort').reset_index(drop=True)
            valid = g['t'].notna() & g['dur'].between(55, 65)
            duplicate = g['t'].duplicated(keep=False)
            valid &= ~duplicate
            dt = g['t'].diff().dt.total_seconds()
            # These are screening assumptions, matching the prior notebook.
            continues = dt.between(58, 62) & valid & valid.shift(fill_value=False)
            run_id = (~continues).cumsum()
            good = g.loc[valid].copy()
            good['run_id'] = run_id.loc[valid].to_numpy()
            for run_number, run in good.groupby('run_id', sort=False):
                n = len(run)
                if n < 15:
                    continue
                # Separate complete 15-row blocks. No overlap within runs.
                for offset in range(0, n - 14, 15):
                    w = run.iloc[offset:offset + 15]
                    positions = [int(v) for v in w['row_position'].to_numpy()]
                    all_windows.append({
                        'station': file.stem,
                        'recorder': str(recorder),
                        'source_parquet': str(file),
                        'run_id': f'{file.stem}|{recorder}|{int(run_number)}',
                        'start_time': w['t'].iloc[0].isoformat(),
                        'last_start_time': w['t'].iloc[-1].isoformat(),
                        'row_positions_json': json.dumps(positions, separators=(',', ':')),
                    })
        print(f'{file.stem}: {len(all_windows) - count_before:,} windows')
    out = pd.DataFrame(all_windows)
    if len(out):
        assert out['row_positions_json'].map(lambda x: len(json.loads(x)) == 15).all()
        assert not out.duplicated(['source_parquet', 'row_positions_json']).any()
    out.to_csv(args.output, index=False)
    print(f'Wrote {len(out):,} indexed windows to {args.output}')
    print('Index only: acoustic values, source calibration and QC flags have NOT been validated.')


if __name__ == '__main__':
    main()
