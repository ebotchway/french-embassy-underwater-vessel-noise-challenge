#!/usr/bin/env python3
"""Read-only verification of the generated South African acoustic appendix."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED = {'figures/station_run_coverage.png','figures/station_run_levels_intervals.png',
 'figures/station_temporal_variability.png','figures/station_relative_spectral_shapes.png',
 'figures/station_relative_deviations.png','validated_station_summary.csv',
 'run_bootstrap_intervals.csv','REAL_ACOUSTIC_RESULTS_APPENDIX.md',
 'REAL_ACOUSTIC_RESULTS_APPENDIX.pdf'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()

def verify(root,out):
 manifest=json.loads((out/'manifest.json').read_text())
 assert manifest['analysis']=='real_acoustic_results_v1','wrong analysis'
 assert set(manifest['outputs_sha256'])==EXPECTED,'missing/extra outputs'
 for rel,digest in manifest['outputs_sha256'].items():
  path=out/rel
  assert path.is_file() and sha(path)==digest,f'output failed SHA check: {rel}'
 for key,rel in manifest['source_paths'].items():
  src=root/rel
  assert src.is_file() and sha(src)==manifest['source_sha256'][key],f'source changed: {rel}'
 st=pd.read_csv(out/'validated_station_summary.csv')
 ci=pd.read_csv(out/'run_bootstrap_intervals.csv')
 assert len(st)==manifest['input_checks']['stations']
 assert int(st.sampled_runs.sum())==manifest['input_checks']['runs']
 assert ci.groupby('station').size().eq(3).all()
 assert set(st.station)==set(ci.station)
 assert (ci.n_distinct_runs>0).all()
 assert ci[['median','p10','p90']].apply(lambda x: np.isfinite(x)).all().all()
 assert ci.loc[ci.n_distinct_runs<10,['run_bootstrap_low','run_bootstrap_high']].isna().all().all()
 assert (out/'REAL_ACOUSTIC_RESULTS_APPENDIX.pdf').read_bytes().startswith(b'%PDF-')
 print('REAL ACOUSTIC APPENDIX INTEGRITY: PASS')
 print(f"Stations: {len(st)}; run entries: {int(st.sampled_runs.sum())}; verified output artifacts: {len(EXPECTED)}")
 print('SHA-256 source and output checks: PASS; PDF header: PASS; run interval safeguards: PASS')
 print('LIMIT: Integrity is not pressure calibration, coordinate verification, or vessel identification.')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path('.'))
 p.add_argument('--output',type=Path,default=Path('reports/research_manuscript_v1/real_acoustic_results_v1'))
 a=p.parse_args();verify(a.root,a.output)
if __name__=='__main__':main()
