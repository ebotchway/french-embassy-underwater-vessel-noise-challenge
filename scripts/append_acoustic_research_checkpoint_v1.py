#!/usr/bin/env python3
"""Append a sourced scientific checkpoint without replacing or rerunning notebooks.

Reads the locally assembled manuscript and pre-existing notebook/learning log.
Appends one Markdown research cell and one learning-log section. Preserves all
existing notebook cells and stored outputs exactly. Creates backups before edit.
Refuses duplicate checkpoints or incomplete manuscript inputs.

  python scripts/append_acoustic_research_checkpoint_v1.py --dry-run
  python scripts/append_acoustic_research_checkpoint_v1.py --apply
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd

TAG='ACOUSTIC_SCIENCE_CHECKPOINT_2026_11_22'
DEFAULT_NOTEBOOK=Path('notebooks/01_south_african_acoustic_eda.ipynb')
DEFAULT_LOG=Path('PROJECT_LEARNING_LOG.md')
DEFAULT_REPORT=Path('reports/research_manuscript_v1')
DEFAULT_BACKUP=Path('reports/research_backups')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_text(report):
    manifest=json.loads((report/'MANUSCRIPT_MANIFEST.json').read_text())
    stats=pd.read_csv(report/'station_relative_anomaly_proxy.csv')
    station_counts=stats.groupby('station').run_id.nunique().to_dict()
    has_realism=bool(manifest.get('simulation_results_included',False))
    lines=[f'## Acoustic-science checkpoint ({TAG})','',
        'This checkpoint consolidates the training-only real-acoustic evidence and the group-separated synthetic baselines. It is not independent evidence that natural vessel presence or calibrated sound pressure was recovered.',
        '',f'**Development run coverage:** {station_counts}.',
        '', '**Experiment 004 frozen synthetic evaluation:** original representation fraction RMSE 0.114430, detector ROC AUC 0.981296; submitted maximum-shift representation fraction RMSE 0.150893, detector ROC AUC 0.978653 (n=300). Training/evaluation runs and vessel hulls are group-separated for this new experiment but historically studied source pools are reused.',
        '', '**Provisional public leaderboard (Submission 001):** score 0.761535762, derived share RMSE approximately 0.238464, diagnostic AUC 0.517533. The private leaderboard determines final ranking; no hidden labels were available.',
        '', '**Real South African recordings:** 460 sampled development runs across six sites, one 15-minute window per run up to the station cap; 303 vessel-associated clips from 109 hulls. Levels are archive representations, not independently calibrated underwater pressure.',
        '', '**Spectral-shape comparison:** 66 station/class comparisons over 25 bands. Station median differences ranged from 7.477 dB (St Francis) to 26.202 dB (Durban, one sampled train run). These are separately peak-referenced shapes; no pressure, source-identification or vessel-detection inference is justified.',
        '', '**Controlled simulation realism:** '+('training-only fixed-share source-spectrum sensitivity comparison completed; inspect `reports/audits/simulation_realism_v1/scenario_summary.csv` and the manuscript for measured changes.' if has_realism else 'not yet run; the corresponding manuscript section remains pending.'),
        '', '**Scientific report:** `reports/research_manuscript_v1/SCIENTIFIC_MANUSCRIPT_DRAFT.md` and `SUBMISSION_READINESS_CHECKLIST.md`. The draft requires verified citations, VTUAD external-data declaration, true station-coordinate provenance, coverage-aware relative acoustic mapping, and final PDF preparation.',
        '', 'No frozen evaluation or sealed station holdout acoustic measurements were used to make the real-acoustic findings or tune the simulator. No additional public submission was generated.','']
    return '\n'.join(lines)


def run(args):
    if not args.report.joinpath('SCIENTIFIC_MANUSCRIPT_DRAFT.md').exists():
        raise FileNotFoundError('Run manuscript assembler before applying checkpoint')
    for p in [args.notebook,args.log]:
        if not p.is_file():raise FileNotFoundError(p)
    import nbformat
    nb=nbformat.read(args.notebook,as_version=4)
    nbformat.validate(nb)
    current_log=args.log.read_text()
    existing='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    if TAG in current_log or TAG in existing:
        raise FileExistsError('Checkpoint already present; refusing duplicate insertion')
    original_sources=[(c.cell_type,c.source) for c in nb.cells]
    original_outputs=[(i,json.dumps(c.get('outputs',[]),sort_keys=True,ensure_ascii=False)) for i,c in enumerate(nb.cells) if c.cell_type=='code']
    addition=record_text(args.report)
    nb.cells.append(nbformat.v4.new_markdown_cell(addition))
    nbformat.validate(nb)
    assert [(c.cell_type,c.source) for c in nb.cells[:-1]]==original_sources
    assert [(i,json.dumps(c.get('outputs',[]),sort_keys=True,ensure_ascii=False)) for i,c in enumerate(nb.cells[:-1]) if c.cell_type=='code']==original_outputs
    if not args.apply:
        print('CHECKPOINT DRY RUN: PASS')
        print('Would append one markdown cell; existing source cells and code outputs unchanged.')
        print('Would append learning-log section and back up both files.')
        print(addition[:750]+'...')
        return
    args.backup.mkdir(parents=True,exist_ok=True)
    backup_nb=args.backup/f'{args.notebook.stem}_before_{TAG}_{sha(args.notebook)[:12]}.ipynb'
    backup_log=args.backup/f'{args.log.stem}_before_{TAG}_{sha(args.log)[:12]}.md'
    for path in (backup_nb,backup_log):
        if path.exists():raise FileExistsError(f'Backup path already exists: {path}')
    shutil.copy2(args.notebook,backup_nb);shutil.copy2(args.log,backup_log)
    # Prepare both files before replacing either.
    with tempfile.TemporaryDirectory(prefix='.acoustic_checkpoint_',dir=args.notebook.parent) as tmp:
        a=Path(tmp)/'notebook.ipynb';b=Path(tmp)/'log.md'
        nbformat.write(nb,a)
        b.write_text(current_log.rstrip()+'\n\n'+addition+'\n')
        nbformat.validate(nbformat.read(a,as_version=4))
        a.replace(args.notebook);b.replace(args.log)
    print('DOCUMENT CHECKPOINT: PASS')
    print('Added 1 notebook markdown cell; existing code/outputs preserved; log section appended.')
    print('Backups:',backup_nb,backup_log)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--notebook',type=Path,default=DEFAULT_NOTEBOOK)
    p.add_argument('--log',type=Path,default=DEFAULT_LOG)
    p.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--backup',type=Path,default=DEFAULT_BACKUP)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--apply',action='store_true')
    args=p.parse_args();run(args)
if __name__=='__main__':main()
