#!/usr/bin/env python3
"""Safely organize a flat underwater-vessel challenge folder (stdlib only).
Usage: python scripts/setup_underwater_project.py --root . --apply
Default is a dry run. Re-running is supported; source files are never overwritten.
"""
import argparse, csv, hashlib, json, re, shutil, sys, zipfile
from pathlib import Path
from datetime import datetime, timezone

DIRS = ['data/zindi/raw','data/zindi/extracted','data/external/vtuad/downloads',
 'data/external/vtuad/metadata','data/external/vtuad/extracted',
 'data/external/hearmyship/metadata','data/external/hearmyship/audio',
 'data/external/onc','data/processed','data/synthetic','scripts','notebooks',
 'models','experiments','submissions','reports','logs']
ZINDI = {'south_african_real_acoustics.zip','test.csv','sample_submission.csv','maritime_starter_notebook.ipynb'}
CODE = {'download_hearmyship.py','audit_underwater_challenge.py','setup_underwater_project.py'}
VTUAD_META = ('metadata','registry','clip_acoustic','dataset_card','citation','environment','sha256','readme')

def destination(name):
    n=name.lower()
    if n in ZINDI: return 'data/zindi/raw'
    if n=='vessels.json': return 'data/external/hearmyship/metadata'
    if n in CODE: return 'scripts'
    if n.startswith('vtuad_') and ('.zip' in n or n.endswith('.wav')): return 'data/external/vtuad/downloads'
    if n.startswith('vtuad_') or any(n.startswith(x) for x in VTUAD_META): return 'data/external/vtuad/metadata'
    return None

def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def zip_check(path):
    try:
        with zipfile.ZipFile(path) as z:
            problem=z.testzip()
            if problem:return f'FAIL CRC: {problem}'
            return f'OK ({len(z.infolist())} entries)'
    except (OSError,zipfile.BadZipFile,RuntimeError) as exc:return f'FAIL ({exc})'

def safe_extract(path, dest):
    dest=dest.resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name=info.filename
            out=(dest/name).resolve()
            if not out.is_relative_to(dest): raise ValueError(f'Unsafe archive member: {name}')
            mode=(info.external_attr >> 16) & 0o170000
            if mode == 0o120000:raise ValueError(f'Symlink blocked: {name}')
            if out.exists() and out.is_file() and out.stat().st_size != info.file_size:
                raise ValueError(f'Existing extracted file has unexpected size: {out}')
        # Zip entries verified by testzip earlier. Do not overwrite extracted files.
        for info in z.infolist():
            out=(dest/info.filename).resolve()
            if info.is_dir():out.mkdir(parents=True,exist_ok=True);continue
            if out.exists():continue
            out.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src, out.open('xb') as target:shutil.copyfileobj(src,target,1024*1024)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',default='.',help='Project root, default current directory')
    a.add_argument('--apply',action='store_true',help='Actually move files; otherwise preview only')
    a.add_argument('--extract-vtuad',action='store_true',help='Also extract validated VTUAD ZIP files (may need many GB)')
    a.add_argument('--skip-zindi-extract',action='store_true')
    args=a.parse_args();root=Path(args.root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    print('ROOT:',root,'MODE:','APPLY' if args.apply else 'PREVIEW')
    for rel in DIRS:
        if args.apply:(root/rel).mkdir(parents=True,exist_ok=True)
    actions=[];warnings=[]
    for source in sorted(root.iterdir(),key=lambda p:p.name.lower()):
        if not source.is_file() or source.is_symlink() or source.name.startswith('.') :continue
        folder=destination(source.name)
        if folder is None:
            warnings.append(f'Unrecognized root file left alone: {source.name}');continue
        target=root/folder/source.name
        if source.resolve()==target.resolve():continue
        if target.exists():
            if source.stat().st_size==target.stat().st_size and hash_file(source)==hash_file(target):
                warnings.append(f'Duplicate retained at root (identical): {source.name}')
            else:warnings.append(f'CONFLICT, not moved: {source.name} -> {target}')
            continue
        actions.append((source,target))
        print('MOVE',source.name,'->',target.relative_to(root))
        if args.apply:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
    if not args.apply:
        print('\nPREVIEW ONLY. Use --apply to move files and run checks.');return
    # Reconstruct split ZIPs; supports names ending .zip.part01 etc and arbitrary part counts.
    downloads=root/'data/external/vtuad/downloads'
    grouped={}
    for file in downloads.iterdir():
        m=re.match(r'^(.*\.zip)\.part(\d+)$',file.name,re.I)
        if m:grouped.setdefault(m.group(1),[]).append((int(m.group(2)),file))
    for name,parts in sorted(grouped.items()):
        parts.sort(); nums=[n for n,_ in parts]
        out=downloads/name
        if nums != list(range(1,nums[-1]+1)) or len(nums)<2:
            warnings.append(f'Incomplete split archive {name}: parts found {nums}; left untouched');continue
        if not out.exists():
            # Don't guess expected last part; for known VTUAD vessel archive require three.
            if name=='vtuad_strict_wav_vessel.zip' and nums != [1,2,3]:
                warnings.append(f'Expected vessel parts [1,2,3], found {nums}; not joining');continue
            print('JOIN',name,'from',len(parts),'parts')
            temp=downloads/(name+'.joining')
            try:
                with temp.open('wb') as dst:
                    for _,part in parts:
                        with part.open('rb') as src:shutil.copyfileobj(src,dst,1024*1024)
                with zipfile.ZipFile(temp) as z:
                    bad=z.testzip()
                    if bad:raise ValueError(f'CRC error in {bad}')
                temp.replace(out)
            except Exception as exc:
                temp.unlink(missing_ok=True)
                warnings.append(f'Failed to join {name}: {exc}')
        else:print('ALREADY JOINED',name)
    report=[]
    paths=[root/'data/zindi/raw/South_African_Real_Acoustics.zip']+sorted(downloads.glob('*.zip'))
    for path in paths:
        if path.exists():
            status=zip_check(path)
            print('ZIP',path.relative_to(root),status)
            report.append((str(path.relative_to(root)),path.stat().st_size,status,hash_file(path)))
            if status.startswith('OK'):
                if path.name=='South_African_Real_Acoustics.zip' and not args.skip_zindi_extract:
                    print('EXTRACT ZINDI',path.name)
                    safe_extract(path,root/'data/zindi/extracted')
                elif path.parent==downloads and args.extract_vtuad:
                    print('EXTRACT VTUAD',path.name)
                    safe_extract(path,root/'data/external/vtuad/extracted'/path.stem)
            else:warnings.append(f'ZIP integrity failure: {path.name}: {status}')
    # Check metadata JSON and competition CSV structure without using Test features for training.
    j=root/'data/external/hearmyship/metadata/vessels.json'
    if j.exists():
        try:
            records=json.loads(j.read_text(encoding='utf-8'))
            paths=[v.get('sound') for v in records if isinstance(v,dict) and isinstance(v.get('sound'),str)]
            print('HEARMYSHIP:',len(records),'records,',len(set(paths)),'unique sound paths')
        except (ValueError,OSError) as e:warnings.append('HearMyShip JSON invalid: '+str(e))
    test=root/'data/zindi/raw/Test.csv';sub=root/'data/zindi/raw/sample_submission.csv'
    for path in (test,sub):
        if path.exists():
            with path.open(newline='',encoding='utf-8-sig') as f:
                reader=csv.reader(f);header=next(reader);count=sum(1 for _ in reader)
            print('CSV',path.name,':',count,'rows,',len(header),'columns')
    logs=root/'logs';logs.mkdir(exist_ok=True)
    with (logs/'archive_integrity.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['archive','bytes','integrity','sha256']);w.writerows(report)
    (logs/'setup_warnings.txt').write_text('\n'.join(warnings)+'\n',encoding='utf-8')
    print('\nSETUP COMPLETE; warnings:',len(warnings))
    for w in warnings:print('WARNING:',w)
    print('Integrity report:',logs/'archive_integrity.csv')
    print('HearMyShip downloader:',root/'scripts/download_hearmyship.py')

if __name__=='__main__':main()
