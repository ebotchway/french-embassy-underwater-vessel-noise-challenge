#!/usr/bin/env python3
"""Conservative disk cleanup and Zenodo VTUAD archive restoration.

Python standard library only. Run from repository root. No automatic removal of
source archives, extracted WAVs, validation partitions or scientific results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ZENODO_API = 'https://zenodo.org/api/records/22920817'  # Specific VTUAD v1.3.0 release
EXPECTED_BACKGROUND_MD5 = 'd28ad0fb6ccc98341ba4f7e953d9c68f'
EXPECTED_WAV_COUNTS = {'vessel': 509, 'background': 16}
EXPECTED_LOCAL_SHA256 = {
    'vtuad_strict_wav_vessel.zip': '96564721a917d8b4dd161fdb90c66db917136786e6f4680028be6c6e03cf8ac6',
    'vtuad_strict_wav_background.zip': 'a64aefd23ed3757dc8b4e909caf7a4040c1916e68ce4a714a835639b08e41761',
}
TEMPS = (
    'phase5_asset_fix_package', 'phase6_audit_package',
    'research_phase4_package', 'research_phase5_package',
    'requirements.before-final-freeze.txt',
)


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def size(path):
    if path.is_file(): return path.stat().st_size
    total = 0
    for base, _, files in os.walk(path, followlinks=False):
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                try: total += p.stat().st_size
                except OSError: pass
    return total


def human(n):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if n < 1024 or unit == 'TiB': return f'{n:.2f} {unit}'
        n /= 1024


def check_root(root):
    if not (root / 'requirements.txt').is_file() or not (root / 'scripts').is_dir():
        raise SystemExit('Not the expected repository root (need scripts/ and requirements.txt).')


def audit(root):
    print('Project:', root)
    for item in TEMPS:
        p = root / item
        print(f'TEMP   {item}: {human(size(p)) if p.exists() else "absent"}')
    d = root / 'data/external/vtuad/downloads'
    for name in EXPECTED_LOCAL_SHA256:
        p = d / name
        print(f'ARCHIVE {name}: {human(size(p)) if p.is_file() else "absent"}')
    for label, p in [('Zindi archive', root / 'data/zindi/raw/South_African_Real_Acoustics.zip'),
                     ('VTUAD data folder', root / 'data/external/vtuad'),
                     ('HearMyShip audio', root / 'data/external/hearmyship/audio')]:
        print(f'DATA   {label}: {human(size(p)) if p.exists() else "absent"}')
    print('Note: folder totals may include archives shown separately.')


def clean(root, apply):
    paths = [(name, root / name) for name in TEMPS if (root / name).exists()]
    tracked = subprocess.run(['git', 'ls-files', '--', *TEMPS], cwd=root,
                             capture_output=True, text=True, check=True)
    if tracked.stdout.strip():
        raise SystemExit('REFUSING CLEANUP: Git tracks some selected paths:\n' + tracked.stdout)
    for name, p in paths:
        if p.is_symlink() or p.resolve().parent != root:
            raise SystemExit(f'Unsafe path, refused: {p}')
        print(f'{"REMOVE" if apply else "WOULD REMOVE"}: {name} ({human(size(p))})')
    if not apply:
        print('DRY RUN: re-run with clean --apply to remove ONLY the paths listed above.')
        return
    for name, p in paths:
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
    print(f'Cleanup complete: {len(paths)} temporary items removed. All data/, reports/, scripts/, notebooks/, .venv/ untouched.')


def api_assets():
    req = urllib.request.Request(ZENODO_API, headers={'User-Agent': 'UnderwaterResearchRecovery/1.0'})
    with urllib.request.urlopen(req, timeout=45) as response:
        rec = json.load(response)
    assets = {}
    for f in rec.get('files', []):
        key = f.get('key') or f.get('filename')
        if not key: continue
        links = f.get('links') or {}
        url = links.get('self') or links.get('download')
        if url and url.startswith('https://') and 'zenodo.org/' in url:
            assets[key.rsplit('/', 1)[-1]] = {'url':url, 'checksum':f.get('checksum'), 'size':f.get('size')}
    return rec.get('id'), assets


def verify_published_file(p, checksum):
    if not checksum or ':' not in checksum:
        raise RuntimeError(f'No supported publisher checksum: {p.name}')
    algorithm, digest = checksum.split(':', 1)
    algorithm = algorithm.lower()
    if algorithm not in ('md5', 'sha256'):
        raise RuntimeError(f'Unsupported publisher checksum {algorithm}: {p.name}')
    if sha(p, algorithm) != digest.lower():
        raise RuntimeError(f'Publisher checksum mismatch: {p.name}')


def get_one(dest, asset, apply):
    expected_size = asset.get('size')
    expected_checksum = asset.get('checksum')
    if dest.exists():
        if not dest.is_file() or (expected_size is not None and dest.stat().st_size != expected_size):
            raise RuntimeError(f'Existing path has unexpected size/type: {dest}')
        verify_published_file(dest, expected_checksum)
        print('VERIFIED existing:', dest.name)
        return
    print(f'{"FETCH" if apply else "WOULD FETCH"}: {dest.name} ({human(expected_size or 0)})')
    if not apply: return
    free = shutil.disk_usage(dest.parent).free
    if expected_size and free < expected_size + 256 * 1024 * 1024:
        raise RuntimeError(f'Insufficient free disk for {dest.name}; available {human(free)}')
    stage = dest.with_name(dest.name + '.incomplete')
    if stage.exists():
        raise RuntimeError(f'Incomplete download exists ({stage}). Preserve and inspect it first.')
    req = urllib.request.Request(asset['url'], headers={'User-Agent': 'UnderwaterResearchRecovery/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=90) as response, stage.open('xb') as output:
            shutil.copyfileobj(response, output, length=4*1024*1024)
        if expected_size is not None and stage.stat().st_size != expected_size:
            raise RuntimeError(f'Length mismatch for {stage.name}')
        verify_published_file(stage, expected_checksum)
        stage.rename(dest)
        print('PUBLISHER CHECKSUM VERIFIED:', dest.name)
    except Exception:
        print(f'Download incomplete/invalid, staged at {stage} (not deleting automatically).')
        raise


def fetch_vtuad(root, apply):
    recid, assets = api_assets()
    if str(recid) != '22920817':
        raise RuntimeError(f'Unexpected Zenodo record ID {recid}')
    wanted = [f'vtuad_strict_wav_vessel.zip.part0{i}' for i in range(1,4)] + ['vtuad_strict_wav_background.zip']
    missing = [name for name in wanted if name not in assets]
    if missing: raise RuntimeError(f'Publisher record is missing assets {missing}; refusing to guess URLs')
    destdir = root / 'data/external/vtuad/downloads'
    if apply: destdir.mkdir(parents=True, exist_ok=True)
    elif not destdir.exists(): print('Would create:', destdir)
    for name in wanted:
        if destdir.exists(): get_one(destdir / name, assets[name], apply)
        else: print(f'WOULD FETCH {name}')
    bg = destdir / wanted[-1]
    if bg.is_file():
        verify_published_file(bg, assets[bg.name]['checksum'])
        if sha(bg, 'md5') != EXPECTED_BACKGROUND_MD5:
            raise RuntimeError('Background MD5 differs from known v1.3.0 fingerprint')
    vessel = destdir / 'vtuad_strict_wav_vessel.zip'
    if vessel.is_file():
        actual = sha(vessel)
        if actual != EXPECTED_LOCAL_SHA256[vessel.name]:
            raise RuntimeError('Existing vessel ZIP SHA256 differs from previously recorded local evidence')
        print('EXISTING VESSEL ZIP: matches prior LOCAL SHA256; this is not publisher authentication')
    elif all((destdir / n).exists() for n in wanted[:3]):
        print('Vessel parts obtained; reconstruction:', vessel)
        if apply:
            needed = sum((destdir/n).stat().st_size for n in wanted[:3])
            if shutil.disk_usage(destdir).free < needed + 256*1024*1024:
                raise RuntimeError('Insufficient disk to reconstruct vessel ZIP; retain split parts.')
            stage = vessel.with_name(vessel.name + '.building')
            if stage.exists(): raise RuntimeError(f'Existing build temp file: {stage}')
            with stage.open('xb') as output:
                for name in wanted[:3]:
                    with (destdir/name).open('rb') as part:
                        shutil.copyfileobj(part, output, 4*1024*1024)
            if sha(stage) != EXPECTED_LOCAL_SHA256[vessel.name]:
                raise RuntimeError('Reconstructed vessel SHA256 differs from previously verified local fingerprint; inspect parts')
            with zipfile.ZipFile(stage) as z:
                if z.testzip() is not None: raise RuntimeError('Reconstructed ZIP CRC failed')
                n = sum(x.filename.lower().endswith('.wav') for x in z.infolist())
                if n != EXPECTED_WAV_COUNTS['vessel']:
                    raise RuntimeError(f'Unexpected vessel clip count {n}')
            stage.rename(vessel)
            print('REBUILT VESSEL ZIP: matches prior local SHA256; ZIP CRC passed; publisher PART checksums verified')
    if not apply: print('DRY RUN: use fetch-vtuad --apply when redownload is actually necessary.')
    print('No extraction performed. Existing extracted WAVs and original research artifacts remain untouched.')
    print('Zenodo page: https://zenodo.org/records/22920817')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('audit')
    c = sub.add_parser('clean', help='Preview deletion of five known disposable files/directories')
    c.add_argument('--apply', action='store_true')
    f = sub.add_parser('fetch-vtuad', help='Recover original VTUAD ZIP assets using Zenodo record API')
    f.add_argument('--apply', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    check_root(root)
    try:
        if args.command == 'audit': audit(root)
        elif args.command == 'clean': clean(root, args.apply)
        else: fetch_vtuad(root, args.apply)
    except Exception as exc:
        raise SystemExit(f'ERROR (nothing silently overwritten or deleted): {exc}') from exc

if __name__ == '__main__': main()
