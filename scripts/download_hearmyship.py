#!/usr/bin/env python3
"""Download HearMyShip audio referenced by an exported vessels.json."""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin
import requests

ALLOWED = {'https://hearmyship.fer.hr'}

def make_url(relative, base):
    if not isinstance(relative, str) or not relative.lower().endswith('.wav'):
        raise ValueError('Not a WAV path')
    pieces = relative.replace('\\', '/').lstrip('/').split('/')
    if any(p in ('', '.', '..') for p in pieces):
        raise ValueError('Invalid path components')
    return urljoin(base.rstrip('/')+'/', '/'.join(quote(p, safe='') for p in pieces))

def download(session, url, dest, tries=3):
    for attempt in range(tries):
        try:
            with session.get(url, stream=True, timeout=(12, 90), allow_redirects=True) as r:
                if r.status_code in (403,404):
                    return 'HTTP '+str(r.status_code), 0
                if r.status_code == 429 or r.status_code >= 500:
                    delay = min(60, float(r.headers.get('Retry-After','0') or 0) or (2**attempt * 3))
                    if attempt < tries-1:
                        time.sleep(delay); continue
                r.raise_for_status()
                if r.url.split('/')[2] != 'hearmyship.fer.hr':
                    return 'Redirected outside authorized host', 0
                tmp = dest.with_suffix('.wav.partial')
                size=0
                with tmp.open('wb') as f:
                    head = next(r.iter_content(chunk_size=1024*256), b'')
                    if not (head.startswith(b'RIFF') and head[8:12]==b'WAVE') and not (head.startswith(b'RF64') and head[8:12]==b'WAVE'):
                        tmp.unlink(missing_ok=True)
                        return 'Not WAV (check download URL)', 0
                    f.write(head);size+=len(head)
                    for part in r.iter_content(chunk_size=1024*256):
                        if part:
                            f.write(part);size+=len(part)
                tmp.replace(dest)
                return 'OK',size
        except (requests.RequestException, OSError) as e:
            error = str(e)[:160]
            if attempt < tries-1:
                time.sleep(2**attempt * 2)
    return 'ERROR '+error,0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json',default='vessels.json',help='Downloaded HearMyShip vessels.json')
    p.add_argument('--output',default='hearmyship_audio')
    p.add_argument('--base-url',default='https://hearmyship.fer.hr/', help='Verified file server URL; default tries the site origin')
    p.add_argument('--limit',type=int,default=1,help='Number of files to try; default 1 for safety')
    p.add_argument('--all',action='store_true',help='Download every WAV entry')
    p.add_argument('--delay',type=float,default=1.5,help='Minimum seconds between requests')
    args=p.parse_args()
    if args.delay < 1: p.error('Use delay >= 1 second to avoid excessive traffic')
    if not args.base_url.startswith('https://hearmyship.fer.hr/') and args.base_url != 'https://hearmyship.fer.hr':
        p.error('Only the HearMyShip domain is allowed by this helper')
    records=json.loads(Path(args.json).read_text(encoding='utf-8'))
    if not isinstance(records,list):p.error('Expected a JSON list')
    valid=[r for r in records if isinstance(r,dict) and isinstance(r.get('sound'),str) and r['sound'].lower().endswith('.wav')]
    chosen=valid if args.all else valid[:args.limit]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'download_manifest.csv'
    print(f'{len(valid)} WAV entries in JSON; processing {len(chosen)}. Base: {args.base_url}',flush=True)
    with requests.Session() as session, manifest.open('a',newline='',encoding='utf-8') as fp:
        session.headers.update({'User-Agent':'Research audio downloader (contact: see HearMyShip site); Python requests','Accept':'audio/wav,audio/*;q=0.9,*/*;q=0.5'})
        writer=csv.writer(fp)
        if manifest.stat().st_size==0:writer.writerow(['id','category','relative_sound','url','local_path','status','bytes'])
        for i,r in enumerate(chosen,1):
            try:
                url=make_url(r['sound'],args.base_url)
                rel=r['sound'].replace('\\','/').lstrip('/')
                dest=out/rel
                # directory components are validated by make_url
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.is_file() and dest.stat().st_size>44:
                    with dest.open('rb') as f: head=f.read(12)
                    if (head[:4] in (b'RIFF',b'RF64')) and head[8:12]==b'WAVE':
                        status='EXISTS';size=dest.stat().st_size
                    else:status,size=download(session,url,dest)
                else:status,size=download(session,url,dest)
            except Exception as e:
                url='';dest=out;status='ERROR '+str(e)[:160];size=0
            writer.writerow([r.get('id'),r.get('category'),r.get('sound'),url,str(dest),status,size]);fp.flush()
            print(f'[{i}/{len(chosen)}] ID={r.get("id")} {status} {size} bytes',flush=True)
            if i==1 and status!='OK' and status!='EXISTS':
                print('First file failed. Stopping; verify base URL from a WAV download in your browser.',flush=True)
                break
            if i<len(chosen):time.sleep(args.delay + random.uniform(0,.5))
    print('Manifest:',manifest)
if __name__=='__main__':main()
