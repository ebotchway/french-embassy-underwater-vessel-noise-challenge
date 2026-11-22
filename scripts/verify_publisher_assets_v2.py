#!/usr/bin/env python3
"""Optional checksum verification against researcher-supplied authoritative publisher records.

CSV columns: asset_name,local_relative_path,checksum_algorithm,publisher_checksum,publisher_record_url
Accepted algorithms: sha256,md5. Verification is per asset, not certification of the full dataset.
Never download, overwrite, or publish source files.
"""
import argparse,csv,hashlib,json,tempfile
from pathlib import Path

def hash_file(path,algorithm):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(root,manifest,output):
    # Canonicalize both sides of the containment check. macOS /var may alias /private/var.
    root=Path(root).resolve(strict=True)
    manifest=Path(manifest)
    output=Path(output)
    if output.exists():raise FileExistsError(f'Report exists: {output}')
    if not manifest.is_file():raise FileNotFoundError(manifest)
    with manifest.open(newline='',encoding='utf-8-sig') as f:
        rdr=csv.DictReader(f)
        required={'asset_name','local_relative_path','checksum_algorithm','publisher_checksum','publisher_record_url'}
        if not required.issubset(rdr.fieldnames or []):raise ValueError('CSV missing mandatory fields')
        data=list(rdr)
    if not data:raise ValueError('Empty asset manifest')
    records=[]
    for r in data:
        alg=r['checksum_algorithm'].strip().lower()
        if alg not in ('sha256','md5'):raise ValueError('Unsupported algorithm: '+alg)
        expected=r['publisher_checksum'].strip().lower()
        if len(expected)!=(64 if alg=='sha256' else 32) or any(ch not in '0123456789abcdef' for ch in expected):
            raise ValueError('Malformed publisher checksum for '+r['asset_name'])
        url=r['publisher_record_url'].strip()
        if not url.startswith('https://'):raise ValueError('Publisher record must use HTTPS')
        rel=Path(r['local_relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Local path must be relative and confined to project root')
        fp=(root/rel).resolve()
        if not fp.is_relative_to(root):raise ValueError('Path outside project root')
        if not fp.is_file(): status='LOCAL_FILE_MISSING';actual=''
        else:
            actual=hash_file(fp,alg)
            status='CHECKSUM_MATCH_ASSET_ONLY' if actual==expected else 'CHECKSUM_MISMATCH'
        records.append(dict(asset_name=r['asset_name'],local_relative_path=str(rel),checksum_algorithm=alg,
            publisher_record_url=url,expected_checksum=expected,actual_checksum=actual,status=status,
            validation_limit='Supplied checksum provenance must be independently reviewed; does not prove full version or distribution identity'))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',prefix='.asset_checks_',suffix='.json',dir=output.parent,delete=False) as f:
        json.dump({'overall_status':'ASSET_COMPARISONS_ONLY_NOT_FULL_RELEASE_CERTIFICATION',
            'asset_count':len(records),'results':records},f,indent=2);f.write('\n');tmp=Path(f.name)
    tmp.rename(output)
    for r in records: print(r['status'],r['asset_name'])
    print('Saved:',output)
    if any(r['status']=='CHECKSUM_MISMATCH' for r in records):raise SystemExit(2)

def selftest():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/'a.zip').write_bytes(b'fixture')
        h=hashlib.sha256(b'fixture').hexdigest()
        manifest=root/'publisher_assets.csv'
        manifest.write_text('asset_name,local_relative_path,checksum_algorithm,publisher_checksum,publisher_record_url\n'
            +f'fixture,a.zip,sha256,{h},https://zenodo.org/records/example\n')
        o=root/'results.json';verify(root,manifest,o)
        assert json.loads(o.read_text())['results'][0]['status']=='CHECKSUM_MATCH_ASSET_ONLY'
        try:verify(root,manifest,o)
        except FileExistsError:pass
        else:raise AssertionError('Overwrite must be blocked')
        # A symlink alias of the root must work; a symlink escape must be rejected.
        alias=root/'root_alias'
        alias.symlink_to(root, target_is_directory=True)
        alias_manifest=alias/'publisher_assets.csv'
        alias_output=root/'alias_results.json'
        verify(alias,alias_manifest,alias_output)
        assert json.loads(alias_output.read_text())['results'][0]['status']=='CHECKSUM_MATCH_ASSET_ONLY'
        outside=Path(d).parent/'__outside_sentinel_nonexistent__'
        (root/'escape').symlink_to(outside)
        escaped=root/'escaped_assets.csv'
        escaped.write_text('asset_name,local_relative_path,checksum_algorithm,publisher_checksum,publisher_record_url\n'
            +f'escape,escape/file,sha256,{h},https://zenodo.org/records/example\n')
        try:verify(root,escaped,root/'escape_results.json')
        except ValueError as exc:
            assert 'outside' in str(exc).lower()
        else:raise AssertionError('Symlink escape must be blocked')
        assert not (root/'escape_results.json').exists()
    print('PUBLISHER ASSET VERIFIER SELF-TEST: PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--publisher-asset-manifest',type=Path)
    p.add_argument('--output',type=Path,default=Path('reports/audits/publisher_asset_checks_phase5_v1.json'))
    p.add_argument('--self-test-only',action='store_true');a=p.parse_args()
    if a.self_test_only:selftest()
    else:
        if a.publisher_asset_manifest is None:p.error('--publisher-asset-manifest is required')
        root=a.root.resolve();manifest=a.publisher_asset_manifest if a.publisher_asset_manifest.is_absolute() else root/a.publisher_asset_manifest
        output=a.output if a.output.is_absolute() else root/a.output
        verify(root,manifest,output)
