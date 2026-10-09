"""Move explicitly classified artifacts locally, with checksums and safe restoration.

Archive: python tools/archive_cleanup.py archive --name NAME paths...
Restore: python tools/archive_cleanup.py restore --name NAME
Protected scientific inputs can never be archived. Unclassified paths are not discovered
or moved automatically. Manifests are saved before moving; reruns resume interrupted work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from paper_inventory import ROOT, ARCHIVE, MANIFEST, digest


def archive(name, paths):
    record = ARCHIVE / (name + '-archive.json')
    if record.exists():
        manifest = json.loads(record.read_text())
        if manifest['roots'] != paths:
            raise ValueError('Existing archive name has different roots')
    else:
        protected = {r['path'] for r in json.loads(MANIFEST.read_text())['files']}
        files = []
        for relative in paths:
            path = ROOT / relative
            if not (path.exists() or path.is_symlink()) or '..' in Path(relative).parts or Path(relative).is_absolute():
                raise ValueError(f'Invalid archive root: {relative}')
            for p in ([path] if path.is_file() or path.is_symlink() else sorted(path.rglob('*'))):
                if not p.is_file() and not p.is_symlink():
                    continue
                rel = str(p.relative_to(ROOT))
                if rel in protected:
                    raise ValueError(f'Protected input: {rel}')
                if p.is_symlink():
                    target = str(p.readlink())
                    files.append({'path': rel, 'bytes': len(target.encode()), 'sha256': hashlib.sha256(target.encode()).hexdigest(), 'symlink': target})
                else:
                    files.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': digest(p)})
        if len({f['path'] for f in files}) != len(files):
            raise ValueError('Overlapping roots')
        manifest = {'baseline_commit': 'bf4e65ff', 'roots': paths,
                    'archive': str((ARCHIVE / 'obsolete').relative_to(ROOT)), 'files': files}
        record.write_text(json.dumps(manifest, indent=2) + '\n')
    archive_root = ROOT / manifest['archive']
    for relative in manifest['roots']:
        src, dst = ROOT / relative, archive_root / relative
        if src.exists() or src.is_symlink():
            if dst.exists() or dst.is_symlink():
                raise FileExistsError(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst)
    verify(manifest, archive_root)
    print(f"Archived {len(manifest['files'])} files, {sum(f['bytes'] for f in manifest['files']) / 2**30:.2f} GiB; {record}")


def verify(manifest, root):
    for entry in manifest['files']:
        path = root / entry['path']
        if 'symlink' in entry:
            valid = path.is_symlink() and str(path.readlink()) == entry['symlink']
        else:
            valid = path.is_file() and digest(path) == entry['sha256']
        if not valid:
            raise ValueError(f'Checksum failed: {path}')


def restore(name):
    manifest = json.loads((ARCHIVE / (name + '-archive.json')).read_text())
    source = ROOT / manifest['archive']
    verify(manifest, source)
    for entry in manifest['files']:
        dst = ROOT / entry['path']
        if dst.exists() or dst.is_symlink():
            same = (dst.is_symlink() and str(dst.readlink()) == entry['symlink']) if 'symlink' in entry else (not dst.is_symlink() and dst.is_file() and digest(dst) == entry['sha256'])
            if not same:
                raise FileExistsError(f'Refusing to overwrite changed file: {dst}')
    for entry in manifest['files']:
        dst = ROOT / entry['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and not dst.is_symlink():
            if 'symlink' in entry:
                dst.symlink_to(entry['symlink'])
            else:
                shutil.copy2(source / entry['path'], dst)
    verify(manifest, ROOT)
    print('Restored and verified', len(manifest['files']), 'files; archive kept.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['archive', 'restore'])
    ap.add_argument('--name', required=True)
    ap.add_argument('paths', nargs='*')
    args = ap.parse_args()
    if not args.name.replace('-', '').replace('_', '').isalnum():
        ap.error('name must contain only letters, numbers, hyphens, or underscores')
    archive(args.name, args.paths) if args.command == 'archive' else restore(args.name)
