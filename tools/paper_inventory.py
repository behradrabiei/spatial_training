"""Freeze, back up, and verify the scientific inputs used by the retained paper studies.

Run `python tools/paper_inventory.py freeze` once before cleanup, then `verify`.
The backup preserves repository-relative paths; restore a missing file from backup/<path>.
No original input is ever modified by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.cleanup_archive/before-codebase-cleanup'
MANIFEST = ROOT / 'paper_results/protected_inputs.json'
EXTRA = ('src/longnav/conf/episode_jsons', 'dump/hm3d_v2_100_labels.json',
         'dump/longnav_eval/keep_one_influence_w32_b10000000')
STUDIES = {
    'ctx_ablation': [('ctx_ablation/aggregate.py', '--strict'), ('ctx_ablation/plot.py',)],
    'frame_influence': [('frame_influence/aggregate.py',), ('frame_influence/plot.py',)],
    'frame_influence_keepone': [('frame_influence/aggregate.py', '--measure', 'keep_one'), ('frame_influence/plot.py', '--measure', 'keep_one')],
    'token_prune': [('token_prune/aggregate.py', '--strict'), ('token_prune/plot.py',)],
    'token_prune_flat': [('token_prune/aggregate.py', '--variant', 'flat', '--strict'), ('token_prune/plot.py', '--variant', 'flat')],
    'layer_token_info': [('layer_token_info/plot.py',), ('layer_token_info/plot.py', '--study', 'nohist', '--layers', '1', '12', '18'), ('layer_token_info/horizon.py',)],
    'info_horizon': [('info_horizon/aggregate.py',), ('info_horizon/plot.py',)],
    'info_horizon_mp3d': [('info_horizon/aggregate.py', '--dataset', 'mp3d'), ('info_horizon/plot.py', '--dataset', 'mp3d')],
    'hybrid_prune': [('hybrid_prune/aggregate.py', '--strict'), ('hybrid_prune/plot.py',)],
    'hybrid_prune_mp3d': [('hybrid_prune/aggregate.py', '--dataset', 'mp3d', '--strict'), ('hybrid_prune/plot.py', '--dataset', 'mp3d')],
    'selector_kl': [('selector_kl/aggregate.py',), ('selector_kl/plot.py',)],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def obsolete(path):
    parts = path.relative_to(ROOT).parts
    if len(parts) < 3 or parts[:2] != ('paper_results', 'hybrid_prune'):
        return False
    return parts[2] in ('runs_aborted', 'results_last_stage', 'figures_last_stage') or (
        len(parts) > 3 and parts[2] == 'runs' and
        any(parts[3].endswith('_b' + str(k)) for k in (1, 4, 8, 16)))


def scientific(path):
    return (path.is_file() and not obsolete(path) and
            not any(part in ('wandb', '.locks', '__pycache__') for part in path.parts) and
            path.suffix not in ('.log', '.lock', '.mp4', '.pyc', '.partial') and
            (path.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.npz', '.npy', '.parquet',
                             '.csv', '.gz', '.tex', '.png', '.jpg', '.pdf') or
             path.name.startswith('results_')))


def episode_labels(run):
    labels = set()
    for p in sorted((run / 'rollout').glob('results_*')):
        for line in p.read_text().splitlines():
            if line.strip():
                labels.add(json.loads(line)['episode_label'])
    for p in sorted((run / 'episodes').glob('*.json')):
        labels.add(json.loads(p.read_text())['episode_label'])
    return sorted(labels)


def freeze():
    if MANIFEST.exists():
        raise SystemExit('Manifest already exists; verify it instead of overwriting the baseline.')
    files = set()
    studies = {}
    for name, commands in STUDIES.items():
        root = ROOT / 'paper_results' / name
        files.update(p for p in root.rglob('*') if scientific(p))
        runs = {}
        for p in sorted(root.glob('**/runs/*')):
            if p.is_dir() and not obsolete(p) and p.name != '.locks':
                runs[str(p.relative_to(ROOT))] = {'episodes': episode_labels(p)}
        # Nested cost-benchmark arms have their own episode records.
        for p in sorted(root.glob('runs/cost_bench*/*')):
            if p.is_dir():
                runs[str(p.relative_to(ROOT))] = {'episodes': episode_labels(p)}
        studies[name] = {'commands': [['python', 'paper/' + c[0], *c[1:]] for c in commands],
                         'runs': runs, 'outputs': str(root.relative_to(ROOT)),
                         'shared_references': ['paper_results/ctx_ablation/runs/r1_full']}
    for name in EXTRA:
        p = ROOT / name
        if not p.exists():
            raise FileNotFoundError(p)
        files.update(q for q in ([p] if p.is_file() else p.rglob('*')) if scientific(q))
    records = []
    for p in sorted(files):
        relative = str(p.relative_to(ROOT))
        target = ARCHIVE / 'protected-inputs' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(p)
        if not target.exists():
            shutil.copy2(p, target)
        if digest(target) != sha:
            raise RuntimeError(f'Backup checksum mismatch: {relative}')
        records.append({'path': relative, 'bytes': p.stat().st_size, 'sha256': sha})
    manifest = {'baseline_commit': 'bf4e65ff', 'backup': str(ARCHIVE.relative_to(ROOT) / 'protected-inputs'),
                'studies': studies, 'files': records}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Protected and backed up {len(records)} files ({sum(r["bytes"] for r in records) / 2**30:.2f} GiB).')


def verify(backup=False):
    manifest = json.loads(MANIFEST.read_text())
    root = ROOT / manifest['backup'] if backup else ROOT
    failures = [r['path'] for r in manifest['files']
                if not (root / r['path']).is_file() or digest(root / r['path']) != r['sha256']]
    if failures:
        raise SystemExit('Changed or missing protected files:\n' + '\n'.join(failures))
    print(f'All {len(manifest["files"])} protected files verified ({root}).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze', 'verify'))
    ap.add_argument('--backup', action='store_true')
    args = ap.parse_args()
    freeze() if args.command == 'freeze' else verify(args.backup)
