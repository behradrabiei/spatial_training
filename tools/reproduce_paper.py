"""Rebuild retained paper outputs in an isolated mirror; never overwrite paper_results.

Example: python tools/reproduce_paper.py --name baseline --source .cleanup_archive/before-codebase-cleanup/baseline-code
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--source', type=Path, default=ROOT)
    envs = Path.home() / 'miniconda3/envs'
    ap.add_argument('--inference-python', type=Path, default=envs / 'longnav_vlm/bin/python')
    ap.add_argument('--plot-python', type=Path, default=envs / 'vln/bin/python')
    args = ap.parse_args()
    for python in (args.inference_python, args.plot_python):
        if not python.is_file():
            ap.error(f'Missing runtime {python}; supply --inference-python / --plot-python')
    manifest = json.loads((ROOT / 'paper_results/protected_inputs.json').read_text())
    mirror = ROOT / '.cleanup_archive/before-codebase-cleanup/validation' / args.name
    mirror.mkdir(parents=True, exist_ok=False)
    for name in ('paper', 'src'):
        shutil.copytree(args.source / name, mirror / name, ignore=shutil.ignore_patterns('__pycache__'))
    for entry in manifest['files']:
        rel = Path(entry['path'])
        if rel.parts[0] == 'src':
            continue
        dst = mirror / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Writers may regenerate these outputs. Everything else is a read-only input.
        if 'results' in rel.parts or 'figures' in rel.parts:
            shutil.copy2(ROOT / rel, dst)
        else:
            dst.symlink_to(ROOT / rel)
    commands = [c for study in manifest['studies'].values() for c in study['commands']]
    # Keep dependencies first: other studies read context-ablation results.
    commands.sort(key=lambda c: (0 if 'ctx_ablation' in c[1] else 1))
    commands += manifest['studies']['hybrid_prune'].get('cost_commands', [])
    commands += manifest.get('additional_commands', [])
    env = dict(os.environ, PYTHONPATH=str(mirror / 'src'), PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    report = []
    for i, cmd in enumerate(commands):
        plot = 'plot' in Path(cmd[1]).stem or Path(cmd[1]).name == 'horizon.py'
        python = args.plot_python if plot else args.inference_python
        log = mirror / f'command-{i:02d}.log'
        with log.open('w') as stream:
            proc = subprocess.run([str(python), *cmd[1:]], cwd=mirror, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report.append({'command': cmd, 'exit_code': proc.returncode, 'log': str(log.relative_to(ROOT))})
        (mirror / 'report.json').write_text(json.dumps(report, indent=2))
        print(f'{i + 1}/{len(commands)} exit={proc.returncode}: {" ".join(cmd[1:])}', flush=True)
    print('Report:', mirror / 'report.json')
    if any(r['exit_code'] for r in report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
