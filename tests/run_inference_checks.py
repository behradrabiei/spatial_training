"""Run retained inference tests serially on one GPU, logging each check separately."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ['inference_unit.py', 'kv_prune_selectors.py', 'ctx_frames_smoke.py',
         'ctx_recompute_smoke.py', 'reindex_smoke.py', 'kv_prune_smoke.py',
         'vtp_prune_smoke.py', 'vtp_flat_smoke.py', 'vtp_hybrid_smoke.py',
         'kv_prune_layer_smoke.py', 'attn_evict_smoke.py', 'grad_attn_smoke.py',
         'kl_influence_smoke.py', 'influence_recompute_smoke.py',
         'token_info_smoke.py', 'horizon_probe_smoke.py', 'selector_probe_smoke.py']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tests', nargs='+', default=TESTS)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    report = []
    for name in args.tests:
        with (args.output / (name + '.log')).open('w') as log:
            proc = subprocess.run([sys.executable, str(ROOT / 'tests' / name)],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report.append({'test': name, 'exit_code': proc.returncode})
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        print(f'{name}: {"PASS" if proc.returncode == 0 else "FAIL"}', flush=True)
    if any(r['exit_code'] for r in report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
