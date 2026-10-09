"""Compare one real Uni-NaVid HM3D episode against the archived baseline source."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_root = ROOT / 'baselines'
    model_root = baseline_root / 'Uni-NaVid'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               TOKENIZERS_PARALLELISM='false', HABITAT_SIM_LOG='quiet', MAGNUM_LOG='quiet',
               DISPLAY='', __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    for variable, cache in [('HF_HOME','huggingface'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp')]:
        env[variable] = str(baseline_root / '.cache' / cache)
    for name, source in [('baseline', args.baseline.resolve()), ('cleaned', ROOT)]:
        env['PYTHONPATH'] = str(source / 'baselines') + ':' + str(model_root)
        command = [str(baseline_root / '.envs/uninavid-hm3d/bin/python'), '-m', 'uninavid_hm3d.eval',
                   '--uninavid-root', str(model_root), '--model-path', str(model_root / 'model_zoo/uninavid-7b-full-224-video-fps-1-grid-2'),
                   '--labels', str(ROOT / 'src/longnav/conf/episode_jsons/hm3d_v2_100.json'),
                   '--output', str(output / name), '--limit', '1']
        with (output / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=model_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name, 'episode completed', flush=True)
    label = '4ok3usBNeis_0'
    def episode(name):
        value = json.loads((output / name / 'episodes' / (label+'.json')).read_text())
        return {k:v for k,v in value.items() if k not in ('elapsed_seconds','peak_cuda_bytes','trace','video')}
    def trace(name):
        rows = [json.loads(line) for line in (output / name / 'traces' / (label+'.jsonl')).read_text().splitlines()]
        return [{k:v for k,v in row.items() if not k.endswith(('_seconds','_bytes'))} for row in rows]
    assert episode('baseline') == episode('cleaned'), 'Episode metrics differ'
    assert trace('baseline') == trace('cleaned'), 'Action/decision traces differ'
    (output / 'comparison.json').write_text(json.dumps({'status':'PASS', 'episode':episode('cleaned'),
                                                       'action_trace':'exact match'}, indent=2)+'\n')
    print('UNINAVID PARITY PASS', episode('cleaned'), flush=True)


if __name__ == '__main__':
    main()
