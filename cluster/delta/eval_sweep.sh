#!/usr/bin/env bash
# Run a list of fixed-subset evals with at most MAX_INFLIGHT (2) jobs queued/running
# (Delta's per-user QOS limit), reporting one line per finished eval.
# Spec file: one eval per line -> run_name|on_or_off|checkpoint_dir_or_empty
# Usage: cluster/delta/eval_sweep.sh <specs> <episodes.json>
set -o pipefail
SPECS="$1"; EPISODES="$2"; MAX_INFLIGHT="${MAX_INFLIGHT:-2}"
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PY=/work/nvme/bgon/brabiei/longnav_runtime/envs/longnav_vlm/bin/python
RUNS=/work/nvme/bgon/brabiei/longnav_runtime/runs; LOGS=/work/nvme/bgon/brabiei/longnav_runtime/logs
agg() { $PY - "$1" <<'PYEOF'
import json, glob, sys, numpy as np
run=sys.argv[1]; rows=[json.loads(l) for f in glob.glob(f"/work/nvme/bgon/brabiei/longnav_runtime/runs/{run}/rollout/results_*") for l in open(f) if l.strip()]
if not rows: print(f"EVAL {run}: no result rows"); sys.exit()
s=np.array([r['success'] for r in rows]); spl=np.array([r['spl'] for r in rows]); n=np.array([r['n_steps'] for r in rows])
d=[r.get('sup/mean_hamlet_mem_ratio') for r in rows if r.get('sup/mean_hamlet_mem_ratio') is not None]
lap=np.array([r.get('last_action_prob', np.nan) for r in rows])
print(f"EVAL {run}: n={len(rows)} success={s.mean():.3f} spl={spl.mean():.3f} steps={n.mean():.0f} timeouts={(n>=350).mean():.2f} stop_conf={np.nanmean(lap):.2f}" + (f" mem_ratio={np.mean(d):.4f}" if d else ""))
PYEOF
}
declare -A job_of done_of spec_h spec_c; order=()
while IFS='|' read -r name hamlet ckpt; do [[ -z "$name" || "$name" == \#* ]] && continue; order+=("$name"); spec_h[$name]="$hamlet"; spec_c[$name]="$ckpt"; done < "$SPECS"
next=0
while true; do
  inflight=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^hamlet_eval$" || true)
  while (( next < ${#order[@]} && inflight < MAX_INFLIGHT )); do
    name="${order[$next]}"; exp="RUN_NAME=${name},EVAL_HAMLET=${spec_h[$name]},EVAL_EPISODES=${EPISODES}"
    [[ -n "${spec_c[$name]}" ]] && exp="${exp},EVAL_CHECKPOINT=${spec_c[$name]}"
    if jid=$(sbatch --parsable --export=ALL,$exp cluster/delta/eval_ckpt.sbatch 2>&1); then job_of[$name]=$jid; echo "submitted $name as job $jid"; next=$((next+1)); inflight=$((inflight+1)); else echo "submit failed for $name: $jid"; break; fi
  done
  q=$(squeue -u "$USER" -h -o "%i" 2>/dev/null || true)
  for name in "${!job_of[@]}"; do
    jid=${job_of[$name]}
    if [[ -z "${done_of[$name]:-}" ]] && ! grep -qx "$jid" <<<"$q"; then
      done_of[$name]=1; log="${LOGS}/hamlet_eval_${jid}.out"
      err=$(grep -cE "Traceback|Error executing|ERROR:" "$log" 2>/dev/null || true)
      echo "finished $name (job $jid, errors=$err)"; (( err > 0 )) && grep -E "Error|ERROR|Traceback" "$log" | grep -v "MSE Error" | tail -2 | cut -c1-200
      agg "$name"
    fi
  done
  (( next >= ${#order[@]} && ${#done_of[@]} == ${#order[@]} )) && { echo "sweep complete"; break; }
  sleep 60
done
