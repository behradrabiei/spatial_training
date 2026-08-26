#!/usr/bin/env bash
# Watch a HAMLET training run: evaluate its checkpoints on the fixed 36-episode
# HM3D-v2 val subset (argmax, 1 GPU each, cluster/delta/eval_ckpt.sbatch) as they
# land, and cancel the training job if the success rate collapses -- the previous
# runs' policies stopped emitting `stop` under argmax between checkpoint 3 and 7
# while their sampled-rollout success stayed high.
#
# Usage (login node, from the repo root):
#   nohup cluster/delta/watch_train.sh <train_job_id> <run_name> \
#       > /work/nvme/bgon/brabiei/longnav_runtime/logs/watch_<run_name>.out 2>&1 &
# Evaluates checkpoints 3, 7, 11, 15, then every 4th one ((N+1) % 16 == 0) and
# checkpoint_final. Results: <run>/eval36.jsonl (one JSON line per eval).
# Knobs: STOP_THRESHOLD (0.10: success at or below -> scancel), WARN_THRESHOLD (0.5),
#        MAX_INFLIGHT (2 eval jobs), POLL_S (120), EVAL_EPISODES (hm3d_v2_val36.json).
set -o pipefail
JOB="$1"; RUN="$2"
[[ -n "$JOB" && -n "$RUN" ]] || { echo "usage: watch_train.sh <train_job_id> <run_name>" >&2; exit 2; }
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PY=/work/nvme/bgon/brabiei/longnav_runtime/envs/longnav_vlm/bin/python
RUNS=/work/nvme/bgon/brabiei/longnav_runtime/runs; LOGS=/work/nvme/bgon/brabiei/longnav_runtime/logs
EPISODES="${EVAL_EPISODES:-$PWD/cluster/delta/hm3d_v2_val36.json}"
STOP_THRESHOLD="${STOP_THRESHOLD:-0.10}"; WARN_THRESHOLD="${WARN_THRESHOLD:-0.5}"
MAX_INFLIGHT="${MAX_INFLIGHT:-2}"; POLL_S="${POLL_S:-120}"
CKPT_DIR="$RUNS/$RUN/checkpoints"; OUT="$RUNS/$RUN/eval36.jsonl"
N_EXPECTED=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$EPISODES")
log() { echo "[watch $(date -u +%H:%M:%S)] $*"; }

should_eval() {  # checkpoint dir name -> 0 if it is on the evaluation schedule
    local name="$1"
    [[ "$name" == "checkpoint_final" ]] && return 0
    [[ "$name" =~ ^checkpoint_([0-9]+)$ ]] || return 1
    local n="${BASH_REMATCH[1]}"
    (( n == 3 || n == 7 || n == 11 || n == 15 || (n + 1) % 16 == 0 ))
}

agg() {  # eval run name, checkpoint name -> one JSON line (also appended to $OUT)
    $PY - "$1" "$2" "$RUNS" "$N_EXPECTED" <<'PYEOF'
import glob, json, sys, numpy as np
run, ckpt, runs, n_expected = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for f in glob.glob(f"{runs}/{run}/rollout/results_*") for l in open(f) if l.strip()]
out = {"ckpt": ckpt, "eval_run": run, "n": len(rows), "n_expected": n_expected}
if rows:
    s = np.array([r["success"] for r in rows]); n = np.array([r["n_steps"] for r in rows])
    lap = np.array([r.get("last_action_prob", np.nan) for r in rows], dtype=float)
    d = [r["sup/mean_hamlet_mem_ratio"] for r in rows if r.get("sup/mean_hamlet_mem_ratio") is not None]
    out.update(success=round(float(s.mean()), 4), spl=round(float(np.mean([r["spl"] for r in rows])), 4),
               steps=round(float(n.mean()), 1), timeouts=round(float((n >= 350).mean()), 3),
               stop_conf=round(float(np.nanmean(lap)), 3) if np.isfinite(lap).any() else None,
               mem_ratio=round(float(np.mean(d)), 5) if d else None)
print(json.dumps(out))
PYEOF
}

declare -A job_of done_of
log "watching job $JOB run $RUN (stop if success <= $STOP_THRESHOLD on $N_EXPECTED episodes)"
while true; do
    now=$(date +%s)
    training_alive=$(squeue -h -j "$JOB" -o "%i" 2>/dev/null | grep -c . || true)
    inflight=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^hamlet_eval$" || true)
    # 1. new complete checkpoints on the schedule (scheduler.pt is written last; wait 60 s after it)
    for d in "$CKPT_DIR"/checkpoint_*; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        [[ -z "${job_of[$name]:-}" ]] || continue
        should_eval "$name" || continue
        [[ -f "$d/scheduler.pt" && -f "$d/adapter_model.safetensors" ]] || continue
        (( now - $(stat -c %Y "$d/scheduler.pt") >= 60 )) || continue
        (( inflight < MAX_INFLIGHT )) || break
        ev="ev36_${RUN}_${name}"
        if jid=$(sbatch --parsable --export=ALL,RUN_NAME="$ev",EVAL_HAMLET=on,EVAL_CHECKPOINT="$d",EVAL_EPISODES="$EPISODES" \
                 cluster/delta/eval_ckpt.sbatch 2>&1); then
            job_of[$name]="$jid"; inflight=$((inflight + 1)); log "submitted eval of $name as job $jid ($ev)"
        else
            log "submit failed for $name (retry next tick): $jid"; break
        fi
    done
    # 2. finished evals
    q=$(squeue -u "$USER" -h -o "%i" 2>/dev/null || true)
    for name in "${!job_of[@]}"; do
        jid="${job_of[$name]}"
        [[ -z "${done_of[$name]:-}" ]] || continue
        grep -qx "$jid" <<<"$q" && continue
        done_of[$name]=1
        line=$(agg "ev36_${RUN}_${name}" "$name"); echo "$line" >> "$OUT"; log "RESULT $line"
        errs=$(grep -cE "Traceback|Error executing|ERROR:" "$LOGS/hamlet_eval_${jid}.out" 2>/dev/null || true)
        (( errs > 0 )) && log "eval job $jid logged $errs error lines: $(grep -E 'Traceback|ERROR' "$LOGS/hamlet_eval_${jid}.out" | tail -1 | cut -c1-160)"
        read -r n succ <<<"$($PY -c "import json,sys; d=json.loads(sys.argv[1]); print(d['n'], d.get('success', 'nan'))" "$line")"
        if [[ "$succ" != "nan" ]] && (( n == N_EXPECTED )); then
            if $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$succ" "$STOP_THRESHOLD"; then
                if (( training_alive > 0 )); then
                    scancel "$JOB" && log "COLLAPSE: $name success=$succ <= $STOP_THRESHOLD -- training job $JOB cancelled"
                else
                    log "COLLAPSE signature on $name (success=$succ) but job $JOB is no longer running"
                fi
            elif $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$succ" "$WARN_THRESHOLD"; then
                log "WARNING: $name success=$succ < $WARN_THRESHOLD (stage-1 adapter: 0.833)"
            fi
        else
            log "eval of $name incomplete ($n/$N_EXPECTED rows); not acting on it"
        fi
    done
    # 3. exit once the training job is gone and nothing is pending
    if (( training_alive == 0 )); then
        pending=0
        for name in "${!job_of[@]}"; do [[ -z "${done_of[$name]:-}" ]] && pending=$((pending + 1)); done
        unsubmitted=0
        for d in "$CKPT_DIR"/checkpoint_*; do
            [[ -d "$d" ]] || continue; name=$(basename "$d")
            [[ -z "${job_of[$name]:-}" ]] && should_eval "$name" && [[ -f "$d/scheduler.pt" ]] && unsubmitted=$((unsubmitted + 1))
        done
        if (( pending == 0 && unsubmitted == 0 )); then
            log "training job $JOB finished and all evals done; exiting"; break
        fi
    fi
    sleep "$POLL_S"
done
