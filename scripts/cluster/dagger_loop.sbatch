#!/bin/bash
# Full DAgger loop on PickXtimes:
#   iter 0 = start with BC policy
#   for i in 1..N_ITERS:
#     collect 20 episodes via dagger_iter (policy in env, expert query every K)
#     append to aggregated dataset (under runs/dagger/iter{i}/episodes/)
#     BC retrain on aggregated dataset → produce dagger_policy_iter{i}.pt
#
# After all iters, eval the final policy on test split with subgoal diagnostic.
#
# Submit:  sbatch scripts/cluster/dagger_loop.sbatch
#SBATCH --account=vision --partition=svl --qos=normal
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=rme-dagger
#SBATCH --output=/vision/u/jevon/robomme_benchmark/logs/slurm/%x-%j.out

set -euo pipefail

source /vision/u/jevon/miniconda3/etc/profile.d/conda.sh
conda activate robomme
export PYTHONNOUSERSITE=1
export ROBOMME_OBS_MODE=rgb
export ROBOMME_SIM_BACKEND=physx_cpu
export WANDB_MODE=disabled

cd /vision/u/jevon/robomme_benchmark
PY=/vision/u/jevon/miniconda3/envs/robomme/bin/python

N_ITERS=${N_ITERS:-4}
N_EPISODES=${N_EPISODES:-20}
K_QUERY=${K_QUERY:-15}
BC_EPOCHS=${BC_EPOCHS:-15}
BC_BATCH=${BC_BATCH:-256}
OUT=${OUT:-runs/dagger}

mkdir -p "$OUT" logs/slurm

# Aggregated dataset uses the same episode-folder layout as BC pipeline:
# runs/dagger/all_episodes/ep_XXXX.npz. We seed it with the original BC
# episodes, then each DAgger iter appends more.
AGG_DIR="$OUT/all_episodes"
mkdir -p "$AGG_DIR"
if [[ -z "$(ls -A "$AGG_DIR" 2>/dev/null)" ]]; then
    echo "[seed] copying original 80 BC episodes into $AGG_DIR"
    cp runs/bc/pickxtimes/episodes/ep_*.npz "$AGG_DIR/"
fi
START_IDX=$(ls "$AGG_DIR"/ep_*.npz 2>/dev/null | wc -l)
echo "[seed] starting episode index: $START_IDX"

CUR_POLICY="runs/bc/pickxtimes/bc_policy_recurrent.pt"

for ITER in $(seq 1 $N_ITERS); do
    echo ""
    echo "================================================================"
    echo "=== DAgger iter $ITER / $N_ITERS ==="
    echo "================================================================"
    ITER_DIR="$OUT/iter${ITER}"
    mkdir -p "$ITER_DIR"

    # --- Collect with current policy ---
    echo "[iter $ITER] collecting $N_EPISODES episodes with k_query=$K_QUERY"
    $PY -u -m scripts.dagger_iter \
        --policy "$CUR_POLICY" --recurrent \
        --n_episodes $N_EPISODES --k_query $K_QUERY \
        --out "$ITER_DIR/episodes" --start_idx $START_IDX --seed $ITER

    # Append new episodes to aggregated dir.
    N_NEW=$(ls "$ITER_DIR/episodes"/ep_*.npz 2>/dev/null | wc -l)
    echo "[iter $ITER] collected $N_NEW new episode-files"
    if [[ "$N_NEW" -gt 0 ]]; then
        cp "$ITER_DIR/episodes"/ep_*.npz "$AGG_DIR/"
        START_IDX=$(( START_IDX + N_NEW ))
    fi

    # --- Build a fresh .pt dataset from aggregated episodes ---
    # bc_pretrain expects a .pt; we generate one by concatenating npz files.
    AGG_PT="$OUT/iter${ITER}/dataset.pt"
    $PY -u -c "
import glob, numpy as np, torch
files = sorted(glob.glob('$AGG_DIR/ep_*.npz'))
acts = []; obs = {}
for f in files:
    d = np.load(f)
    acts.append(d['actions'])
    for k in d.files:
        if k == 'actions': continue
        obs.setdefault(k, []).append(d[k])
    d.close()
N = sum(a.shape[0] for a in acts)
obs_concat = {k: np.concatenate(v, axis=0) for k, v in obs.items()}
act_concat = np.concatenate(acts, axis=0).astype(np.float32)
print(f'aggregated {len(files)} episodes, {N} transitions')
torch.save({'obs': obs_concat, 'actions': act_concat,
            'ep_lengths': np.asarray([a.shape[0] for a in acts], dtype=np.int32),
            'success_mask': np.ones(len(acts), dtype=bool),
            'task': 'PickXtimes'}, '$AGG_PT')
print(f'saved → $AGG_PT  ({sum(v.nbytes for v in obs_concat.values())/1e6:.0f} MB)')
"

    # --- Retrain BC on aggregated dataset ---
    NEW_POLICY="$OUT/iter${ITER}/dagger_policy_recurrent.pt"
    echo "[iter $ITER] BC retraining on aggregated dataset → $NEW_POLICY"
    $PY -u -m scripts.bc_pretrain_pickxtimes \
        --dataset "$AGG_PT" \
        --out "$NEW_POLICY" \
        --epochs $BC_EPOCHS --batch_size $BC_BATCH --recurrent

    CUR_POLICY="$NEW_POLICY"

    # --- Quick eval to track progress ---
    echo "[iter $ITER] evaluating policy on 10 test episodes (subgoal-progress diagnostic)"
    $PY -u -m scripts.eval_subgoal "$NEW_POLICY" recurrent 10 || echo "  eval failed; continuing"
done

echo ""
echo "================================================================"
echo "=== Final DAgger eval (20 test episodes) ==="
echo "================================================================"
$PY -u -m scripts.eval_subgoal "$CUR_POLICY" recurrent 20

echo ""
echo "[dagger] complete; final policy at $CUR_POLICY"
