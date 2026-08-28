#!/bin/bash
#SBATCH --account=CHANGE_ME        # SLURM account / allocation
#SBATCH --partition=CHANGE_ME      # GPU partition
#SBATCH --job-name=uaq-abl-4arms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Four-arm ablation of the two layers added on 2026-08-25 (InstanceNorm3d and
# LeakyReLU between the 2-channel convolution and the final one), separated
# instead of being evaluated together.
#
#   A0  paper as it is   A1  LeakyReLU only   A2  InstanceNorm only   A3  both
#
# The four arms are independent: no communication, one 64 GB A100 each on the
# same node. Cost = the wall-clock of ONE arm, not four.
#
# Note on billing: `billing` follows the CPUs requested, not the GPUs. Asking for
# one GPU with 32 cores gets you charged for a whole node. For a single-GPU job
# ask for cpus-per-task=8.

set -euo pipefail

DATA=${DATA:?set DATA to the dataset root (containing train/ and validation/)}
ROOT=${ROOT:?set ROOT to the working directory}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-4}
SEED=${SEED:-20260825}
# wall-clock margin: the arms stop cleanly before SLURM kills them
MAXH=${MAXH:-23.0}
PY=$ROOT/venv/bin/python

echo "node $(hostname) | $(date) | data $DATA | $EPOCHS epochs | batch $BATCH | seed $SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# one GPU per arm, all four at once
for i in 0 1 2 3; do
    ARM=A$i
    OUT=$ROOT/$ARM
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=$i \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$ROOT/code/train_arm.py" \
        --arm "$ARM" --data "$DATA" --out "$OUT" \
        --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --workers 7 \
        --max-hours "$MAXH" \
        > "$OUT/stdout.log" 2>&1 &
    echo "  arm $ARM -> GPU $i (pid $!)"
done

# if one arm dies the other three stay valid: wait for everything anyway and
# report which one fell over, instead of failing the whole job
fail=0
for job in $(jobs -p); do
    wait "$job" || fail=$((fail + 1))
done

echo "=== end $(date) - failed arms: $fail ==="
for i in 0 1 2 3; do
    printf "A%d  " "$i"
    tail -1 "$ROOT/A$i/stdout.log" 2>/dev/null || echo "(no log)"
done

# The comparison is only valid at equal epochs: if one arm stopped earlier than
# the others, this line says at which epoch ALL of them must be read.
echo "--- epochs completed per arm (compare at the minimum) ---"
for i in 0 1 2 3; do
    printf "A%d  %s\n" "$i" "$(tail -n +2 "$ROOT/A$i/log.csv" 2>/dev/null | wc -l)"
done
