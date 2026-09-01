#!/bin/bash
#SBATCH --job-name=uaq-150
#SBATCH --account=CHANGE_ME        # SLURM account / allocation
#SBATCH --partition=CHANGE_ME      # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A.log
#
# ONE link of the chain that takes an arm to the paper's 150 epochs. Submit it
# several times with --dependency=afterany: each link resumes from last.pth and
# stops cleanly before its own wall-clock ceiling; surplus links exit in seconds
# with "already complete".
#
# Why 150 and not fewer: the beta ramp is tuned for that length (0.1 up to epoch
# 10, 2.0 at 50, 5.0 at 100) and ReduceLROnPlateau has patience 10. At 43 epochs
# beta is 1.525 out of 5.0 and the absolute values are not comparable with the
# paper's tables.
#
# Why 8 hours and not 24. On a busy queue a 24 h request needs a 24 h hole in the
# backfill and can wait for hours; an 8 h request drops into a gap. Measured on
# one such queue (1081 pending against 1320 running): the same job, same account,
# same priority, waited 2.5 h as a 24 h request and started in under 2 minutes as
# an 8 h one. Resuming from a checkpoint makes the shorter link free - it only
# costs more links.
#
# expandable_segments is needed at 2 mm: without it, OOM with 12.10 GiB of
# fragmentation out of 62.35 GiB in use.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -eux
cd "${ROOT:?}"
source venv/bin/activate

python ablation/train_arm.py \
    --arm "${ARM:?}" --res "${RES:?}" --data "${DATA:?}" \
    --out "${OUT:-$ROOT/run150_${RES}_${ARM}}" \
    --wmse-alpha "${ALPHA:-1.0}" \
    --epochs "${EPOCHS:-150}" --seed "${SEED:-20260828}" --workers 7 \
    --max-hours "${MAXH:-7.5}" ${LIMIT:+--limit-train $LIMIT --limit-val 8}

echo "=== epochs in the CSV ==="
tail -n +2 "${OUT:-$ROOT/run150_${RES}_${ARM}}/log.csv" | wc -l
