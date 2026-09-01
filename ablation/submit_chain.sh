#!/usr/bin/env bash
# Submits a chain of train_150.sbatch links tied by --dependency=afterany.
#
# afterany and not afterok: a link that stops on its wall-clock ceiling exits 0
# anyway, but if one died on an OOM we still want the next to start and resume
# from the last good checkpoint rather than block the chain.
#
# Example, the controlled WMSE-against-MSE comparison at equal budget:
#   ROOT=... DATA=.../uaq-4mm RES=4mm ARM=A0 N=8 ALPHA=0.0 \
#     OUT=$ROOT/run150_4mm_A0_mse bash ablation/submit_chain.sh
# with ALPHA=1.0 being the published loss and ALPHA=0.0 a plain MSE.
set -euo pipefail
: "${RES:?}" "${ARM:?}" "${ROOT:?}" "${DATA:?}"
N=${N:-8}
EXP="ALL,ROOT=$ROOT,DATA=$DATA,RES=$RES,ARM=$ARM,EPOCHS=${EPOCHS:-150}"
EXP="$EXP,MAXH=${MAXH:-7.5},SEED=${SEED:-20260828},ALPHA=${ALPHA:-1.0}"
EXP="$EXP${OUT:+,OUT=$OUT}${LIMIT:+,LIMIT=$LIMIT}"
prev=""
for i in $(seq 1 "$N"); do
    if [ -z "$prev" ]; then
        id=$(sbatch --parsable --job-name="uaq150-$RES-$ARM" --export="$EXP" \
                    "$ROOT/ablation/train_150.sbatch")
    else
        id=$(sbatch --parsable --job-name="uaq150-$RES-$ARM" \
                    --dependency=afterany:"$prev" --export="$EXP" \
                    "$ROOT/ablation/train_150.sbatch")
    fi
    echo "  link $i/$N -> job $id${prev:+ (after $prev)}"
    prev=$id
done
