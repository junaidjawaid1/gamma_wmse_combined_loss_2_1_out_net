#!/bin/bash
#SBATCH --job-name=uaq-test-resume
#SBATCH --account=CHANGE_ME        # SLURM account / allocation
#SBATCH --partition=CHANGE_ME      # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log
#
# Parity test for the cross-job resume. Epochs are shortened with --limit-train,
# so this costs minutes rather than hours. One GPU with cpus-per-task=8: asking
# for 32 cores with a single GPU gets the whole node billed.
#
# Three runs on the same data and the same seed:
#   A  6 epochs in one go                     <- reference
#   B  3 epochs + resume up to 6              <- what we want to verify
#   C  same as B, but with the criterion state (wmse_avg) cleared in the
#      checkpoint: this is the naive resume, the one you would write if you
#      forgot the registered buffers. It is there to give the test POWER: if B
#      tracks A but C tracks it too, the test is measuring nothing.
#
# set -e matters: without it a Python traceback leaves the job COMPLETED and the
# test looks like it passed. That happened on the first attempt here.

set -ex
cd "${ROOT:?set ROOT to the working directory}" || exit 1
source venv/bin/activate
T=$ROOT/test_resume
rm -rf "$T"; mkdir -p "$T"
COMMON="--arm A1 --data ${DATA:?set DATA} --limit-train 40 --limit-val 16 --batch 4 --seed 20260828"

python code/train_arm.py $COMMON --out "$T/A" --epochs 6

python code/train_arm.py $COMMON --out "$T/B" --epochs 3
python code/train_arm.py $COMMON --out "$T/B" --epochs 6

python code/train_arm.py $COMMON --out "$T/C" --epochs 3
python - <<'PY'
import os, torch
p = os.path.join(os.environ["ROOT"], "test_resume/C/last.pth")
ck = torch.load(p, map_location="cpu", weights_only=False)
# the naive resume: save the weights and forget the state inside the loss
ck["criterion"] = dict(ck["criterion"])
ck["criterion"]["wmse_avg"] = torch.tensor(0.0)
torch.save(ck, p)
print("C: wmse_avg cleared, as a resume that forgets the buffers would leave it")
PY
python code/train_arm.py $COMMON --out "$T/C" --epochs 6

echo "=== log.csv of the three runs ==="
for r in A B C; do echo "--- $r ---"; cat "$T/$r/log.csv"; done
