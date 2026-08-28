# Ablation and long runs on an external cluster

This directory holds the code we (Alessio Langiu) used to run experiments for this
paper while Kalifano's compute nodes were down. It is additive: nothing outside this
directory is touched, and none of the original scripts are modified.

The SLURM headers carry `CHANGE_ME` for account and partition. Fill them in for
whichever machine you run on.

## Contents

| file | what it is |
|---|---|
| `train_arm.py` | one arm of the ablation, faithful to the **4 mm** `combined_loss_train_norm.py` that lives on Kalifano, plus checkpoint/resume across jobs |
| `ablation_4arms.sbatch` | the four arms, one GPU each, on a single node |
| `test_resume.sbatch` | parity test for the resume, with a control arm |

`train_arm.py` imports `network_2_1.py`, `data_pipeline_128_16_128.py` and
`losses_opt_1mm.py`. The first and the third are the ones in this repository; the
data pipeline is the 4 mm variant.

## Why four arms

Commit `ac70107` adds **two** layers between the 2-channel and the 1-channel
convolution: `LeakyReLU(0.3)` and `InstanceNorm3d`. Evaluated together, a change in
the result cannot be attributed to either. The four arms separate them:

    A0  the paper as it is    A1  LeakyReLU only    A2  InstanceNorm only    A3  both

They share the graph, the initial weights, the seed and the sample order; the layers
under test are switched off with `nn.Identity` rather than by writing four networks,
which keeps the consumption of the random generator identical.

## What the ablation found

Four A100s, one arm each, stopped by a wall-clock guard at **43 epochs for all four**,
so the comparison is at the same epoch by construction.

| arm | mean GPR, epochs 30-42 | best |
|---|---:|---:|
| A0 paper | 95.07 | 95.58 |
| A1 LeakyReLU | 94.88 | 95.31 |
| A2 InstanceNorm | 95.23 | 95.52 |
| A3 both | 95.25 | 95.56 |

The spread **between** arms is 0.372 pp; the average scatter **within** one arm from
epoch to epoch is 0.346 pp. They are the same number, so at this budget the arms are
indistinguishable. The InstanceNorm collapse at epoch 0 (A2 3.79%, A3 1.43%, against
A0 85.45%) is transient: the first epoch from which an arm stays above 94% for three
consecutive epochs is 13 for A0/A1 and 16 for A2/A3, i.e. about three epochs of cost.

Two limits, stated with the result: one seed and one run per arm; and 43 epochs of the
paper's 150, with `beta` still at 1.525 on the 0.1 -> 2.0 -> 5.0 ramp. The comparison
between arms is fair because the truncation is identical, but **these absolute numbers
are not comparable with Table 1**.

Also worth knowing when reading any of these logs: **compare GPR, never the loss**.
`CombinedWMSEGammaLoss` normalises the WMSE term by a running average of itself, so
that term sits near 1 for any model. At epoch 4, A2 and A3 had lower validation loss
than A0 and A1 and four times worse GPR.

## The resume, and why it is not just the weights

150 epochs at 4 mm is ~79 hours on one A100 (1,900 s per epoch, measured), against a
24-hour queue limit, so a full-length run has to be chained across jobs. `train_arm.py`
writes `last.pth` every epoch and picks it up on restart.

Saving the weights alone is not enough. The optimiser moments, the ReduceLROnPlateau
counters, the GradScaler scale and — easiest to miss — the `wmse_avg` buffer inside the
loss all have to travel with them; restarting with `wmse_avg` at zero changes the scale
of the loss for tens of steps. `SimpleBetaScheduler` does not need saving, because
`step(ep)` computes beta from the epoch and restores itself.

`test_resume.sbatch` checks this on short epochs, with a **control**: run A is six
epochs uninterrupted, run B is 3+3 with a resume, and run C is the same as B but with
`wmse_avg` cleared in the checkpoint, imitating a naive resume. Result on one A100:

| epoch | A uninterrupted | B resumed | C naive resume |
|---|---:|---:|---:|
| 2 (before the resume) | 70.88 | 70.74 | 70.96 |
| 3 | 74.78 | 74.57 | **78.52** |
| 4 | 77.18 | 77.36 | **83.07** |
| 5 | 79.70 | **79.72** | **85.72** |

B tracks A to within 0.2 pp, the same scatter the three runs show before any resume
happens. C departs by **6.0 pp** by epoch 5 — and note the direction: forgetting one
buffer does not degrade the curve, it *improves* it, which is the dangerous case. The
mechanism is visible in the loss: `wmse_avg` is set to the current WMSE on the first
forward pass when it is zero, so a cleared buffer renormalises the WMSE term to exactly
1.0, whereas a correctly carried average is larger than the current value (the WMSE has
been falling) and leaves that term weighing less. Clearing it silently reweights WMSE
against gamma.

A naive resume would therefore have produced a better-looking number that is an artefact
of the restart. That is why the control arm is in the test.

## Reproducing

    export ROOT=/path/to/workdir DATA=/path/to/4mm/dataset
    mkdir -p $ROOT/code && cp ablation/train_arm.py $ROOT/code/
    # plus network_2_1.py, network_elements.py, losses_opt_1mm.py,
    # data_pipeline_128_16_128.py in the same directory
    sbatch ablation/ablation_4arms.sbatch

Environment used, matched to the one that produced the published 4 mm checkpoint:
Python 3.10, torch 2.7.1+cu118, numpy 1.26.4, pymedphys 0.41.0.
