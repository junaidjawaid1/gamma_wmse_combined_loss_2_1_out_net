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
| `check_p999_pipeline.py` | what moving the p99.9 anchor into the data pipeline does to the scales |
| `check_gamma_scale_invariance.py` | shows the training gamma is invariant to a common rescaling |
| `peak_to_percentile_ratio.py` | why the anchor changes anything: the two fields have different peak-to-percentile ratios |
| `train_150.sbatch`, `submit_chain.sh` | one link of a chained full-length run, and the chain submitter |
| `eval_anchors.py` | Table 1 under both anchors, refined and baseline rows, paired per volume |
| `baseline_anchors.py` | the baseline row alone under both anchors - no network, no checkpoint |
| `check_training_anchor.py` | what moving the p99.9 anchor into **training** actually changes |
| `check_head_scale.py` | whether the head after `ac70107` can still see the scale of the dose |
| `beam_geometry.py` | beam angle and field depth of every volume, recovered from the dose field |
| `gpr_by_angle.py` | joins those angles with the measured pass rates and splits grazing from non-grazing |

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

## Cost, measured on one A100 64 GB

`train_arm.py` takes `--res 4mm` or `--res 2mm`, which sets shape, voxel size, gamma
criterion, example counts and default batch to match the corresponding training script.
One full epoch, timed on both:

| | seconds/epoch | peak memory | 150 epochs, one GPU | 24-hour jobs to chain |
|---|---:|---:|---:|---:|
| 4 mm, batch 4 | 1,900 | 49.7 GiB | 79 hours | 4 |
| 2 mm, batch 2 | 3,401 | 53.8 GiB | 142 hours | 6 |

So the 2 mm costs 1.79 times the 4 mm per epoch, and it fits on a single card: scaling the
measured memory down by batch size puts it near 27 GiB at batch 1.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional at 2 mm.** Without it
the same job dies out of memory with 62.35 GiB in use of which **12.10 GiB are reserved but
unallocated** - fragmentation, not demand. With it, the run completes at 53.81 GiB.

## The full-length run, and an independent reproduction

`A0` - the paper's own configuration - was retrained from scratch for the full 150
epochs on a different machine and a different toolchain, then scored on the same 175
test volumes as the released checkpoint.

| 4 mm, n = 175 | released checkpoint | retrained A0, 150 epochs |
|---|---:|---:|
| refined 2%/2mm | 99.39 +- 1.99 | 99.33 +- 2.18 |
| refined 1%/1mm | 96.96 +- 6.05 | 96.88 +- 5.87 |
| refined 2%/2mm, p99.9 anchor | 99.59 +- 0.49 | 99.63 +- 0.50 |
| refined 1%/1mm, p99.9 anchor | 98.34 +- 1.48 | 98.50 +- 1.48 |

Paired per volume, the retrained model is 0.06-0.07 pp from the released one on the
published criteria and 0.16 pp above it under the robust anchor. **The method
reproduces**, which is a stronger statement than re-scoring released weights.

The evaluation carries its own control: the released checkpoint was re-scored on this
machine and compared with an earlier scoring of the same checkpoint elsewhere. The
**baseline** columns, which never pass through the network, came out identical to
0.0000; the **refined** columns differ by at most 0.2721 pp, which is forward-pass
non-determinism across hardware. A deviation everywhere, or zero everywhere, would each
have meant something was wrong.

One observation about selection rather than training. The delivered `best.pth` comes
from **epoch 75**, where beta was 3.38, and its margin over the next best point is
0.012 pp against an epoch-to-epoch scatter of about 0.2. The beta ramp to 5.0 therefore
never reaches the model that is used, and picking by the running maximum on a flat curve
is close to picking at random - two identical runs would deliver checkpoints from
different epochs. A moving average over k epochs would be steadier.

Results: `results/run150_4mm_A0.csv`, `results/anchors_4mm_UAQ_ckpt.csv`,
`results/anchors_4mm_A0_150ep.csv`.

## The anchor belongs to the scoring, not to the pipeline

`eval_anchors.py` scores the same predictions twice, changing only the scalar each field
is divided by. At 4 mm the refined row moves from 99.39 +- 1.99 to 99.59 +- 0.49 at
2%/2mm and from 96.96 +- 6.05 to 98.34 +- 1.48 at 1%/1mm. What changes is the tail, not
the mean: volumes below 90% go from 16 to none and the worst volume from 52.40 to 90.19.
(An
earlier scoring of the same checkpoint on different hardware put that worst volume at
52.23; the 0.17 pp difference is the same forward-pass non-determinism quantified just
above, and is a fair illustration of how much of it there is.)
The same anchor lifts the **baseline** far more - +8.19 pp at 2%/2mm - which is why it
has to be applied to both rows, never to the refined row alone.

`baseline_anchors.py` isolates that effect with no network in the loop, and also reports
the peak-to-p99.9 ratio of each field, which is the mechanism: the maximum of a 5k field
is a noise spike exceeding the reference peak by 1.14x to 6.12x.

**Moving the same anchor into the data pipeline is a different change, and it does not do
what it looks like.** `check_training_anchor.py` measures it: the training tolerance is
recomputed inside the loss from `target.max()` on every call, so rescaling the target in
the pipeline rescales the tolerance with it and the relative tolerance is unchanged. The
entire residual effect on the gamma term is a rescaling of the prediction alone, by the
ratio of the two peak-to-percentile ratios; what genuinely changes is the WMSE weighting
profile, `exp(alpha*y_true)`, which is a different experiment. The script prints three
rows and two of them agree to six significant figures.

Related, and easy to miss: `check_head_scale.py` shows that the `InstanceNorm3d` added by
`ac70107` makes the head blind to absolute scale - multiplying the head's input by five
moves the output maximum by a factor 1.00009, against 1.44 without it. Under line 205 this
is invisible, because the output is renormalised to its own maximum anyway. It matters the
moment the network is asked to emit absolute dose.

## Beam angle, recovered from the data

The `.npy` files carry no beam metadata, so `beam_geometry.py` recovers the angle from the
dose field itself by dose-weighted PCA in the X-Z plane, and `gpr_by_angle.py` joins it with
the measured pass rates.

At **4 mm** the test set contains no grazing beams: 0 of 175 volumes below 20 degrees, the
minimum being 31.0, while training and validation both reach 0. At **2 mm** it does - 33 of
89 - and there the grazing volumes score **above** the rest, by +0.43 pp at 2%/2mm and
+1.53 pp at 1%/1mm; the three worst volumes sit at 98, 75 and 39 degrees. So the angles
missing from the 4 mm test set are the easier ones, and the 4 mm figures are if anything
slightly conservative.

Results: `results/beam_geometry_4mm_test.csv`, `results/beam_geometry_2mm_test.csv`.

**One thing we could not recover from the data: the beam energy of the 4 mm test set.**
`TOPAS_test.py` samples a fixed 150 MeV and `TOPAS_sim_test_2.py` samples uniform(90, 145);
we tried three observables - field extent, geometric range from the patient surface, and
water-equivalent range integrated from the CT - and each was rejected by the same control:
applied to the validation split, which is generated at a fixed energy, all three give the
same dispersion as the training split, so none of them sees the energy. The field-extent
and geometric-range measures are dominated by anatomy, and in a thoracic CT the lung reads
as air, which truncates any surface-based range. Which of the two scripts produced that
split is the one question we would still ask.

## What is running right now

Three jobs are in flight as this is written; their results will be added to `results/`
when they land. The scripts are all in this directory, so the runs can be inspected or
repeated before the numbers exist.

**1. WMSE against a plain MSE, at equal training budget.** This is the comparison
Section 3.2 asserts but never ran under matched conditions. The summary files in the
tree point the other way - on the network of [6], MSE 95.33 against WMSE 90.09 at
2%/2mm - but those are runs with different training budgets, so nothing follows from
them either way. Same network (A0), same seed, same sample order, same 150 epochs; only
`--wmse-alpha` differs, and `alpha = 0` makes every weight `exp(0) = 1`, i.e. exactly a
plain MSE, without touching a line of the loss. The WMSE arm did not have to be paid
for: it is the full-length A0 run already reported above.

    ROOT=$ROOT DATA=$DATA RES=4mm ARM=A0 N=8 ALPHA=0.0 \
      OUT=$ROOT/run150_4mm_A0_mse bash ablation/submit_chain.sh

What will be concluded, and only this: which loss reaches the higher validation GPR **at
the same epoch**, judged on the regime (mean of the last 20 epochs, not the single
maximum - the maximum of a noisy curve is a maximum of noise). If the difference sits
inside the epoch-to-epoch scatter of one arm, the conclusion is "indistinguishable", not
"equal" and not "the WMSE does nothing". One seed per arm, so the defensible statement is
about this run. Nothing from it belongs in Table 1: the validation GPR is computed inside
the loop, at the training criterion, on the validation split.

**2. The 2 mm A0 run to 150 epochs**, which will give the 2 mm row regenerated from an
independent training, as was done at 4 mm.

**3. The 2 mm baseline under both anchors** (`baseline_anchors.py`, no network involved).
This is the measurement that decides how much of the 12.54% baseline in Table 2 is noise
and how much is normalisation. Until it finishes, the honest statement is that the
mechanism is measured - the 5k field's maximum overshoots the reference peak by 1.14x to
6.12x - and its size at 2 mm is not.

## Reproducing

    export ROOT=/path/to/workdir DATA=/path/to/4mm/dataset
    mkdir -p $ROOT/code && cp ablation/train_arm.py $ROOT/code/
    # plus network_2_1.py, network_elements.py, losses_opt_1mm.py,
    # data_pipeline_128_16_128.py in the same directory
    sbatch ablation/ablation_4arms.sbatch

    # a full-length chained run; ALPHA=0.0 gives a plain MSE, ALPHA=1.0 the published loss
    ROOT=$ROOT DATA=$DATA RES=4mm ARM=A0 N=8 bash ablation/submit_chain.sh

Environment used, matched to the one that produced the published 4 mm checkpoint:
Python 3.10, torch 2.7.1+cu118, numpy 1.26.4, pymedphys 0.41.0.
