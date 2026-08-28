# The files behind the numbers

Every figure quoted in our note of 28 August comes from one of these.

| file | what it is |
|---|---|
| `A0.csv` `A1.csv` `A2.csv` `A3.csv` | the four ablation arms, one row per epoch: `epoch,beta,lr,train_loss,val_loss,val_gpr,sec,peak_gib`. 43 epochs each, same seed and sample order, one GPU per arm |
| `anchors_4mm_n175.csv` | your 4 mm checkpoint re-scored on your 175 test volumes under both anchors, refined and baseline: `ref_max_*`, `ref_p99.9_*`, `base_max_*`, `base_p99.9_*` for the 2%/2mm and 1%/1mm criteria. No retraining: the same predictions, scored twice |
| `anchors_2mm_n89.csv` | the same comparison at 2 mm on 89 volumes, plus a least-squares oracle column - the best any per-volume rescaling could reach |
| `probe_2mm_one_epoch.csv` | one full epoch at 2 mm, batch 2, on 787 training and 77 validation volumes: this is where the 3,401 s and the 53.81 GiB peak come from |
| `rotations_tta_4mm_n50.csv` | the rotation experiment on 50 volumes at 4 mm: `rot0..rot3` are the four 90 degree orientations scored separately, `tta_*` is gamma computed once on the average of the four re-aligned predictions |

Two cautions when reading them.

**Compare `val_gpr`, not `val_loss`.** `CombinedWMSEGammaLoss` normalises its WMSE term by a
running average of itself, so that term sits near 1 for any model. At epoch 4 of this run,
A2 and A3 have lower validation loss than A0 and A1 while scoring four times worse on GPR.

**`peak_gib` is the maximum since the process started**, so if a run were resumed across
jobs that column would describe the job rather than the whole training. These four arms ran
in a single job, so here it is the whole training.
