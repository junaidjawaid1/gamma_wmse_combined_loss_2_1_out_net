#!/usr/bin/env python3
"""What happens if the p99.9 anchor is moved into the DATA PIPELINE.

Why this exists. Moving the 99.9th percentile normalisation from evaluation into
`data_pipeline_256_32_256.py` looks like a one-line change. It is not, because of
line 205 of `combined_loss_train_norm.py`:

    output_norm = output / (output.amax(dim=(2,3,4), keepdim=True) + 1e-10)
    loss, loss_dict = criterion(output_norm, target)

The prediction is renormalised to a maximum of exactly 1 before the loss, always,
whatever the pipeline did. This script measures the consequence rather than
arguing it.

Nothing here needs a trained network. The prediction is set equal to the target
- a perfect prediction, the most favourable case there is - and then line 205 is
applied to it. Whatever error survives that is a floor: no amount of training can
go below it.

Usage:
    check_p999_pipeline.py --data /path/to/2mm/dataset --split train --n 20
"""
import argparse
import os

import numpy as np


def minmax(x):
    """What the pipeline does today (lines 47-48)."""
    return (x - x.min()) / (x.max() - x.min())


def p999(x):
    """The anchor requested in the mail."""
    return x / (np.percentile(x, 99.9) + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root containing train/ validation/ test/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    d = os.path.join(a.data, a.split, "dose_1m")
    files = sorted(f for f in os.listdir(d) if f.endswith(".npy"))[: a.n]
    print(f"{len(files)} volumes from {d}\n")

    rows = []
    for f in files:
        y = np.load(os.path.join(d, f)).astype(np.float64)
        t_mm, t_99 = minmax(y), p999(y)
        # perfect prediction, then line 205
        p_mm = t_mm / (t_mm.max() + 1e-10)
        p_99 = t_99 / (t_99.max() + 1e-10)
        m = t_99 > 1e-6
        rows.append(dict(
            tmax_mm=t_mm.max(), tmax_99=t_99.max(),
            res_mm=float(np.abs(p_mm - t_mm).max()),
            rel=float(np.median(np.abs(p_99 - t_99)[m] / t_99[m])) * 100,
            res_abs=float(np.abs(p_99 - t_99).max()),
            above1=float((t_99 > 1.0).mean()) * 100,
        ))

    def col(k):
        return [r[k] for r in rows]

    print("PIPELINE AS IT IS (min-max) - perfect prediction + line 205")
    print(f"  target maximum ................ {np.mean(col('tmax_mm')):.4f}  (expected 1.0000)")
    print(f"  largest error ................. {np.max(col('res_mm')):.2e}   -> line 205 is harmless\n")

    print("PIPELINE AT p99.9 - same perfect prediction, same line 205")
    print(f"  target maximum ................ {np.mean(col('tmax_99')):.4f}"
          f"  (min {np.min(col('tmax_99')):.4f}, max {np.max(col('tmax_99')):.4f})")
    print(f"  voxels above 1 ................ {np.mean(col('above1')):.3f}%  (~0.1% by construction)")
    print(f"  scale factor imposed .......... {np.mean(col('tmax_99')):.2f}x  (this is max/p99.9)")
    print(f"  median relative error ......... {np.mean(col('rel')):.1f}% ON EVERY VOXEL")
    print(f"  largest absolute error ........ {np.mean(col('res_abs')):.3f} (at the peak)")
    print(f"  volumes with error > 2% ....... {sum(1 for x in col('rel') if x > 2)}/{len(rows)}\n")

    print("Reading. It is not that the residual concentrates on the peak. Line 205 divides")
    print("the prediction by its own maximum, so the prediction is anchored at the maximum")
    print("whatever the pipeline does. If the target is anchored at p99.9 its maximum is")
    print("max/p99.9, and the two fields differ by that factor on EVERY voxel: a uniform")
    print("relative error, not a localised defect. The peak is only where it bites hardest")
    print("in absolute dose. The gamma criterion works at 1-2% of the reference dose.")


if __name__ == "__main__":
    main()
