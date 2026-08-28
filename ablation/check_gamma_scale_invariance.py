"""Is the training gamma loss invariant to the scale of the target?

If it is, moving the p99.9 anchor into the data pipeline cannot touch the gamma
term, because `compute_gamma` derives delta and the cutoff from `target.max()`
inside itself:

    delta        = self.dose_threshold * target.max()
    cutoff_value = self.dose_cutoff   * target.max()
    dose_diff_sq = ((pred_neighbors - target) / (delta + 1e-10)) ** 2

Run it and read the two columns.
"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from losses_opt_1mm import GammaIndexLoss, WeightedMSE, CombinedWMSEGammaLoss

torch.manual_seed(0)
# synthetic field with a narrow peak, like a Bragg peak
D = torch.zeros(1, 1, 16, 8, 16)
zz = torch.linspace(0, 1, 16).view(1,1,-1,1,1)
D += torch.exp(-((zz-0.75)**2)/0.004) + 0.15*torch.rand_like(D)
tgt = D / D.amax()                       # min-max target, maximum 1
pred = (tgt + 0.02*torch.randn_like(tgt)).clamp(0, None)

g = GammaIndexLoss(dose_percent=1.0, dta_mm=2.0, voxel_size_mm=(4.,4.,4.),
                   dose_cutoff=0.2, beta_init=0.1, max_gamma=10.0)
w = WeightedMSE(alpha=1.0)

print(f"{'k':>6} {'gamma':>12} {'wmse':>14} {'wmse/wmse(k=1)':>16}")
base_w = None
for k in (1.0, 2.0, 3.2, 4.76):
    lg = float(g(pred*k, tgt*k))
    lw = float(w(pred*k, tgt*k))
    if base_w is None: base_w = lw
    print(f"{k:6.2f} {lg:12.8f} {lw:14.4f} {lw/base_w:16.2f}")

print("\nReading:")
print("  the gamma column is identical at every k -> the gamma term does NOT see the")
print("  pipeline anchor: delta = dose_percent * target.max() rescales with the data.")
print("  the wmse column does: its weight is exp(alpha*y_true), and with y_true reaching")
print("  3.2 instead of 1 it becomes far more concentrated on the peak.")
