#!/usr/bin/env python
"""What does moving the p99.9 anchor into TRAINING actually change?

The proposal is to anchor both fields at their own 99.9th percentile during
training, and to rebuild the loss tolerance "the same way as in the evaluation
script". This script decides what that does, because it cannot be settled by
reading:

  the training tolerance does NOT come from the data pipeline. It is recomputed
  inside the loss on every call, from the maximum of the target:

      delta        = dose_percent * target.max()      (losses_opt_1mm.py)
      cutoff_value = dose_cutoff  * target.max()

  Rescaling the target therefore rescales the tolerance with it, and the
  RELATIVE tolerance is unchanged. What does not cancel is that prediction and
  target end up divided by two DIFFERENT statistics, so the net effect is a
  rescaling of the prediction alone by b/a, where

      a = max(target)/p99.9(target)      b = max(pred)/p99.9(pred)

The identity is algebraic and holds for any a and b; this script measures it
rather than arguing it, and separately shows what happens to the WMSE term,
whose weight exp(alpha*y_true) does see the scale.

    python ablation/check_training_anchor.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from losses_opt_1mm import GammaIndexLoss, WeightedMSE  # noqa: E402

torch.manual_seed(0)

# synthetic field with a narrow peak, like a Bragg peak
D = torch.zeros(1, 1, 64, 16, 64)
zz = torch.linspace(0, 1, 64).view(1, 1, -1, 1, 1)
D += torch.exp(-((zz - 0.75) ** 2) / 0.004) + 0.15 * torch.rand_like(D)

tgt0 = (D - D.min()) / (D.max() - D.min())          # today's pipeline: min-max
raw_pred = (tgt0 + 0.05 * torch.randn_like(tgt0)).clamp(0, None)


def p999(x):
    return torch.quantile(x.flatten(), 0.999)


g = GammaIndexLoss(dose_percent=1.0, dta_mm=2.0, voxel_size_mm=(4., 4., 4.),
                   dose_cutoff=0.2, beta_init=1.0, max_gamma=10.0)
w = WeightedMSE(alpha=1.0)

# V0: both anchored at their own maximum (min-max pipeline + line 205)
tgt_V0 = tgt0
pred_V0 = raw_pred / raw_pred.amax()

# V2: both anchored at their own 99.9th percentile
tgt_V2 = tgt0 / p999(tgt0)
pred_V2 = raw_pred / p999(raw_pred)

a = float(tgt_V2.amax())
b = float(pred_V2.amax() / pred_V0.amax())
print(f"a = max(tgt)/p999(tgt)  = {a:.4f}")
print(f"b = max(pred)/p999(pred) = {b:.4f}")
print(f"b/a = {b / a:.4f}\n")

pred_equiv = pred_V0 * (b / a)

rows = [("V0  (min-max pipeline, output anchored at max)", pred_V0, tgt_V0),
        ("V2  (p99.9 on both)", pred_V2, tgt_V2),
        ("V0 with the PREDICTION alone scaled by b/a", pred_equiv, tgt_V0)]
print(f"{'':46} {'gamma':>12} {'GPR':>8} {'wmse':>12}")
for name, p, t in rows:
    print(f"{name:46} {float(g(p, t)):12.8f} {g.compute_pass_rate(p, t):8.3f} "
          f"{float(w(p, t)):12.6f}")

print("""
How to read it:
  rows 2 and 3 agree on the gamma column -> in training the p99.9 does NOT move
  the tolerance anchor: its entire effect on the gamma term is a rescaling of
  the prediction alone by b/a.
  the wmse column does change between rows 1 and 2 -> there the effect is real,
  but it is a change to the exp(alpha*y_true) WEIGHTING PROFILE, not the anchor
  that was intended. The benefit of the anchor lives at SCORING time.
""")
