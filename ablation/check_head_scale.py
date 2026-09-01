#!/usr/bin/env python
"""After commit ac70107, can the head still see the scale of the dose?

network_2_1.py now ends with

    self.conv       = LazyConv3d(out_channels=2, kernel_size=1)
    self.norm       = LazyInstanceNorm3d()      <- added by ac70107
    self.leaky_relu = LeakyReLU(0.3)            <- added by ac70107
    self.final_conv = LazyConv3d(out_channels=1, kernel_size=1)

InstanceNorm3d normalises each channel of each sample to zero mean and unit
variance. If it does, the absolute scale of what arrives from `conv` is erased,
and `final_conv` (1x1 plus bias) can only produce a scale fixed by its weights,
not one that depends on the input volume.

Under line 205 of combined_loss_train_norm.py this is INVISIBLE, because the
output is renormalised to its own maximum before the loss anyway. It becomes
decisive as soon as the network is asked to produce an absolute scale - which is
what a clinical use of the output would need.

Measured rather than argued: same input times k, look at how much the output
moves. Only the head is instantiated here; the rest of the network does not
affect the property.

    python ablation/check_head_scale.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
feat = torch.rand(1, 32, 16, 8, 16)      # what arrives from the decoder


def head(with_norm):
    torch.manual_seed(0)
    conv = nn.Conv3d(32, 2, kernel_size=1)
    norm = nn.InstanceNorm3d(2) if with_norm else nn.Identity()
    act = nn.LeakyReLU(0.3) if with_norm else nn.Identity()
    final = nn.Conv3d(2, 1, kernel_size=1)
    return lambda z: final(act(norm(conv(z))))


for label, with_norm in (("with InstanceNorm+LeakyReLU (ac70107, = arm A3)", True),
                         ("without, i.e. the head of the paper (= arm A0)", False)):
    f = head(with_norm)
    with torch.no_grad():
        base = f(feat)
        print(f"\n{label}")
        print(f"  {'k':>6} {'max(out)':>14} {'max(out)/max(out@k=1)':>24}")
        for k in (1.0, 2.0, 5.0):
            out = f(feat * k)
            print(f"  {k:6.1f} {float(out.amax()):14.6f} "
                  f"{float(out.amax() / base.amax()):24.6f}")

print("""
How to read it: if the ratio column stays at 1.000000 as k grows, the output no
longer depends on the scale of the input -> the head is blind to scale, and
final_conv can only emit a scale fixed by the weights, never one set by the
volume.
""")
