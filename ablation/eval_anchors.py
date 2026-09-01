#!/usr/bin/env python
"""Table 1 under both normalisation anchors: the maximum against the 99.9th percentile.

Same predictions, same volumes, same gamma call; only the scalar each field is
divided by changes.

  max    each field divided by its own maximum. This is the convention of the paper.
  p99.9  each field divided by its own 99.9th percentile over the whole volume.

Both the refined and the baseline row are scored under both anchors, because the
anchor belongs to the SCORING and not to the model: applying it to the refined
row alone reads better than either convention and is the one thing review will
not allow.

Two safeguards worth knowing about, both of which caught real mistakes:

1. The checkpoint key. Checkpoints written by ablation/train_arm.py use "model";
   the UAQ ones use "model_state_dict" or "state_dict". With strict=False a key
   that is not recognised loads NO weights at all and the script would happily
   score a randomly initialised network, printing only a count nobody reads. It
   now refuses to run if a single tensor is missing.

2. The head. A checkpoint written before commit ac70107 has no norm.* tensors,
   because that head is conv(->2) -> final_conv(->1) with nothing in between. The
   script detects this from the state dict and puts nn.Identity in place of norm
   and leaky_relu, so the network that is scored is the network the weights were
   trained for.

    python ablation/eval_anchors.py --test-dir <root>/test --net-dir . \
        --ckpt <...>.pth --out-csv anchors.csv --n 175 --voxel 4.0
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pymedphys
import torch
import torch.nn as nn

DOSE_CUTOFF = 0.2


def gamma_gpr(pred, target, pct, dta, voxel):
    d, h, w = target.shape
    coords = (np.arange(d) * voxel, np.arange(h) * voxel, np.arange(w) * voxel)
    ref_max = target.max()
    if ref_max < 1e-10:
        return 100.0
    valid = target >= DOSE_CUTOFF * ref_max
    if valid.sum() == 0:
        return 100.0
    gmap = pymedphys.gamma(
        coords, target, coords, pred,
        dose_percent_threshold=pct, distance_mm_threshold=dta,
        lower_percent_dose_cutoff=DOSE_CUTOFF * 100, max_gamma=10.0,
        local_gamma=False, global_normalisation=ref_max,
    )
    g = gmap[valid]
    g = g[~np.isnan(g)]
    return float((g < 1.0).sum() / len(g) * 100)


def anchor(a, kind):
    return float(a.max()) if kind == "max" else float(np.percentile(a, 99.9))


def load_sample(test_dir: Path, i: int):
    ct = np.load(test_dir / "ct" / f"CT_{i:03d}.npy").astype(np.float32)
    r5k = np.load(test_dir / "dose_5k" / f"{i:03d}_5k.npy").astype(np.float32)
    r1m = np.load(test_dir / "dose_1m" / f"{i:03d}_1M.npy").astype(np.float32)
    ct_norm = (ct + 1000.0) / 3000.0
    d5k = (r5k - r5k.min()) / (r5k.max() - r5k.min())
    d1m = (r1m - r1m.min()) / (r1m.max() - r1m.min())
    x = torch.from_numpy(np.stack([ct_norm, d5k], axis=0)[None]).float()
    return x, d5k, d1m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--net-dir", default=".", help="directory holding network_2_1.py")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n", type=int, default=175)
    ap.add_argument("--voxel", type=float, default=4.0)
    ap.add_argument("--shape", default="128,16,128")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    shape = tuple(int(v) for v in a.shape.split(","))
    sys.path.insert(0, a.net_dir)
    from network_2_1 import unet
    from network_elements import Expansion, MultiScaleElaboration, Reduction

    dev = a.device
    for _cls in (MultiScaleElaboration, Reduction, Expansion):
        _orig = _cls.__init__

        def _mk(_o):
            def _init(self, *args, **kw):
                kw["device"] = dev
                _o(self, *args, **kw)
            return _init
        _cls.__init__ = _mk(_orig)

    test_dir = Path(a.test_dir)
    print(f"test set: {test_dir} ({a.n} volumes) | ckpt: {Path(a.ckpt).name}", flush=True)

    model = unet(input_shape=shape).to(dev)
    with torch.no_grad():
        model(torch.randn(1, 2, *shape, device=dev))

    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

    if not any(k.startswith("norm.") for k in sd) and hasattr(model, "norm"):
        model.norm = nn.Identity()
        model.leaky_relu = nn.Identity()
        print("checkpoint predates ac70107: norm and leaky_relu replaced by Identity",
              flush=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"weights loaded: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    if missing:
        raise SystemExit(
            f"REFUSING TO SCORE: {len(missing)} tensors were not loaded "
            f"(first: {list(missing)[:5]}). With strict=False a randomly initialised "
            f"network would produce plausible and false numbers.")
    model.eval()

    criteria = [(2.0, 2.0), (1.0, 1.0)]
    anchors = ["max", "p99.9"]
    cols = ["sample"]
    for which in ("ref", "base"):
        for k in anchors:
            for p, _ in criteria:
                cols.append(f"{which}_{k}_{p:g}pct")
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    acc = {c: [] for c in cols[1:]}

    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        t0 = time.time()
        for i in range(1, a.n + 1):
            x, d5k, d1m = load_sample(test_dir, i)
            with torch.no_grad():
                pred = model(x=x.to(dev))[0, 0].cpu().numpy()

            row = {"sample": i}
            for which, field in (("ref", pred), ("base", d5k)):
                for k in anchors:
                    p_ = field / (anchor(field, k) + 1e-10)
                    r_ = d1m / (anchor(d1m, k) + 1e-10)
                    for pct, dta in criteria:
                        key = f"{which}_{k}_{pct:g}pct"
                        v = gamma_gpr(p_, r_, pct, dta, a.voxel)
                        row[key] = round(v, 4)
                        acc[key].append(v)
            w.writerow(row)
            fh.flush()
            el = time.time() - t0
            print(f"{i}/{a.n}  ref max {row['ref_max_1pct']:6.2f} -> p99.9 "
                  f"{row['ref_p99.9_1pct']:6.2f} | base max {row['base_max_2pct']:6.2f} -> "
                  f"p99.9 {row['base_p99.9_2pct']:6.2f}   ({el / i:.1f} s/vol, "
                  f"ETA {(a.n - i) * el / i / 60:.0f} min)", flush=True)

    print()
    for k in cols[1:]:
        arr = np.array(acc[k])
        print(f"== {k:18s} {arr.mean():6.2f} +/- {arr.std(ddof=1):5.2f}  "
              f"(min {arr.min():6.2f}, below 90%: {int((arr < 90).sum())}, n={len(arr)})")
    for which in ("ref", "base"):
        for pct, _ in criteria:
            d = (np.array(acc[f"{which}_p99.9_{pct:g}pct"])
                 - np.array(acc[f"{which}_max_{pct:g}pct"]))
            print(f"paired {which} {pct:g}%: {d.mean():+.2f} pp  better "
                  f"{int((d > 0).sum())}/{len(d)}  worse {int((d < 0).sum())}  "
                  f"worst case {d.min():+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
