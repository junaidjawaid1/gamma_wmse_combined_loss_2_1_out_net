#!/usr/bin/env python
"""Beam angle and field depth of every volume, recovered from the dose field.

The .npy files carry no beam metadata (names are just indices), so anything we
want to know about how a split was sampled has to be deduced from the arrays.

  angle: dose-weighted PCA of the voxels above the cutoff, in the X-Z plane (the
         plane the beams rotate in). The principal axis is the beam direction;
         the angle is defined modulo 180 degrees, as the generation scripts are.
  depth: extent of the field along that axis between the 2nd and 98th percentile
         of the projected dose, in mm.

What it was used for: TOPAS_test.py samples angles with randint(20, 150) while
TOPAS_sim.py (train) uses randint(0, 180). Running this on the three splits
shows the difference in the data itself rather than in the scripts.

NOTE on depth: it does NOT discriminate the beam energy. Run on the validation
split, which is generated at a fixed 150 MeV, it gives the same coefficient of
variation as the training split, which is not - so it is dominated by anatomy.
This is stated here so nobody draws an energy conclusion from that column.

    python ablation/beam_geometry.py --data <root> --split test --voxel 4.0
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import numpy as np


def geometry(d1m: np.ndarray, voxel: float, cutoff: float = 0.2) -> dict:
    ref_max = float(d1m.max())
    valid = d1m >= cutoff * ref_max
    idx = np.argwhere(valid)                       # (n, 3) ordered z, y, x
    if len(idx) < 10:
        return dict(angle_deg=np.nan, depth_mm=np.nan, n_valid=len(idx))
    w = d1m[valid].astype(np.float64)

    pts = np.stack([idx[:, 2], idx[:, 0]], axis=1).astype(np.float64)   # (x, z)
    mean = np.average(pts, axis=0, weights=w)
    c = pts - mean
    cov = (c * w[:, None]).T @ c / w.sum()
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]

    ang = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0

    proj = c @ axis
    order = np.argsort(proj)
    cw = np.cumsum(w[order]) / w.sum()
    lo = proj[order][np.searchsorted(cw, 0.02)]
    hi = proj[order][np.searchsorted(cw, 0.98)]
    return dict(angle_deg=float(ang), depth_mm=float((hi - lo) * voxel),
                n_valid=int(len(idx)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root holding train/ validation/ test/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--voxel", type=float, default=4.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = os.path.join(a.data, a.split, "dose_1m")
    files = sorted((f for f in os.listdir(d) if f.endswith(".npy")),
                   key=lambda f: int(re.search(r"(\d+)", f).group(1)))
    print(f"{len(files)} volumes from {d}")

    rows = []
    for f in files:
        g = geometry(np.load(os.path.join(d, f)).astype(np.float64), a.voxel)
        g["file"] = f
        rows.append(g)

    ang = np.array([r["angle_deg"] for r in rows])
    dep = np.array([r["depth_mm"] for r in rows])
    ang, dep = ang[~np.isnan(ang)], dep[~np.isnan(dep)]

    print(f"\nangle (deg, mod 180)  min {ang.min():6.1f}  max {ang.max():6.1f}  "
          f"mean {ang.mean():6.1f}  sd {ang.std(ddof=1):5.1f}")
    print(f"  below 20 deg: {int((ang < 20).sum())}/{len(ang)}   "
          f"above 150: {int((ang > 150).sum())}/{len(ang)}   "
          f"inside [20,150]: {int(((ang >= 20) & (ang <= 150)).sum())}/{len(ang)}")
    print(f"\ndepth (mm)            min {dep.min():6.1f}  max {dep.max():6.1f}  "
          f"mean {dep.mean():6.1f}  sd {dep.std(ddof=1):5.1f}  "
          f"cv {dep.std(ddof=1) / dep.mean():.3f}   (does NOT see the energy)")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["file", "angle_deg", "depth_mm", "n_valid"])
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nCSV: {a.out}")


if __name__ == "__main__":
    main()
