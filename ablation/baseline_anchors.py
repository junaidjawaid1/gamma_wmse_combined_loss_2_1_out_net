#!/usr/bin/env python
"""The 5k baseline row under both anchors. No network involved.

Section 4 calls the contrast between the two baselines - 87.06 at 4 mm against
12.54 at 2 mm - "the clearest evidence of the noise-resolution trade-off", and
the Conclusion repeats it. Part of that contrast is not noise but normalisation.

Both fields are anchored at their own maximum before scoring, and the maximum of
a 5k field is a noise spike: measured per volume it exceeds the reference peak by
1.14x to 6.12x. Anchoring at the maximum therefore compresses the noisy field by
a factor that varies from volume to volume, and it does so most where the field
is noisiest. At 4 mm, moving to a robust anchor lifts the baseline from 87.22 to
95.41 (+8.19 pp) while the refined row barely moves.

There is no checkpoint, no architecture and no GPU here: the 5k field is compared
with the 1M field, exactly as the baseline row of the tables does. The only thing
that changes between the two columns is the scalar the two fields are divided by.

Gamma convention identical to eval_anchors.py (global on the reference peak in
the current scale, 20% cutoff, max_gamma 10).

    python ablation/baseline_anchors.py --test-dir <root>/test --voxel 2.0 \
        --out-csv baseline_anchors.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pymedphys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--voxel", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    td = Path(a.test_dir)
    files = sorted((f for f in (td / "dose_1m").iterdir() if f.suffix == ".npy"),
                   key=lambda f: int(re.search(r"(\d+)", f.name).group(1)))
    if a.n:
        files = files[: a.n]
    print(f"{len(files)} volumes from {td}", flush=True)

    # The CSV is written INCREMENTALLY, one row per volume. Writing it after the
    # loop means losing everything if the job is killed by the wall-clock limit,
    # which at a few minutes per volume is the likely case, not the rare one.
    fh = open(a.out_csv, "w", newline="")
    wr = None
    rows = []

    for f in files:
        i = f.name.split("_")[0]
        r1m = np.load(f).astype(np.float64)
        r5k = np.load(td / "dose_5k" / f"{i}_5k.npy").astype(np.float64)
        d5k = (r5k - r5k.min()) / (r5k.max() - r5k.min())
        d1m = (r1m - r1m.min()) / (r1m.max() - r1m.min())
        row = {"sample": i,
               "peak_over_p999_5k": d5k.max() / (np.percentile(d5k, 99.9) + 1e-12),
               "peak_over_p999_1m": d1m.max() / (np.percentile(d1m, 99.9) + 1e-12)}
        for kind in ("max", "p99.9"):
            p = d5k / (anchor(d5k, kind) + 1e-12)
            t = d1m / (anchor(d1m, kind) + 1e-12)
            for pct, dta, name in ((2.0, 2.0, "2pct"), (1.0, 1.0, "1pct")):
                row[f"{kind}_{name}"] = gamma_gpr(p, t, pct, dta, a.voxel)
        rows.append(row)
        if wr is None:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            wr.writeheader()
        wr.writerow(row)
        fh.flush()
        print(f"  {i}  max {row['max_2pct']:6.2f}/{row['max_1pct']:6.2f}   "
              f"p99.9 {row['p99.9_2pct']:6.2f}/{row['p99.9_1pct']:6.2f}", flush=True)
    fh.close()

    def st(v):
        n = len(v)
        m = sum(v) / n
        return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))

    print(f"\n=== 5k BASELINE, voxel {a.voxel} mm, n = {len(rows)} ===")
    for c, name in (("max_2pct", "2%/2mm, anchored at the maximum"),
                    ("max_1pct", "1%/1mm, anchored at the maximum"),
                    ("p99.9_2pct", "2%/2mm, anchored at p99.9"),
                    ("p99.9_1pct", "1%/1mm, anchored at p99.9")):
        m, s = st([r[c] for r in rows])
        print(f"  {name:34s} {m:6.2f} +- {s:5.2f}   min {min(r[c] for r in rows):6.2f}")
    r5 = [r["peak_over_p999_5k"] for r in rows]
    r1 = [r["peak_over_p999_1m"] for r in rows]
    print(f"\n  peak/p99.9 of the 5k field:   {min(r5):.2f}x - {max(r5):.2f}x "
          f"(median {sorted(r5)[len(r5) // 2]:.2f})")
    print(f"  peak/p99.9 of the reference:  {min(r1):.2f}x - {max(r1):.2f}x "
          f"(median {sorted(r1)[len(r1) // 2]:.2f})")


if __name__ == "__main__":
    main()
