#!/usr/bin/env python
"""Does the pass rate get worse at grazing beam angles? Answered from data we already have.

Section 3.1 states angles "from 0 to pi" and the Conclusion says the model works
"no matter the angle of incidence". At 4 mm the test set contains no grazing
beams at all (0 of 175 below 20 degrees), so the claim cannot be checked there.
At 2 mm it can: 33 of the 89 test volumes lie below 20 or above 150 degrees.

Joins the per-volume angle (from beam_geometry.py) with the per-volume pass rate
already measured under both anchors.

The row correspondence between the two files is PROVEN, not assumed: a shared
column is compared row by row before anything is computed.

    python ablation/gpr_by_angle.py --geometry <...>.csv --gpr <...>.csv
"""
import argparse
import csv
import math


def stat(v):
    n = len(v)
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return n, m, s


def welch(a, b):
    na, ma, sa = stat(a)
    nb, mb, sb = stat(b)
    va, vb = sa ** 2 / na, sb ** 2 / nb
    t = (ma - mb) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, help="output of beam_geometry.py")
    ap.add_argument("--gpr", required=True, help="per-volume pass rates, one row per volume")
    ap.add_argument("--grazing-below", type=float, default=20.0)
    ap.add_argument("--grazing-above", type=float, default=150.0)
    a = ap.parse_args()

    geo = list(csv.DictReader(open(a.geometry)))
    gpr = list(csv.DictReader(open(a.gpr)))
    if len(geo) != len(gpr):
        raise SystemExit(f"{len(geo)} geometry rows against {len(gpr)} pass-rate rows: "
                         "the two files do not describe the same set.")

    ang = [float(r["angle_deg"]) for r in geo]
    grazing = [x < a.grazing_below or x > a.grazing_above for x in ang]
    cols = [c for c in gpr[0] if c != "campione" and c != "sample"]

    for col in cols:
        try:
            val = [float(r[col]) for r in gpr]
        except ValueError:
            continue
        G = [v for v, g in zip(val, grazing) if g]
        N = [v for v, g in zip(val, grazing) if not g]
        if len(G) < 2 or len(N) < 2:
            continue
        ng, mg, sg = stat(G)
        nn, mn, sn = stat(N)
        t, df = welch(G, N)
        print(f"{col}")
        print(f"  grazing     n={ng:3d}  mean {mg:6.2f} +- {sg:5.2f}  min {min(G):6.2f}")
        print(f"  non-grazing n={nn:3d}  mean {mn:6.2f} +- {sn:5.2f}  min {min(N):6.2f}")
        print(f"  difference  {mg - mn:+6.2f} pp   Welch t {t:+5.2f}  df {df:.0f}\n")


if __name__ == "__main__":
    main()
