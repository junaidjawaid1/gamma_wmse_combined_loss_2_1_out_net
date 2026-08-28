#!/usr/bin/env python3
"""Why the p99.9 anchor changes anything at all.

The gamma index is invariant to a COMMON rescaling of the two fields: the
tolerance is derived from the reference's maximum, so scaling both by the same
constant leaves the ratio alone. Switching from `amax` to the 99.9th percentile
is not a common rescaling: each field is divided by its own statistic, and the
two fields do not have the same peak-to-percentile ratio.

This measures that difference. It is the entire size of the effect.
"""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="split directory with dose_5k/ and dose_1m/")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    r5, r1 = [], []
    for i in range(1, a.n + 1):
        f5 = os.path.join(a.data, "dose_5k", f"{i:03d}_5k.npy")
        f1 = os.path.join(a.data, "dose_1m", f"{i:03d}_1M.npy")
        if not (os.path.exists(f5) and os.path.exists(f1)):
            continue
        x = np.load(f5).astype(np.float64)
        y = np.load(f1).astype(np.float64)
        r5.append(x.max() / np.percentile(x, 99.9))
        r1.append(y.max() / np.percentile(y, 99.9))

    print(f"n = {len(r5)} volumes\n")
    print(f"  5k field (noisy, the input)     max/p99.9 = {np.mean(r5):6.3f} +- {np.std(r5):.3f}"
          f"   (min {min(r5):.2f}, max {max(r5):.2f})")
    print(f"  1M field (converged, reference) max/p99.9 = {np.mean(r1):6.3f} +- {np.std(r1):.3f}"
          f"   (min {min(r1):.2f}, max {max(r1):.2f})")
    print(f"\n  ratio of the two            = {np.mean(r5)/np.mean(r1):.3f}")
    print("""
Reading. Under the published convention each field is divided by its own maximum.
For the converged reference that maximum is the Bragg peak. For the noisy 5k field
it is a noise spike sitting about eight times above the field's own 99.9th
percentile, against three for the reference. Dividing by it squashes the noisy
field relative to the reference by a factor of about 2.5, and by a factor that
varies volume by volume (the 5k spread is four times the reference's).

That is not a matter of taste in conventions: it is a scaling artefact that
penalises whichever field is noisier, and it penalises it unevenly across volumes.
It predicts what the anchor comparison shows - that the baseline row, which is the
noisy field itself, gains far more from the robust anchor than the refined row,
and that most of the gain is in the tail rather than in the mean.""")


if __name__ == "__main__":
    main()
