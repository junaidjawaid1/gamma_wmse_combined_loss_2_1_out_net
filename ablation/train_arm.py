#!/usr/bin/env python
"""One arm of the UAQ ablation at 4 mm, faithful to combined_loss_train_norm.py
(the 4 mm copy that lives on Kalifano, not the 2 mm one in this repository).

The four arms differ only in the two layers added on 2026-08-25 (commit ac70107)
between the 2-channel and the final 1-channel convolution:

    A0  the paper as it is    neither layer   (= the network on Kalifano)
    A1  LeakyReLU(0.3) only
    A2  InstanceNorm3d only
    A3  both                  (= the network in this repository)

Everything else is identical across arms and identical to the original script:
volumes 128x16x128, VOXEL_SIZE_MM=4.0, gamma 1%/2mm, cutoff 0.2, Adam(1e-5),
ReduceLROnPlateau(max, 0.7, patience 10), beta 0.1->2.0->5.0, autocast bfloat16
+ GradScaler, 1491 training volumes x4 rotations, 154 for validation.

Deliberate differences from the original, all of them stated:
  - no wandb (CSV logging instead: the compute nodes of this cluster have no
    outbound network);
  - the seed is fixed and passed on the command line, identical across arms, so
    the initialisation and the sample order are the same and the only thing that
    differs between two arms is the architecture;
  - the number of epochs is a parameter: the ablation compares arms against each
    other at equal budget, and the budget is chosen in advance, not afterwards;
  - GPR is computed on validation only, not on every training batch: it is a
    metric, it does not enter the gradient, and per batch it would cost time
    without changing what is being compared;
  - data paths on the command line.

Resuming across jobs. 150 epochs is about 79 hours on one A100 and the queue
caps a job at 24 hours, so the run has to be chained. Every epoch writes
`last.pth` with all the state needed to continue, and on restart the script
picks it up by itself (`--no-resume` to ignore it).

What `last.pth` holds, and why each piece is there:
  - `model`      the weights;
  - `opt`        the Adam moments: without them the first step after the restart
                 is the step of a brand-new optimiser, and it shows in the curve;
  - `lr_sched`   ReduceLROnPlateau carries its own best and its count of epochs
                 without improvement: resetting it delays the learning-rate drop;
  - `scaler`     the GradScaler scale, which does recover on its own but wastes
                 a few steps doing so;
  - `criterion`  the piece that is easiest to forget. It holds two registered
                 buffers: `wmse_avg`, the running average by which the WMSE term
                 normalises itself (momentum 0.99), and `gamma_loss.beta`.
                 Restarting with `wmse_avg` at zero changes the scale of the loss
                 for tens of steps. `SimpleBetaScheduler`, by contrast, does NOT
                 need saving: `step(ep)` computes beta from the epoch, it is a
                 pure function and restores itself;
  - `best`       the best GPR so far, otherwise the first epoch of the new job
                 overwrites `best.pth` with a worse model;
  - the states of the random generators, the DataLoader one included, so the
                 sample order continues instead of starting over.

The CSV is opened in append mode when resuming: opening it for writing would
truncate the log of the epochs already done.
Note: `peak_gib` restarts from zero in each job (it is the maximum since process
start), so after a resume that column describes the job, not the whole training.

Verified with a three-way parity test (see test_resume.sbatch): an uninterrupted
6-epoch run, the same run as 3+3 with a resume, and a control in which
`wmse_avg` is zeroed in the checkpoint to imitate a naive resume. The resumed run
has to track the uninterrupted one, and the control has to depart from it:
without the control the test cannot tell a working resume from a test with no
power.

Usage:
    train_arm.py --arm A2 --data $DATA/uaq-4mm --out $OUT/A2 \\
                 --epochs 60 --batch 4 --seed 20260825
    # second job, same command line: it resumes from last.pth
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_2_1 import unet
from data_pipeline_128_16_128 import data_pipeline
from losses_opt_1mm import (GammaIndexLoss, SimpleBetaScheduler, WeightedMSE,
                            CombinedWMSEGammaLoss)

# --- the two resolutions, each faithful to its own training script ---
# Careful: BOTH scripts are called combined_loss_train_norm.py, one here on GitHub
# (2 mm) and one on Kalifano (4 mm), and they optimise DIFFERENT gamma criteria.
# The data pipeline, by contrast, is the same file twice: data_pipeline_128_16_128.py
# and data_pipeline_256_32_256.py are byte-identical, the shape comes from the .npy.
RES = {
    "4mm": dict(voxel=4.0, shape=(128, 16, 128), n_train=1491, n_val=154,
                dose_pct=1.0, dta_mm=2.0, batch=4),
    "2mm": dict(voxel=2.0, shape=(256, 32, 256), n_train=787, n_val=77,
                dose_pct=2.0, dta_mm=2.0, batch=2),
}
DOSE_CUTOFF = 0.2


def build_model(arm, device, INPUT_SHAPE):
    """The repository network, with the two new layers switched off per arm.

    Switching them off with nn.Identity, rather than writing four different
    networks, keeps the rest of the graph identical, and with it the order in
    which the random generator is consumed: two arms start from the same weights.
    """
    model = unet(input_shape=INPUT_SHAPE)
    if arm in ("A0", "A1"):
        model.norm = nn.Identity()
    if arm in ("A0", "A2"):
        model.leaky_relu = nn.Identity()
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A0", "A1", "A2", "A3"])
    ap.add_argument("--res", default="4mm", choices=["4mm", "2mm"],
                    help="resolution. Sets shape, voxel size, gamma criterion, example "
                         "counts and default batch, each faithful to the training script "
                         "for that resolution.")
    ap.add_argument("--data", required=True, help="radice con train/ e validation/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=0, help="0 = the default for the chosen resolution")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--wmse-alpha", type=float, default=1.0,
                    help="alpha of the WMSE weight exp(alpha*y_true). 1.0 is the paper. "
                         "0.0 makes every weight 1, i.e. a plain MSE: that is the "
                         "controlled comparison Section 3.2 never ran.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-anchor", default="max", choices=["max", "p999"],
                    help="scale of the TARGET. 'max' = what the pipeline does today "
                         "(min-max, maximum 1). 'p999' = the change requested by mail: the "
                         "target is re-anchored at its own 99.9th percentile, so its maximum "
                         "rises above 1 (measured: about 3.2x on the 2 mm volumes).")
    ap.add_argument("--out-anchor", default="max", choices=["max", "p999", "none"],
                    help="this is line 205. 'max' = as it is today, the prediction is "
                         "divided by its own maximum. 'none' = drop it, and the network has "
                         "to learn the absolute scale. 'p999' = move it to the 99.9th "
                         "percentile, so prediction and target share the anchor.")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="use only the first N training volumes. FOR TESTS ONLY: it "
                         "shortens an epoch from ~1900 s to a few tens of seconds. A real "
                         "run never uses it.")
    ap.add_argument("--limit-val", type=int, default=0, help="same, for validation")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore last.pth and start from epoch 0 (overwrites the CSV)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="if >0, stop CLEANLY before this limit instead of being killed "
                         "by the batch scheduler in the middle of an epoch")
    args = ap.parse_args()
    R = RES[args.res]
    DOSE_PERCENT_THRESHOLD, DTA_MM_THRESHOLD = R["dose_pct"], R["dta_mm"]
    VOXEL_SIZE_MM, INPUT_SHAPE = R["voxel"], R["shape"]
    TRAIN_EXAMPLES, VAL_EXAMPLES = R["n_train"], R["n_val"]
    if args.batch == 0:
        args.batch = R["batch"]
    t_start = time.perf_counter()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dev = "cuda"

    def build(arm, device):
        return build_model(arm, device, INPUT_SHAPE)

    def anchor(t, mode):
        """Re-anchor a volume (B,1,D,H,W). Returns t unchanged for 'none'.
        The p99.9 is computed per sample, as the evaluation does."""
        if mode == "none":
            return t
        if mode == "max":
            return t / (t.amax(dim=(2, 3, 4), keepdim=True) + 1e-10)
        q = torch.quantile(t.flatten(2).float(), 0.999, dim=2).view(-1, 1, 1, 1, 1)
        return t / (q + 1e-10)

    print("arm %s | res %s | shape %s | voxel %.1f mm | gamma %.0f%%/%.0fmm | batch %d | "
          "torch %s | %s | seed %d"
          % (args.arm, args.res, INPUT_SHAPE, VOXEL_SIZE_MM, DOSE_PERCENT_THRESHOLD,
             DTA_MM_THRESHOLD, args.batch, torch.__version__,
             torch.cuda.get_device_name(0), args.seed), flush=True)

    n_tr = args.limit_train or TRAIN_EXAMPLES
    n_va = args.limit_val or VAL_EXAMPLES
    if args.limit_train or args.limit_val:
        print("WARNING: REDUCED RUN (%d/%d train, %d/%d val): a test, not a measurement"
              % (n_tr, TRAIN_EXAMPLES, n_va, VAL_EXAMPLES), flush=True)
    train_ds = data_pipeline(path=os.path.join(args.data, "train/"),
                             index_list=[f"{i:03d}" for i in range(1, n_tr + 1)],
                             threshold=0.1, refine=False)
    val_ds = data_pipeline(path=os.path.join(args.data, "validation/"),
                           index_list=[f"{i:03d}" for i in range(1, n_va + 1)],
                           threshold=0.1, refine=False)
    g = torch.Generator()
    g.manual_seed(args.seed)          # same sample order across all arms
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=g,
                          num_workers=args.workers, pin_memory=True, prefetch_factor=4,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, prefetch_factor=4,
                        persistent_workers=True)

    model = build(args.arm, dev)
    with torch.no_grad():                                  # initialise the Lazy* layers
        model(x=torch.randn(1, 2, *INPUT_SHAPE, device=dev))
    n_par = sum(p.numel() for p in model.parameters())

    wmse = WeightedMSE(alpha=args.wmse_alpha).to(dev)
    gamma = GammaIndexLoss(dose_percent=DOSE_PERCENT_THRESHOLD, dta_mm=DTA_MM_THRESHOLD,
                           voxel_size_mm=(VOXEL_SIZE_MM,) * 3, dose_cutoff=DOSE_CUTOFF,
                           beta_init=0.1, max_gamma=10.0).to(dev)
    criterion = CombinedWMSEGammaLoss(wmse_loss=wmse, gamma_loss=gamma, wmse_weight=1.0,
                                      gamma_weight=1.0, outside_weight=0.0,
                                      outside_threshold=0.0, momentum=0.99).to(dev)
    beta_sched = SimpleBetaScheduler(loss_fn=gamma, beta_start=0.1, beta_mid=2.0,
                                     beta_end=5.0, warmup_iters=10, phase1_end=50,
                                     phase2_end=100, update_interval=5)
    opt = optim.Adam(model.parameters(), lr=1e-5)
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.7,
                                                   patience=10, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    last_path = os.path.join(args.out, "last.pth")
    start_ep, best = 0, -1.0
    if os.path.exists(last_path) and not args.no_resume:
        # map_location="cpu" and not dev: the generator states are ByteTensors and
        # `torch.set_rng_state` rejects them if they arrive on CUDA. The
        # load_state_dict calls move things to the parameters' device themselves.
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if (ck.get("arm") != args.arm or ck.get("seed") != args.seed
                or ck.get("wmse_alpha", 1.0) != args.wmse_alpha):
            sys.exit("last.pth belongs to another run (arm=%s seed=%s wmse_alpha=%s): not "
                     "resuming. Use --no-resume or a different --out."
                     % (ck.get("arm"), ck.get("seed"), ck.get("wmse_alpha", 1.0)))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        lr_sched.load_state_dict(ck["lr_sched"])
        scaler.load_state_dict(ck["scaler"])
        criterion.load_state_dict(ck["criterion"])
        best = ck["best"]
        start_ep = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_torch"].cpu())
        np.random.set_state(ck["rng_numpy"])
        if ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        g.set_state(ck["rng_loader"].cpu())
        print("RESUMED from %s: epoch %d, best GPR so far %.4f%%, wmse_avg %.6f"
              % (last_path, start_ep, best,
                 float(criterion.wmse_avg)), flush=True)

    csv_path = os.path.join(args.out, "log.csv")
    if start_ep == 0:
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "beta", "lr", "train_loss", "val_loss",
                                     "val_gpr", "sec", "peak_gib"])
    elif not os.path.exists(csv_path):
        sys.exit("resuming from epoch %d but %s is missing: the log is gone, and "
                 "stopping beats writing a CSV with a hole in it." % (start_ep, csv_path))

    if start_ep >= args.epochs:
        print("already complete: %d epochs out of the %d requested" % (start_ep, args.epochs), flush=True)
        return

    ep = start_ep - 1
    for ep in range(start_ep, args.epochs):
        beta = beta_sched.step(ep)
        model.train()
        t0 = time.perf_counter()
        tr_loss = n = 0.0
        for X, Y in train_dl:
            x = X.to(dev, non_blocking=True); y = Y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if args.target_anchor == "p999":
                y = anchor(y, "p999")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x=x)
                out = anchor(out, args.out_anchor)      # <- this is line 205
                loss, _ = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tr_loss += loss.item() * X.size(0); n += X.size(0)
        tr_loss /= max(n, 1)

        model.eval()
        vl = vg = m = 0.0
        with torch.no_grad():
            for X, Y in val_dl:
                x = X.to(dev, non_blocking=True); y = Y.to(dev, non_blocking=True)
                if args.target_anchor == "p999":
                    y = anchor(y, "p999")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x=x)
                    out = anchor(out, args.out_anchor)
                    loss, _ = criterion(out, y)
                    # GPR is not in the loss dict: it is asked of the criterion,
                    # and it is the number used to pick the best model
                    gpr = criterion.compute_pass_rate(out, y)
                if m == 0:      # first batch of the epoch: the scales, for the smoke test
                    scale_t, scale_p = float(y.amax()), float(out.amax())
                vl += loss.item() * X.size(0)
                vg += float(gpr) * X.size(0)
                m += X.size(0)
        vl /= max(m, 1); vg /= max(m, 1)
        lr_sched.step(vg)

        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        dt = time.perf_counter() - t0
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(["%d" % ep, "%.4f" % beta,
                                     "%.2e" % opt.param_groups[0]["lr"],
                                     "%.6f" % tr_loss, "%.6f" % vl, "%.4f" % vg,
                                     "%.1f" % dt, "%.2f" % peak])
        print("ep %3d | beta %.2f | train %.4f | val %.4f | GPR %.2f%% | max(target) %.3f "
              "max(pred) %.3f | %.1f s | picco %.1f GiB"
              % (ep, beta, tr_loss, vl, vg, scale_t, scale_p, dt, peak), flush=True)

        if vg > best:
            best = vg
            torch.save({"model": model.state_dict(), "epoch": ep, "val_gpr": vg,
                        "arm": args.arm, "seed": args.seed, "params": n_par},
                       os.path.join(args.out, "best.pth"))

        # Full state for resuming. Written to a temporary file first and then
        # renamed: if the job is killed during the save, last.pth stays the one
        # from the previous epoch instead of being half written.
        tmp = last_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "lr_sched": lr_sched.state_dict(), "scaler": scaler.state_dict(),
                    "criterion": criterion.state_dict(),
                    "epoch": ep, "best": best, "arm": args.arm, "seed": args.seed,
                    "wmse_alpha": args.wmse_alpha,
                    "rng_torch": torch.get_rng_state(),
                    "rng_numpy": np.random.get_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                    "rng_loader": g.get_state()}, tmp)
        os.replace(tmp, last_path)

        # Wall-clock guard: if the next epoch does not fit, stop here. An arm
        # that ends at a declared 43 epochs beats one killed halfway through the
        # 44th, with the CSV truncated at an arbitrary point.
        if args.max_hours > 0:
            speso = (time.perf_counter() - t_start) / 3600.0
            if speso + (dt / 3600.0) * 1.15 > args.max_hours:
                print("TIME STOP: %.2f h spent out of %.2f, the next epoch "
                      "(~%.2f h) does not fit" % (speso, args.max_hours, dt / 3600.0),
                      flush=True)
                break

    print("END arm %s | epochs completed %d (this job: %d to %d) | "
          "best validation GPR %.4f%%"
          % (args.arm, ep + 1, start_ep, ep, best), flush=True)


if __name__ == "__main__":
    main()
