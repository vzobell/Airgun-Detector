"""Make a review set from a deployment run, to check a detector's output by eye.

    python make_check_set.py <results folder> <set name> [--n 32] [--seed 1] [--after YYYY-MM-DD] [--before YYYY-MM-DD]

    python make_check_set.py "F:/GOM/.../GOM_GC_15_AirgunDetections" check_GC_15

<results folder> holds the .mat files written by a detector for one deployment
(the original's layout: allSmpPts, rawStart, ...). Five-minute windows are
drawn at random from the files found there, a quarter from each of four kinds:
    no detections,  1 to 10,  11 to 30,  more than 30 detections in the window
so that both spare detections and missed shots can show up. The detections
become the starting marks. Open the set with
    review_marks('<set name>')
and correct the marks; compare_check_set.py then reports how the detector did.

The set is written to evaluation/<set name>/ and is not a test or training
set: its windows were chosen by looking at the detector's own output.
"""
import argparse
import csv
import datetime
import glob
import os

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
FS, SEG, WIN_SEGS = 2000, 150000, 4
KINDS = [("no detections", 0, 0), ("1 to 10 detections", 1, 10), ("11 to 30 detections", 11, 30),
         ("more than 30 detections", 31, 10 ** 9)]

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("name")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--after", default=None)
ap.add_argument("--before", default=None)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

blocks = {k[0]: [] for k in KINDS}
dets = {}
files = sorted(glob.glob(os.path.join(a.results, "*.mat")))
for p in files:
    m = sio.loadmat(p)
    name = os.path.basename(p)[:-4] + ".x.wav"
    site = "_".join(name.split("_")[1:3])
    raw = m["rawStart"]
    smp = m["allSmpPts"][:, 0] - 1 if m["allSmpPts"].size else np.zeros(0)
    dets[name] = smp
    n_seg = raw.shape[0]
    dur = np.ravel(m["rawDur"])
    count = np.bincount((smp // (SEG * WIN_SEGS)).astype(int), minlength=n_seg // WIN_SEGS + 1)
    for b in range(n_seg // WIN_SEGS):
        s0 = b * WIN_SEGS
        if not np.allclose(dur[s0:s0 + WIN_SEGS], SEG / FS):
            continue                                              # only whole 75 s segments
        r = raw[s0]
        t = datetime.datetime(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
        day = t.strftime("%Y-%m-%d")
        if (a.after and day < a.after) or (a.before and day >= a.before):
            continue
        for kind, lo, hi in KINDS:
            if lo <= count[b] <= hi:
                blocks[kind].append((site, name, s0 + 1, t, int(count[b])))

chosen = []
per = a.n // len(KINDS)
for kind, _, _ in KINDS:
    pool = blocks[kind]
    idx = rng.permutation(len(pool))[:per]
    chosen += [(kind,) + pool[i] for i in idx]
    print("%-24s %6d windows available, %d chosen" % (kind, len(pool), len(idx)))
chosen.sort(key=lambda c: c[4])

out = os.path.join(HERE, a.name)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "windows.csv"), "w", newline="") as fw, open(os.path.join(out, "picks.csv"), "w", newline="") as fp:
    cw, cp = csv.writer(fw), csv.writer(fp)
    cw.writerow(["window", "condition_guess", "site", "file", "start_segment", "start_time", "proposed_shots"])
    cp.writerow(["window", "shot", "time_s"])
    for k, (kind, site, name, seg, t, n) in enumerate(chosen):
        wid = "C%03d" % (k + 1)
        s0 = (seg - 1) * SEG
        ts = (dets[name][(dets[name] >= s0) & (dets[name] < s0 + SEG * WIN_SEGS)] - s0) / FS
        cw.writerow([wid, kind, site, name, seg, t.strftime("%Y-%m-%d %H:%M:%S"), len(ts)])
        cp.writerows([[wid, j + 1, "%.2f" % x] for j, x in enumerate(np.sort(ts))])
for f in ("truth.csv", "review.csv"):                     # start the review from the detector's marks
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))
print("%d windows from %d files written to %s" % (len(chosen), len(files), out))
print("days covered: %s to %s" % (chosen[0][4].strftime("%Y-%m-%d"), chosen[-1][4].strftime("%Y-%m-%d")))
