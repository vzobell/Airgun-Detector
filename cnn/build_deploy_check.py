"""Draw a review set spread across one whole deployment, to measure a detector there.

    python build_deploy_check.py GC_15 deploy_GC_15 [--per-month 4]

  - Random five-minute windows, a fixed number per calendar month, so the
    whole deployment is covered. At most one window per day.
  - Only days not used by any labelled set (test sets 1-3, training 1-3).
  - No detector takes part in choosing them.
  - Candidate windows come from the deployment scan (evaluation/scans) when
    there is one, otherwise from the xwav headers (four 75 s raw files in a
    row with no gap).
  - The network's detections at a low threshold (0.15) are written as one
    source of starting marks, so faint candidates are shown too. The other two
    detectors are added by build_deploy_check.sh and pooled by make_proposals.py.

Writes evaluation/<set name>/windows.csv and detections_proposals_cnn.csv.
"""
import argparse
import csv
import datetime
import os
from collections import defaultdict

import numpy as np
import scipy.io as sio
from scipy import signal

import data
import ensemble
import score
from run_network import read_xwav_header
from select_windows import datenum_to_dt

ap = argparse.ArgumentParser()
ap.add_argument("site")
ap.add_argument("name")
ap.add_argument("--per-month", type=int, default=4)
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=20261118)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

used = set()
for folder in ("set1", "set2", "set3", "train1", "train2", "train3"):
    for r in csv.DictReader(open(os.path.join(data.TESTS, folder, "windows.csv"))):
        used.add((r["site"], r["start_time"][:10]))

paths = data.file_index({a.site})
scan = os.path.join(data.TESTS, "scans", {"CE_03": "scan_ce03.mat"}.get(a.site, "scan_GOM_%s.mat" % a.site))
cand = []        # (file name, first segment, start time, background level or None)
if os.path.exists(scan):
    m = sio.loadmat(scan, squeeze_me=True, struct_as_record=False)
    R, fl = m["R"], m["fl"]
    for i in np.where(R[:, -1] == data.SEG)[0]:
        cand.append((fl[int(R[i, 0]) - 1].name, int(R[i, 1]), datenum_to_dt(R[i, 2]), float(R[i, 6])))
else:            # no scan of this deployment: take the windows from the xwav headers
    for name in sorted(n for n in paths if n.startswith("GOM_%s_" % a.site)):
        _, _, raws = read_xwav_header(paths[name])
        for k in range(len(raws) - data.WIN_SEGS + 1):
            r = raws[k:k + data.WIN_SEGS]
            if all(x["n"] == data.SEG and x["first"] == (k + j) * data.SEG for j, x in enumerate(r)) and \
               (r[-1]["time"] - r[0]["time"]).total_seconds() == (data.WIN_SEGS - 1) * data.SEG / data.FS:
                cand.append((name, k + 1, r[0]["time"], None))
    print("%d windows of continuous recording found in the headers" % len(cand))
by_month = defaultdict(list)
for c in cand:
    if (a.site, c[2].strftime("%Y-%m-%d")) not in used:
        by_month[c[2].strftime("%Y-%m")].append(c)

models, _ = ensemble.load("network_airgun")
chosen = []
for month in sorted(by_month):
    taken, n = set(), 0
    for i in rng.permutation(len(by_month[month])):
        name, seg, t0, bg = by_month[month][i]
        day = t0.strftime("%Y-%m-%d")
        if day in taken:
            continue
        w = dict(site=a.site, file=name, path=paths[name], seg=seg, start=t0.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            y = data.read_window(w)
            if bg is None:        # median band level 5-250 Hz, dB re counts^2/Hz
                f, p = signal.welch(y - y.mean(), fs=data.FS, nperseg=data.NPERSEG)
                bg = 10 * np.log10(np.median(p[(f >= data.FMIN) & (f <= data.FMAX)]) + 1e-12)
            w["background"] = bg
            S = data.spectrogram(y)
        except Exception as err:
            print("  skipped %s seg %d: %s" % (name, w["seg"], err))
            continue
        w["det"] = score.pick(ensemble.probability(models, S), a.threshold)
        chosen.append(w)
        taken.add(day)
        n += 1
        if n == a.per_month:
            break
    print("%s: %d windows from %d unused blocks" % (month, n, len(by_month[month])), flush=True)

out = os.path.join(data.TESTS, a.name)
os.makedirs(out, exist_ok=True)
chosen.sort(key=lambda w: w["start"])
with open(os.path.join(out, "windows.csv"), "w", newline="") as f:
    c = csv.writer(f)
    c.writerow(["window", "condition_guess", "site", "file", "start_segment", "start_time", "proposed_shots"])
    for k, w in enumerate(chosen):
        w["window"] = "G%02d" % (k + 1)
        c.writerow([w["window"], "background %.0f dB" % w["background"], w["site"], w["file"], w["seg"], w["start"], 0])
with open(os.path.join(out, "detections_proposals_cnn.csv"), "w", newline="") as f:
    c = csv.writer(f)
    c.writerow(["window", "start_s", "end_s"])
    for w in chosen:
        c.writerows([[w["window"], "%.2f" % s0, "%.2f" % s1] for s0, s1 in w["det"]])
with open(os.path.join(out, "truth.csv"), "w", newline="") as f:
    f.write("window,time_s,source\n")
bg = sorted(w["background"] for w in chosen)
print("%d windows written to %s (background level %.0f to %.0f dB, median %.0f)" % (len(chosen), out, bg[0], bg[-1], bg[len(bg) // 2]))
