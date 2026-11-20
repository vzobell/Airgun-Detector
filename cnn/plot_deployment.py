"""Per-day summary of a deployment run, with the recording's background level.

    python plot_deployment.py <results folder> [scan file]

<results folder> is a <deployment>_AirgunDetections folder written by
run_network.py. The optional scan file (evaluation/scans/scan_*.mat) gives the
background level of sample blocks through the deployment, drawn in the bottom
panel so changes in the recording can be seen next to the detections.
Writes <deployment>_per_day.png into the results folder.
"""
import csv
import datetime
import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

folder = sys.argv[1].rstrip("/\\")
name = os.path.basename(folder).replace("_AirgunDetections", "")
rows = list(csv.DictReader(open(os.path.join(folder, name + "_minutes_per_day.csv"))))
day = [datetime.datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
rec = np.array([int(r["minutes_recorded"]) for r in rows])
pos = np.array([int(r["minutes_with_airguns"]) for r in rows])
det = np.array([int(r["detections"]) for r in rows])
share = np.where(rec > 0, 100.0 * pos / np.maximum(rec, 1), np.nan)

n_panels = 3 if len(sys.argv) > 2 else 2
fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.9 * n_panels), sharex=True)
ax = axes[0]
ax.bar(day, det, width=1.0, color="#009E73")
ax.set_ylabel("Detections per day")
ax.set_title("%s: network detections, %d in all, %d days recorded" % (name, det.sum(), (rec > 0).sum()), loc="left", fontsize=11)
ax = axes[1]
ax.bar(day, share, width=1.0, color="#0072B2")
ax.set_ylabel("Minutes with airguns (%)")
ax.set_ylim(0, 100)
if n_panels == 3:
    m = sio.loadmat(sys.argv[2], squeeze_me=True, struct_as_record=False)
    R = m["R"]
    t = [datetime.datetime(1, 1, 1) + datetime.timedelta(days=float(x) - 367.0) for x in R[:, 2]]
    ax = axes[2]
    ax.plot(t, R[:, 6], ".", ms=2, color="#555555")
    ax.set_ylabel("Background level\n(dB re counts)")
    ax.set_title("Background level of sample blocks from the deployment scan", loc="left", fontsize=10)
for ax in axes:
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
fig.tight_layout()
out = os.path.join(folder, name + "_per_day.png")
fig.savefig(out, dpi=150)
print("saved", out)
months = {}
for d, r, p, n in zip(day, rec, pos, det):
    k = d.strftime("%Y-%m")
    a = months.setdefault(k, [0, 0, 0])
    a[0] += r; a[1] += p; a[2] += n
for k, (r, p, n) in sorted(months.items()):
    print("%s  recorded %5.1f days, minutes with airguns %3.0f%%, detections %6d" % (k, r / 1440.0, 100.0 * p / max(r, 1), n))
