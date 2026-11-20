"""Precision-recall curves per site (per shot), for the windows of one test set.

    python plot_pr_site.py set3

Needs pr_curve_sites.csv (written by sweep_pr.m) and the network in
cnn/models/network_airgun.pt. One column per site of the set; the bottom row
enlarges the upper right corner. The large open marker is the setting in use.

    recall    = shots found / shots marked
    precision = detections that hold a shot / all detections

Writes figures/PR_curves_<set>_by_site.png and prints the numbers at the
settings in use.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cnn"))
import data          # noqa: E402
import score         # noqa: E402
import ensemble      # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "set3"
COL = {"original": "#D55E00", "prototype": "#0072B2", "cnn": "#009E73"}
LABEL = {"original": "Original", "prototype": "Shape-based cross-correlation", "cnn": "Network (CNN)"}
MARK = {"original": "o", "prototype": "s", "cnn": "^"}
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def pr(p):
    rec = p["found"] / max(p["shots"], 1)
    prec = (p["detections"] - p["spare"]) / p["detections"] if p["detections"] else np.nan
    return rec, prec


mat = {}
for r in csv.DictReader(open(os.path.join(HERE, "pr_curve_sites.csv"))):
    if r["set"] != SET:
        continue
    p = {k: int(float(r[k])) for k in ("shots", "detections", "found", "spare", "step", "current")}
    mat.setdefault((r["detector"], r["site"]), []).append(p)
sites = sorted({k[1] for k in mat})

models, thrs = ensemble.load("network_airgun")
used = ensemble.threshold_of(thrs)
ws = data.cache_set(SET)
probs = {w["window"]: ensemble.probability(models, w["S"]) for w in ws}
for site in sites:
    sub = [w for w in ws if w["site"] == site]
    pts = []
    for k, thr in enumerate(THRS):
        _, rows = score.score_set(sub, {w["window"]: score.pick(probs[w["window"]], thr) for w in sub})
        pts.append(dict(step=k, current=int(abs(thr - used) < 1e-6), shots=sum(r["shots"] for r in rows),
                        detections=sum(r["detections"] for r in rows), found=sum(r["found"] for r in rows),
                        spare=sum(r["spare"] for r in rows)))
    mat[("cnn", site)] = pts

fig, axes2 = plt.subplots(2, len(sites), figsize=(5.2 * len(sites), 10.4), sharey="row", squeeze=False)
for axes, zoom in zip(axes2, (False, True)):
    for ax, site in zip(axes, sites):
        for det in ("original", "prototype", "cnn"):
            pts = sorted(mat[(det, site)], key=lambda p: p["step"])
            xy = np.array([pr(p) for p in pts])
            ax.plot(xy[:, 0], xy[:, 1], "-", color=COL[det], lw=2, marker=MARK[det], ms=4, label=LABEL[det])
            k = [i for i, p in enumerate(pts) if p["current"]][0]
            ax.plot(xy[k, 0], xy[k, 1], MARK[det], ms=12, mfc="none", mec=COL[det], mew=2)
        n_win = sum(1 for w in ws if w["site"] == site)
        ax.set_title("%s: %d shots in %d windows%s" % (site.replace("_", " "), mat[("original", site)][0]["shots"], n_win,
                                                       ", enlarged" if zoom else ""), fontsize=11, loc="left")
        ax.set_xlabel("Recall (shots found / shots marked)")
        ax.set_xlim(*((0.5, 1.005) if zoom else (0, 1.02)))
        ax.set_ylim(*((0.6, 1.005) if zoom else (0, 1.02)))
        ax.grid(color="0.9", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Precision (detections on a shot / all detections)")
axes2[0][0].plot([], [], "o", ms=11, mfc="none", mec="0.3", mew=2, label="Setting in use")
axes2[0][0].legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "figures", "PR_curves_%s_by_site.png" % SET)
fig.savefig(out, dpi=170)
print("saved", out)
for site in sites:
    for det in ("original", "prototype", "cnn"):
        p = [q for q in mat[(det, site)] if q["current"]][0]
        print("%s %-9s in use: found %d of %d, %d detections, %d spare -> recall %.2f precision %.2f" %
              (site, det, p["found"], p["shots"], p["detections"], p["spare"], *pr(p)))
