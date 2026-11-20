"""Precision-recall curves of the network architectures tried, per shot.

    python plot_arch.py

Each version is four copies trained like the network in use (train1-3, 80
epochs, --mix 0.3, seeds 1-4) and averaged. Scored on set1 and set2 together;
the six deployment sets are not used, so they stay an untouched test. The
threshold is stepped from 0.05 to 0.95; the large open marker is 0.40, the
threshold in use.

Writes figures/PR_curves_architectures.png and prints the numbers at 0.40.
"""
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

SETS = ("set1", "set2")
USED = 0.40
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
# name, label, colour, marker - fixed order, the version in use first
VERSIONS = [
    ("network_airgun", "In use: dilated, ±25 s context (208k parameters)", "#2a78d6", "o"),
    ("arch_small_seed", "Half the filters (53k)", "#eb6834", "s"),
    ("arch_crnn_seed", "Recurrent (GRU) instead of dilated (82k)", "#1baf7a", "^"),
    ("arch_large_seed", "Double the filters (828k)", "#eda100", "D"),
    ("arch_short_seed", "Short context, ±0.3 s (105k)", "#e87ba4", "v"),
]

ws = [w for s in SETS for w in data.cache_set(s)]
curves = {}
for name, *_ in VERSIONS:
    models, _ = ensemble.load(name)
    assert models, name
    probs = [ensemble.probability(models, w["S"]) for w in ws]
    pts = []
    for thr in THRS:
        _, rows = score.score_set(ws, {w["window"]: score.pick(p, thr) for w, p in zip(ws, probs)})
        shots = sum(r["shots"] for r in rows)
        found = sum(r["found"] for r in rows)
        det = sum(r["detections"] for r in rows)
        spare = sum(r["spare"] for r in rows)
        pts.append((found / shots, (det - spare) / det if det else np.nan, found, shots, det, spare))
    curves[name] = np.array(pts)

shots = int(curves["network_airgun"][0, 3])
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, zoom in zip(axes, (False, True)):
    for name, label, col, mk in VERSIONS:
        c = curves[name]
        ax.plot(c[:, 0], c[:, 1], "-", color=col, lw=2, marker=mk, ms=5, label=label)
        k = int(np.argmin(abs(THRS - USED)))
        ax.plot(c[k, 0], c[k, 1], mk, ms=13, mfc="none", mec=col, mew=2)
    ax.set_xlim(*((0.7, 1.005) if zoom else (0, 1.02)))
    ax.set_ylim(*((0.7, 1.005) if zoom else (0, 1.02)))
    ax.set_xlabel("Recall (shots found / shots marked)")
    ax.set_title("Enlarged" if zoom else "Sets 1 and 2: %d shots in %d windows" % (shots, len(ws)),
                 fontsize=11, loc="left", color="0.2")
    ax.grid(color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.6")
    ax.tick_params(colors="0.35")
axes[0].set_ylabel("Precision (detections on a shot / all detections)")
axes[0].plot([], [], "o", ms=12, mfc="none", mec="0.3", mew=2, label="Threshold 0.40 (in use)")
axes[0].legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(HERE, "figures", "PR_curves_architectures.png")
fig.savefig(out, dpi=200)
print("saved", out)
k = int(np.argmin(abs(THRS - USED)))
for name, label, *_ in VERSIONS:
    r, p, f, s, d, sp = curves[name][k]
    print("%-16s at 0.40: found %d of %d, %d detections, %d spare -> recall %.2f precision %.2f" % (name, f, s, d, sp, r, p))
