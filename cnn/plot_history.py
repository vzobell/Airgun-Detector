"""Draw how training went, epoch by epoch, for one or more training runs.

    python plot_history.py hist80 hist160

Reads models/<name>_history.csv (written by train.py) and draws, per run:
  left:  the loss on the training crops and on the held-back windows
  right: the share of held-back shots found, and spare detections per 100
         shots, at a fixed threshold of 0.5 (every 5 epochs)
If the held-back loss is still falling at the last epoch, longer training
may help. If it has flattened, or rises while the training loss keeps
falling, more epochs will not help (the network is memorising).
Writes ../evaluation/figures/M_training.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:]
TRAIN, HELD, SPARE = "#0072B2", "#D55E00", "#666666"

fig, axes = plt.subplots(len(names), 2, figsize=(11, 3.6 * len(names)), squeeze=False)
for row, name in zip(axes, names):
    H = list(csv.DictReader(open(os.path.join(HERE, "models", name + "_history.csv"))))
    ep = [int(r["epoch"]) for r in H]
    ax = row[0]
    ax.plot(ep, [float(r["train_loss"]) for r in H], color=TRAIN, lw=2, label="training crops")
    ax.plot(ep, [float(r["heldback_loss"]) for r in H], color=HELD, lw=2, label="held-back windows")
    ax.set_ylabel("Loss")
    ax.set_title("%s: loss per epoch" % name, loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    S = [r for r in H if r["heldback_found"] != ""]
    e5 = [int(r["epoch"]) for r in S]
    ax = row[1]
    ax.plot(e5, [100.0 * int(r["heldback_found"]) / int(r["heldback_shots"]) for r in S], color=HELD, lw=2, marker="o", ms=4,
            label="held-back shots found (%)")
    ax.plot(e5, [100.0 * int(r["heldback_spare"]) / int(r["heldback_shots"]) for r in S], color=SPARE, lw=2, marker="s", ms=4,
            label="spare detections per 100 shots")
    ax.set_ylim(0, 100)
    ax.set_title("%s: held-back windows at threshold 0.5" % name, loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    for a in row:
        a.set_xlabel("Epoch")
        a.grid(color="0.9", lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    last = H[-1]
    best = min(H, key=lambda r: float(r["heldback_loss"]))
    print("%s: held-back loss lowest at epoch %s (%.4f); at the last epoch %s it is %.4f; training loss %.4f" %
          (name, best["epoch"], float(best["heldback_loss"]), last["epoch"], float(last["heldback_loss"]), float(last["train_loss"])))
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "evaluation", "figures", "M_training.png")
fig.savefig(out, dpi=160)
print("saved", out)
