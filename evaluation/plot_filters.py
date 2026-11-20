"""Show the 16 first-layer filters of the network, and what each one picks out.

    python plot_filters.py [set] [window] [start_s] [length_s]

Left: each filter's 3 x 3 pattern (time across, frequency up; red = adds,
blue = subtracts). Right: that filter's picture of a real spectrogram, after
negatives are set to zero (bright = pattern found there). Uses the first of
the four copies in cnn/models/network_airgun.pt.
Writes figures/M_filters.png.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cnn"))
import data          # noqa: E402
import ensemble      # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "set1"
WIN = sys.argv[2] if len(sys.argv) > 2 else "W05"
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
LEN = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

models, _ = ensemble.load("network_airgun")
first = models[0].front[0].net                     # Conv2d, BatchNorm, ReLU
W = first[0].weight.detach().numpy()[:, 0]         # (16, 3 frequency, 3 time)
w = [x for x in data.cache_set(SET) if x["window"] == WIN][0]
a, b = int(T0 / data.HOP_S), int((T0 + LEN) / data.HOP_S)
with torch.no_grad():
    R = first(torch.from_numpy(w["S"]).unsqueeze(0).unsqueeze(0))[0].numpy()   # (16, 99, T)

fig = plt.figure(figsize=(13, 15))
gs = fig.add_gridspec(17, 2, width_ratios=[1, 9], hspace=0.35, wspace=0.04, left=0.06, right=0.98, top=0.965, bottom=0.03)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(w["S"][:, a:b] * 20, origin="lower", aspect="auto", cmap="viridis", vmin=-10, vmax=40,
          extent=[T0, T0 + LEN, data.FMIN, data.FMAX])
ax.set_xticks([])
ax.set_ylabel("Hz", fontsize=8)
ax.tick_params(labelsize=7)
ax.set_title("Input: spectrogram (%s %s, %.0f to %.0f s)" % (SET, WIN, T0, T0 + LEN), loc="left", fontsize=10)
fig.add_subplot(gs[0, 0]).axis("off")
lim = np.abs(W).max()
for k in range(16):
    ax = fig.add_subplot(gs[k + 1, 0])
    ax.imshow(W[k], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("%d" % (k + 1), rotation=0, labelpad=10, fontsize=9, va="center")
    if k == 0:
        ax.set_title("Pattern", fontsize=9)
    ax = fig.add_subplot(gs[k + 1, 1])
    r = R[k][:, a:b]
    ax.imshow(r, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=max(np.percentile(r, 99.5), 1e-6),
              extent=[T0, T0 + LEN, data.FMIN, data.FMAX])
    ax.set_yticks([])
    if k < 15:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.tick_params(labelsize=7)
    if k == 0:
        ax.set_title("What each filter picks out (bright = pattern found)", loc="left", fontsize=10)
out = os.path.join(HERE, "figures", "M_filters.png")
fig.savefig(out, dpi=130)
print("saved", out)
