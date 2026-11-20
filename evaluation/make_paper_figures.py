"""Figures for the manuscript: the three detectors on the six test deployments.

    python make_paper_figures.py

Needs pr_curve_sites.csv (sweep_pr.m) for the original and shape-based
detectors, the network in cnn/models/network_airgun.pt, the labelled
deploy_<site>_<n> sets, and, for the last figure, the full-deployment results
on F:. Writes to figures/:

  fig_detectors_PR_by_site.png      precision-recall per site, both deployments pooled
  fig_detectors_by_deployment.png   recall and precision at the settings in use
  fig_example_windows.png           one typical window per site, marks and detections
  fig_airguns_per_day.png           share of minutes with airguns, whole deployments

The architecture comparison is plot_arch.py (fig PR_curves_architectures.png).
"""
import csv
import datetime
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cnn"))
import data          # noqa: E402
import score         # noqa: E402
import ensemble      # noqa: E402

FIG = os.path.join(HERE, "figures")
RUNS = r"F:\GOM\GOM_ambientNoise\airgun_detections\AirgunDetections_network_gpu"
SITES = {"GC": ("GC_17", "GC_18"), "MC": ("MC_19", "MC_20"), "DC": ("DC_17", "DC_18")}
SITE_NAME = {"GC": "Green Canyon (GC)", "MC": "Mississippi Canyon (MC)", "DC": "De Soto Canyon (DC)"}
DETS = ("original", "prototype", "cnn")
COL = {"original": "#D55E00", "prototype": "#0072B2", "cnn": "#009E73"}
LABEL = {"original": "Original", "prototype": "Shape-based cross-correlation", "cnn": "Network (CNN)"}
MARK = {"original": "o", "prototype": "s", "cnn": "^"}
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "0.6", "xtick.color": "0.35", "ytick.color": "0.35",
                     "axes.titlecolor": "0.15", "axes.titlesize": 11, "axes.titlelocation": "left"})


def tidy(ax):
    ax.grid(color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pr(p):
    return p["found"] / max(p["shots"], 1), ((p["detections"] - p["spare"]) / p["detections"]
                                             if p["detections"] else np.nan)


# ---- counts per deployment and setting step: MATLAB sweep + network sweep ----
models, thrs = ensemble.load("network_airgun")
USED = ensemble.threshold_of(thrs)
counts = defaultdict(list)        # (detector, deployment) -> [dict(step, current, shots, detections, found, spare)]
for r in csv.DictReader(open(os.path.join(HERE, "pr_curve_sites.csv"))):
    dep = r["set"].replace("deploy_", "")
    if r["detector"] in DETS and any(dep in v for v in SITES.values()):
        counts[(r["detector"], dep)].append({k: int(float(r[k])) for k in
                                             ("step", "current", "shots", "detections", "found", "spare")})
sets, probs, net_dets = {}, {}, {}
for deps in SITES.values():
    for dep in deps:
        ws = data.cache_set("deploy_" + dep)
        sets[dep] = ws
        probs[dep] = {w["window"]: ensemble.probability(models, w["S"]) for w in ws}
        for k, thr in enumerate(THRS):
            dets = {w["window"]: score.pick(probs[dep][w["window"]], thr) for w in ws}
            _, rows = score.score_set(ws, dets)
            counts[("cnn", dep)].append(dict(step=k, current=int(abs(thr - USED) < 1e-6),
                                             shots=sum(r["shots"] for r in rows),
                                             detections=sum(r["detections"] for r in rows),
                                             found=sum(r["found"] for r in rows), spare=sum(r["spare"] for r in rows)))
        net_dets[dep] = {w["window"]: score.pick(probs[dep][w["window"]], USED) for w in ws}


def pooled(det, deps):
    by = defaultdict(lambda: dict(shots=0, detections=0, found=0, spare=0, current=0))
    for dep in deps:
        for p in counts[(det, dep)]:
            q = by[p["step"]]
            for k in ("shots", "detections", "found", "spare"):
                q[k] += p[k]
            q["current"] = max(q["current"], p["current"])
    return [by[k] for k in sorted(by)]


# ---- figure 1: precision-recall per site ----
fig, axes2 = plt.subplots(2, 3, figsize=(14, 9.6), sharey="row")
for axes, zoom in zip(axes2, (False, True)):
    for ax, (site, deps) in zip(axes, SITES.items()):
        for det in DETS:
            pts = pooled(det, deps)
            xy = np.array([pr(p) for p in pts])
            ax.plot(xy[:, 0], xy[:, 1], "-", color=COL[det], lw=2, marker=MARK[det], ms=4, label=LABEL[det])
            k = [i for i, p in enumerate(pts) if p["current"]][0]
            ax.plot(xy[k, 0], xy[k, 1], MARK[det], ms=12, mfc="none", mec=COL[det], mew=2)
        n_win = sum(1 for d in deps for w in sets[d])
        ax.set_title("%s\n%s and %s: %d shots, %d windows%s" % (SITE_NAME[site], deps[0].replace("_", " "),
                     deps[1].replace("_", " "), pooled("cnn", deps)[0]["shots"], n_win, " (enlarged)" if zoom else ""))
        ax.set_xlim(*((0.5, 1.005) if zoom else (0, 1.02)))
        ax.set_ylim(*((0.8, 1.005) if zoom else (0, 1.02)))
        ax.set_xlabel("Recall (shots found / shots marked)")
        tidy(ax)
    axes[0].set_ylabel("Precision (detections on a shot / all detections)")
axes2[0][0].plot([], [], "o", ms=11, mfc="none", mec="0.3", mew=2, label="Setting in use")
axes2[0][0].legend(loc="lower left", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_detectors_PR_by_site.png"), dpi=200)
plt.close(fig)

# ---- figure 2: recall and precision per deployment at the settings in use ----
deps = [d for v in SITES.values() for d in v]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
ypos = {d: -i - (i // 2) * 0.6 for i, d in enumerate(deps)}       # a gap between sites
off = {"original": 0.22, "prototype": 0.0, "cnn": -0.22}
print("\nAt the settings in use (recall / precision):")
for det in DETS:
    rp = {d: pr([p for p in counts[(det, d)] if p["current"]][0]) for d in deps}
    for ax, j in zip(axes, (0, 1)):
        ax.plot([rp[d][j] for d in deps], [ypos[d] + off[det] for d in deps], MARK[det], color=COL[det], ms=8,
                ls="none", label=LABEL[det])
        for d in deps:
            ax.text(rp[d][j] + (0.012 if j == 0 else 0.0025), ypos[d] + off[det], "%.2f" % rp[d][j],
                    va="center", fontsize=7.5, color="0.35")
    print("  %-11s " % det + "  ".join("%s %.2f/%.2f" % (d, *rp[d]) for d in deps))
for ax, j in zip(axes, (0, 1)):
    ax.set_xlim(*((0, 1.08) if j == 0 else (0.9, 1.012)))
    ax.set_xlabel(("Recall (shots found / shots marked)", "Precision (detections on a shot / all detections)")[j])
    tidy(ax)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks([ypos[d] for d in deps])
axes[0].set_yticklabels(["%s (%d shots)" % (d.replace("_", " "), [p for p in counts[("cnn", d)]][0]["shots"])
                         for d in deps])
axes[0].legend(loc="lower left", frameon=False, fontsize=9, bbox_to_anchor=(0, 1.0), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_detectors_by_deployment.png"), dpi=200)
plt.close(fig)

# ---- figure 3: one typical window per site ----
# The window shown is chosen by rule, not by eye: of each site's newer deployment,
# the airgun window whose network recall is closest to that deployment's overall
# network recall; the 60 s excerpt is the one with the most marks.
other = {}
for det in ("original", "prototype"):
    for dep in deps:
        for r in csv.DictReader(open(os.path.join(HERE, "deploy_" + dep, "detections_proposals_%s.csv" % det))):
            other.setdefault((det, dep, r["window"]), []).append(float(r["start_s"]))
fig, axes = plt.subplots(3, 1, figsize=(13, 9.6))
for ax, (site, (_, dep)) in zip(axes, SITES.items()):
    ws = sets[dep]
    _, rows = score.score_set(ws, net_dets[dep])
    overall = sum(r["found"] for r in rows) / sum(r["shots"] for r in rows)
    cand = [r for r in rows if r["shots"] >= 20]
    pick = min(cand, key=lambda r: abs(r["found"] / r["shots"] - overall))
    w = [x for x in ws if x["window"] == pick["window"]][0]
    t = np.arange(w["S"].shape[1]) * data.HOP_S
    starts = [s0 for s0, _ in net_dets[dep][w["window"]]]

    def excerpt_recall(s):
        m = [x for x in w["marks"] if s <= x < s + 60]
        return sum(any(abs(x - d) <= score.TOL for d in starts) for x in m) / max(len(m), 1), len(m)
    # the excerpt whose network recall is closest to the deployment's, the one with most shots if tied
    a = min(range(0, int(t[-1]) - 59, 10), key=lambda s: (round(abs(excerpt_recall(s)[0] - overall), 2),
                                                         -excerpt_recall(s)[1]))
    k = (t >= a) & (t < a + 60)
    ax.imshow(w["S"][:, k] * 20, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=30,
              extent=[0, 60, data.FMIN, data.FMAX])
    rows_y = {"marks": 330, "original": 305, "prototype": 285, "cnn": 265}
    ax.plot([m - a for m in w["marks"] if a <= m < a + 60], [rows_y["marks"]] * sum(a <= m < a + 60 for m in w["marks"]),
            "v", color="0.15", ms=7, clip_on=False)
    for det in DETS:
        st = ([s0 for s0, _ in net_dets[dep][w["window"]]] if det == "cnn" else other.get((det, dep, w["window"]), []))
        st = [s - a for s in st if a <= s < a + 60]
        ax.plot(st, [rows_y[det]] * len(st), "|", color=COL[det], ms=11, mew=2.5, clip_on=False)
    for name, y in rows_y.items():
        ax.text(60.4, y, "Marked shots" if name == "marks" else LABEL[name], va="center", fontsize=8.5,
                color="0.15" if name == "marks" else COL[name])
    ax.set_ylim(data.FMIN, 345)
    ax.set_yticks([50, 100, 150, 200, 250])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Frequency (Hz)")
    r_ex, n_ex = excerpt_recall(a)
    ax.set_title("%s, %s window %s, %d-%d s: network found %d of %d shots here, %d of %d in the whole window"
                 % (SITE_NAME[site], dep.replace("_", " "), w["window"], a, a + 60, round(r_ex * n_ex), n_ex,
                    pick["found"], pick["shots"]), fontsize=10)
axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
fig.subplots_adjust(right=0.83)
fig.savefig(os.path.join(FIG, "fig_example_windows.png"), dpi=200)
plt.close(fig)

# ---- figure 4: share of minutes with airguns per day, whole deployments ----
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, sharey=True)
for ax, site in zip(axes, SITES):
    shown = []
    for f in sorted(glob.glob(os.path.join(RUNS, "GOM_%s_*_AirgunDetections" % site, "*_minutes_per_day.csv"))):
        rows = list(csv.DictReader(open(f)))
        day = np.array([datetime.datetime.strptime(r["date"], "%Y-%m-%d") for r in rows])
        rec = np.array([int(r["minutes_recorded"]) for r in rows])
        pos = np.array([int(r["minutes_with_airguns"]) for r in rows])
        ok = rec >= 60                                         # days with at least an hour recorded
        share = np.where(ok, 100.0 * pos / np.maximum(rec, 1), np.nan)
        ax.fill_between(day, 0, share, color="#2a78d6", alpha=0.25, lw=0, step="mid")
        ax.plot(day, share, color="#2a78d6", lw=0.9, drawstyle="steps-mid")
        name = os.path.basename(os.path.dirname(f)).replace("_AirgunDetections", "").replace("GOM_", "")
        ax.text(day[0], 103, name.replace("_", " "), fontsize=8, color="0.35")
        shown.append(name)
    ax.set_ylim(0, 112)
    ax.set_title("%s: %s" % (SITE_NAME[site], ", ".join(s.replace("_", " ") for s in shown) or "no deployments yet"),
                 fontsize=10)
    ax.set_ylabel("Minutes with\nairguns (%)")
    tidy(ax)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_airguns_per_day.png"), dpi=200)
plt.close(fig)
print("figures written to", FIG)
