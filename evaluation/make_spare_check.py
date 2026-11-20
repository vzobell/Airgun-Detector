"""Build a label check for a reviewed set: every spare detection of all three detectors.

    python make_spare_check.py deploy_GC_18 [deploy_MC_20 ...]

A spare detection is one with no mark within 2 s (the scoring tolerance),
made by the original, the shape-based or the network detector at the settings
in use. They are pooled and shown as hints (blue triangles) without saying
which detector made them, so the check does not favour any one detector.

Writes check_<set>/ with the windows that have hints (windows marked
"airguns unclear" or dropped are left out, they are not scored), their current
review.csv / truth.csv / humps.csv / picks.csv and hints.csv. Review it with
review_marks('check_<set>'); the result is then copied back into the set.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cnn"))
import data          # noqa: E402
import ensemble      # noqa: E402
import score         # noqa: E402

TOL = 2.0


def spare_starts(marks, det, hump_join):
    """Start times of the detections left unused by score.score_window (same matching)."""
    marks = np.sort(np.asarray(marks, dtype=float))
    events = [[x] for x in marks]
    if hump_join and len(marks):
        events = [[marks[0]]]
        for x in marks[1:]:
            if x - events[-1][-1] > hump_join:
                events.append([x])
            else:
                events[-1].append(x)
    used_ = np.zeros(len(det), dtype=bool)
    for ev in events:
        if len(ev) == 1:
            k = np.where((ev[0] >= det[:, 0] - score.TOL) & (ev[0] <= det[:, 1] + score.TOL))[0]
        else:
            k = np.where((det[:, 1] >= ev[0] - score.TOL) & (det[:, 0] <= ev[-1] + score.TOL))[0]
        free = k[~used_[k]]
        if len(free):
            if len(ev) == 1:
                used_[free[0]] = True
            else:
                used_[k] = True
    return list(det[~used_, 0])

models, thrs = ensemble.load("network_airgun")
used = ensemble.threshold_of(thrs)

for name in sys.argv[1:]:
    src = os.path.join(HERE, name)
    status = {r["window"]: r["status"] for r in csv.DictReader(open(os.path.join(src, "review.csv")))}
    marks = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(src, "truth.csv"))):
        marks[r["window"]].append(float(r["time_s"]))
    humps = {}
    if os.path.exists(os.path.join(src, "humps.csv")):
        humps = {r["window"]: float(r["join_s"]) for r in csv.DictReader(open(os.path.join(src, "humps.csv")))}
    det = defaultdict(lambda: defaultdict(list))      # window -> detector -> [(start, end)]
    for d in ("original", "prototype"):
        for r in csv.DictReader(open(os.path.join(src, "detections_proposals_%s.csv" % d))):
            det[r["window"]][d].append((float(r["start_s"]), float(r["end_s"])))
    for w in data.cache_set(name):
        det[w["window"]]["cnn"] = list(score.pick(ensemble.probability(models, w["S"]), used))
    hints, count = defaultdict(list), defaultdict(int)
    for win, by in det.items():
        if status.get(win) not in ("airguns", "no airguns", "unknown"):
            continue
        m = marks[win] if status[win] == "airguns" else []
        spare = []
        for d, lst in by.items():
            for t in spare_starts(m, np.array(lst, dtype=float).reshape(-1, 2), humps.get(win)):
                count[d] += 1
                spare.append(t)
        for t in sorted(spare):
            if all(abs(t - h) > TOL for h in hints[win]):
                hints[win].append(t)
    out = os.path.join(HERE, "check_" + name.replace("deploy_", ""))
    os.makedirs(out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(src, "windows.csv"))) if hints.get(r["window"])]
    keep = {r["window"] for r in rows}
    with open(os.path.join(out, "windows.csv"), "w", newline="") as f:
        c = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        c.writeheader()
        c.writerows(rows)
    for fn in ("review.csv", "truth.csv", "humps.csv", "picks.csv"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            lines = open(p).read().splitlines()
            with open(os.path.join(out, fn), "w", newline="") as f:
                f.write("\n".join([lines[0]] + [l for l in lines[1:] if l.split(",")[0] in keep]) + "\n")
    with open(os.path.join(out, "hints.csv"), "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["window", "time_s"])
        for win in sorted(keep):
            c.writerows([[win, "%.2f" % t] for t in hints[win]])
    print("%s: %d windows, %d places to check (spare: original %d, shape-based %d, network %d) -> %s"
          % (name, len(keep), sum(len(hints[w]) for w in keep), count["original"], count["prototype"], count["cnn"], out))
