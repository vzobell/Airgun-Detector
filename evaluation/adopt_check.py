"""Copy a finished label check back into its set.

    python adopt_check.py GC_18

Keeps the old labels as truth_before_check.csv / review_before_check.csv (and
humps / picks), then replaces the rows of every window in check_<name> with
the checked ones. Prints what changed.
"""
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
src, dst = os.path.join(HERE, "check_" + name), os.path.join(HERE, "deploy_" + name)
keep = {r["window"] for r in csv.DictReader(open(os.path.join(src, "windows.csv")))}


def rows(path):
    if not os.path.exists(path):
        return None, []
    lines = open(path).read().splitlines()
    return lines[0], lines[1:]


old_status = {r["window"]: r["status"] for r in csv.DictReader(open(os.path.join(dst, "review.csv")))}
old_n = {}
for r in csv.DictReader(open(os.path.join(dst, "truth.csv"))):
    old_n[r["window"]] = old_n.get(r["window"], 0) + 1
for fn in ("review.csv", "truth.csv", "humps.csv", "picks.csv"):
    head_c, new = rows(os.path.join(src, fn))
    head_d, cur = rows(os.path.join(dst, fn))
    if head_c is None and head_d is None:
        continue
    backup = os.path.join(dst, fn.replace(".csv", "_before_check.csv"))
    if os.path.exists(os.path.join(dst, fn)) and not os.path.exists(backup):
        shutil.copy(os.path.join(dst, fn), backup)
    out = [l for l in cur if l.split(",")[0] not in keep] + new
    with open(os.path.join(dst, fn), "w", newline="") as f:
        f.write("\n".join([head_d or head_c] + out) + "\n")

new_status = {r["window"]: r["status"] for r in csv.DictReader(open(os.path.join(dst, "review.csv")))}
new_n = {}
for r in csv.DictReader(open(os.path.join(dst, "truth.csv"))):
    new_n[r["window"]] = new_n.get(r["window"], 0) + 1
for w in sorted(keep):
    if old_status.get(w) != new_status.get(w) or old_n.get(w, 0) != new_n.get(w, 0):
        print("  %s: %s, %d marks -> %s, %d marks" % (w, old_status.get(w), old_n.get(w, 0), new_status.get(w), new_n.get(w, 0)))
print("deploy_%s: %d marks before, %d after" % (name, sum(old_n.values()), sum(new_n.values())))
