"""Run and score the network, and pack its copies into one file.

    python ensemble.py network_airgun set1 set2 set3
    python ensemble.py network_airgun set3 --threshold 0.5
    python ensemble.py --pack train123mix_seed network_airgun

The network is trained several times from different random starts
(train.py --seed 1, 2, ...), because single copies differ noticeably; the
probabilities of the copies are averaged before peaks are picked.

--pack writes all copies whose names start with the given text into one file,
models/<name>.pt, together with the threshold to use (the mean of the
thresholds each copy chose on its own held-back training windows, rounded to
0.05). That one file is then the detector.

Scoring takes either a packed file's name or the start of the copies' names,
looks in models/ and then archived/, and writes
models/<name>_detections_<set>.csv.
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch

import data
import score
from model import ShotNet

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [os.path.join(HERE, "models"), os.path.join(HERE, "archived")]


def _build(n_freq, state, arch="base"):
    m = ShotNet(n_freq, arch)
    m.load_state_dict(state)
    m.eval()
    return m


def load(name):
    """Models and thresholds for a packed file (models/<name>.pt) or for all copies named <name>*.pt."""
    for folder in FOLDERS:
        p = os.path.join(folder, name + ".pt")
        if os.path.exists(p):
            ck = torch.load(p)
            if "states" in ck:
                return [_build(ck["n_freq"], st, ck.get("arch", "base")) for st in ck["states"]], [ck["threshold"]]
    for folder in FOLDERS:
        models, thr = [], []
        for p in sorted(glob.glob(os.path.join(folder, name + "*.pt"))):
            ck = torch.load(p)
            if "state" in ck:
                models.append(_build(ck["n_freq"], ck["state"], ck.get("arch", "base")))
                thr.append(ck["threshold"])
        if models:
            return models, thr
    return [], []


def threshold_of(thrs):
    return round(float(np.mean(thrs)) / 0.05) * 0.05


def probability(models, S):
    """Average shot-start probability of the copies for one spectrogram, one value per time step."""
    with torch.no_grad():
        x = torch.from_numpy(S).unsqueeze(0)
        return np.mean([torch.sigmoid(m(x))[0].numpy() for m in models], axis=0)


def pack(prefix, name):
    states, thrs, n_freq, files = [], [], None, []
    for folder in FOLDERS:
        for p in sorted(glob.glob(os.path.join(folder, prefix + "*.pt"))):
            ck = torch.load(p)
            if "state" in ck:
                states.append(ck["state"])
                thrs.append(ck["threshold"])
                n_freq, arch = ck["n_freq"], ck.get("arch", "base")
                files.append(os.path.basename(p))
        if states:
            break
    assert states, "no models found whose names start with " + prefix
    out = os.path.join(HERE, "models", name + ".pt")
    torch.save(dict(states=states, n_freq=n_freq, arch=arch, threshold=threshold_of(thrs), copies=files, copy_thresholds=thrs), out)
    print("packed %d copies (%s) into %s, threshold %.2f" % (len(states), ", ".join(files), out, threshold_of(thrs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--pack", action="store_true", help="pack copies named <name>* into models/<sets[0]>.pt")
    a = ap.parse_args()
    if a.pack:
        pack(a.name, a.sets[0])
        return

    models, thrs = load(a.name)
    thr = a.threshold if a.threshold is not None else threshold_of(thrs)
    print("%d copies averaged, threshold used %.2f" % (len(models), thr))
    for s in a.sets:
        ws = data.cache_set(s)
        dets = {w["window"]: score.pick(probability(models, w["S"]), thr) for w in ws}
        tot, rows = score.score_set(ws, dets)
        print("RESULT %-5s %s" % (s, score.describe(tot)))
        print("DETAIL " + " | ".join("%s %d/%d s%d" % (r["window"], r["found"], r["shots"], r["spare"]) for r in rows))
        with open(os.path.join(HERE, "models", "%s_detections_%s.csv" % (a.name, s)), "w", newline="") as f:
            c = csv.writer(f)
            c.writerow(["window", "start_s", "end_s"])
            for w in ws:
                c.writerows([[w["window"], "%.2f" % d0, "%.2f" % d1] for d0, d1 in dets[w["window"]]])


if __name__ == "__main__":
    main()
