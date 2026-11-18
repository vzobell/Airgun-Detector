# Archived models

Earlier versions of the network, kept for the record. They are not used by the detector.

| Files | Trained on |
|---|---|
| `smoke.pt` | First trial, trained on test set 1 only |
| `train1_seed1..4.pt` | Labelling round 1 (134 windows, 2,539 shots) |
| `train12_seed1..4.pt` | Rounds 1 and 2 (234 windows, 5,177 shots) |
| `train12mix_seed1..4.pt` | Rounds 1 and 2, with background noise mixed in |

The `*_detections_*.csv` files are each model's detections on the test sets at the time it was trained.

| `train123mix_seed1..4.pt` | Rounds 1, 2 and 3 (318 windows, 6,551 shots), with background noise mixed in. These four copies are what `../models/network_airgun.pt` is made of. |
| `hist80_history.csv`, `hist160_history.csv` | Loss per pass for one copy trained for 80 and for 160 passes (drawn by `plot_history.py`) |

The current network is `../models/network_airgun.pt`: the four `train123mix` copies packed into one file by `python ensemble.py --pack train123mix_seed network_airgun`.

To score an archived version, give the start of its file names, for example `python ensemble.py train12_seed set3`.
