"""A small network that reads a spectrogram and gives, for every 0.1 s, the
probability that an airgun shot starts there.

Two parts:
  1. 2-D convolutions over (frequency, time) that learn what a shot looks like
     across frequency. Frequency is pooled away; time resolution is kept.
  2. 1-D convolutions over time with growing gaps (dilations), so each output
     can use about 25 s of context on either side, e.g. the rhythm of a survey.
"""
import torch
import torch.nn as nn


class Block2d(nn.Module):
    def __init__(self, cin, cout, pool_f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d((pool_f, 1)))

    def forward(self, x):
        return self.net(x)


class Block1d(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.net(x)


class GRUContext(nn.Module):
    """Two-way recurrent layer along time (the "CRNN" alternative to the dilated convolutions)."""
    def __init__(self, ch):
        super().__init__()
        self.gru = nn.GRU(ch, ch // 2, batch_first=True, bidirectional=True)

    def forward(self, x):                              # (B, ch, T)
        return x + self.gru(x.transpose(1, 2))[0].transpose(1, 2)


# Variants compared for the manuscript. "base" is the network in use.
ARCHS = {
    "base":  dict(filters=(16, 32, 48, 64), width=64, dilations=(1, 2, 4, 8, 16, 32, 64)),
    "short": dict(filters=(16, 32, 48, 64), width=64, dilations=(1, 2)),           # about +-1 s of context
    "small": dict(filters=(8, 16, 24, 32), width=32, dilations=(1, 2, 4, 8, 16, 32, 64)),
    "large": dict(filters=(32, 64, 96, 128), width=128, dilations=(1, 2, 4, 8, 16, 32, 64)),
    "crnn":  dict(filters=(16, 32, 48, 64), width=64, dilations=None),             # recurrent layer instead
}


class ShotNet(nn.Module):
    def __init__(self, n_freq, arch="base"):
        super().__init__()
        c = ARCHS[arch]
        f1, f2, f3, f4 = c["filters"]
        width = c["width"]
        self.front = nn.Sequential(Block2d(1, f1, 2), Block2d(f1, f2, 2), Block2d(f2, f3, 2), Block2d(f3, f4, 3))
        f_out = n_freq // 2 // 2 // 2 // 3
        self.squeeze = nn.Sequential(nn.Conv1d(f4 * f_out, width, 1), nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        if c["dilations"] is None:
            self.context = GRUContext(width)
        else:
            self.context = nn.Sequential(*[Block1d(width, d) for d in c["dilations"]])
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        """x: (batch, n_freq, n_time) -> logits (batch, n_time)"""
        h = self.front(x.unsqueeze(1))                 # (B, 64, F', T)
        h = h.flatten(1, 2)                            # (B, 64*F', T)
        h = self.context(self.squeeze(h))
        return self.head(h).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for a in ARCHS:
        m = ShotNet(99, a)
        print("%-6s parameters: %7d" % (a, count_parameters(m)), tuple(m(torch.zeros(2, 99, 3001)).shape))
