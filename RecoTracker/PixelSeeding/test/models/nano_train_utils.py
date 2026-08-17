"""Shared helpers for the nano-ntuple track-classifier trainers (train_prompt_hp_nano.py,
train_disp_nano.py). Self-contained, so the nano workflow has no dependency on track-level
dump infrastructure.
"""
import numpy as np
import torch
import torch.nn as nn


class MLPn(nn.Module):
    """Fully-connected classifier: nf inputs -> hidden layers (ReLU) -> 1 logit."""

    def __init__(self, nf, hidden):
        super().__init__()
        layers, prev = [], nf
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ExportModule(nn.Module):
    """forward(track_features: [N,F] half or float) -> scores [N,1] in [0,1].
    Standardization + MLP + sigmoid; dtype-agnostic (the PixelTrackTorchHighPuritySelector
    producer converts module AND inputs to kHalf, so ops run in the input dtype). Output
    kept [N,1] -- the producer's TensorCollection copy requires it; offline callers squeeze.
    """

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, track_features):
        x = (track_features - self.mean) / self.std
        return torch.sigmoid(self.net(x))


def thr_at_recall(y, s, recall):
    """Score threshold that retains `recall` fraction of the real (y==1) population."""
    real = np.sort(s[y == 1])
    return real[int((1.0 - recall) * len(real))]


def scores(model, X, dev, bs=131072):
    """Batched sigmoid scoring; returns float32 ndarray (CMSSW torch lacks NumPy interop)."""
    model.eval()
    out = []
    with torch.no_grad():
        Xt = torch.tensor(X)
        for i in range(0, len(Xt), bs):
            out.append(np.asarray(torch.sigmoid(model(Xt[i:i + bs].to(dev))).cpu().tolist(),
                                  dtype=np.float32))
    return np.concatenate(out)
