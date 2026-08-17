#!/usr/bin/env python3
"""Score a BAKED in-kernel track-DNN header (CATrackDNNWeights_<bank>.h) on a trackNano
test split, optionally comparing it against another bank and/or its source .pt.

WHY: the GPU runs the BAKED HEADER, not the .pt; the bake (weight transposes, layout,
standardization arrays, %.8e text round-trip) is only exercised at build time, so a
bake bug is invisible to the trainer's metrics. This script reproduces CATrackDNN.h's
forward pass in numpy and scores the header itself, so a stale bank (baked from a model
trained on an older fit) shows up as well.

THRESHOLD RULE: one decision per track (k=1) => per-TP survival == track recall.
The per-triplet gate is different (~10 real triplets/TP => 0.99 per-triplet recall is
only ~0.90 per-TP survival; see retrain_triplet_gate.sh). The BAKED kDefaultThreshold
is a fallback: the producer cfi's trackDNNThreshold overrides it, an in-situ pick.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_disp_nano import FEATS, load_nano, split_by_event  # noqa: E402

HDR_ARRAYS = ("kMean", "kScale", "kW1", "kB1", "kW2", "kB2", "kW3", "kB3")


def parse_header(path):
    """Read a baked CATrackDNNWeights_<bank>.h. Tolerates clang-format reflowing
    (whitespace-insensitive comma-separated float parsing)."""
    txt = open(path).read()
    out = {}
    for name in HDR_ARRAYS:
        m = re.search(r"%s\[\d+\]\s*=\s*\{(.*?)\};" % name, txt, re.S)
        if m is None:
            raise RuntimeError("%s: array %s not found" % (path, name))
        out[name] = np.array([float(v.strip().rstrip("f")) for v in m.group(1).split(",")])
    out["shape"] = tuple(int(re.search(k + r"\s*=\s*(\d+)", txt).group(1))
                         for k in ("kNFeat", "kNH1", "kNH2"))
    out["thr"] = float(re.search(r"kDefaultThreshold\s*=\s*([\d.eE+-]+)f", txt).group(1))
    return out

def score_header(h, X):
    """Bit-for-bit mirror of ALPAKA_ACCELERATOR_NAMESPACE::caTrackDNN_<bank>::score()."""
    nf, n1, n2 = h["shape"]
    xn = (X - h["kMean"]) / h["kScale"]
    a = np.maximum(xn @ h["kW1"].reshape(nf, n1) + h["kB1"], 0.0)
    b = np.maximum(a @ h["kW2"].reshape(n1, n2) + h["kB2"], 0.0)
    o = b @ h["kW3"].reshape(n2, 1)[:, 0] + h["kB3"][0]
    return 1.0 / (1.0 + np.exp(-o))


def score_pt(path, X):
    blob = torch.load(path, map_location="cpu")
    sd = blob["state"]
    xn = (X - np.asarray(blob["mean"])) / np.asarray(blob["std"])
    ks = [k for k in sd if k.endswith(".weight")]
    a = xn
    for i, k in enumerate(ks):
        W = np.asarray(sd[k].tolist())
        b = np.asarray(sd[k.replace(".weight", ".bias")].tolist())
        a = a @ W.T + b
        if i < len(ks) - 1:
            a = np.maximum(a, 0.0)
    return 1.0 / (1.0 + np.exp(-a[:, 0]))


def thr_at_recall(y, s, rc):
    return float(np.quantile(s[y == 1], 1.0 - rc))


def rej_at_recall(y, s, rc):
    t = thr_at_recall(y, s, rc)
    return t, float(1.0 - (s[y == 0] >= t).mean())


def per_bin_recall(y, s, col, edges, thr, label):
    """Per-bin real retention at a fixed threshold. For k=1 this IS per-TP survival."""
    print("  per-%s real retention at thr=%.6f:" % (label, thr))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y == 1) & (col >= lo) & (col < hi)
        n = int(m.sum())
        if n < 200:
            print("    [%6.2f,%6.2f)  n=%-8d  --" % (lo, hi, n))
            continue
        print("    [%6.2f,%6.2f)  n=%-8d  keep=%.5f" % (lo, hi, n, float((s[m] >= thr).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nano", nargs="+")
    ap.add_argument("--header", default=None, help="baked CATrackDNNWeights_<bank>.h to score")
    ap.add_argument("--pt", default=None, help="model .pt to score / compare against")
    ap.add_argument("--bank", choices=["prompt", "displaced"], default="prompt")
    ap.add_argument("--recalls", default="0.9999,0.999,0.998,0.995,0.99,0.98,0.95")
    # Fixed ladder plus the two thresholds the chain runs, so the deployed working point is in the table.
    ap.add_argument("--thresholds", default="0.001,0.005,0.01,0.02,0.05,0.054264,0.1,0.369401")
    a = ap.parse_args()
    if not a.header and not a.pt:
        ap.error("give --header and/or --pt")
    prefix = "TrkPrompt" if a.bank == "prompt" else "TrkDisp"

    df = load_nano(a.nano, prefix)
    _, _, te = split_by_event(df)
    X = te[FEATS].values.astype(np.float64)
    y = te.label.values.astype(int)
    print("test rows=%d real=%.4f (bank=%s)" % (len(y), y.mean(), a.bank))

    scores = {}
    if a.header:
        h = parse_header(a.header)
        print("header %s: shape=%s bakedThr=%.6f" % (a.header, h["shape"], h["thr"]))
        scores["header"] = score_header(h, X)
    if a.pt:
        scores["pt"] = score_pt(a.pt, X)

    from sklearn.metrics import roc_auc_score
    for k, s in scores.items():
        print("AUC %-8s = %.5f" % (k, roc_auc_score(y, s)))

    if len(scores) == 2:
        d = np.abs(scores["header"] - scores["pt"]).max()
        # A header baked from this .pt agrees to the float32 text round-trip (~1e-7); more than that
        # means a bake/layout bug or two different models.
        print("max|header - pt| = %.3e  %s" % (d, "(bake round-trip OK)" if d < 1e-5 else
                                               "(DIFFERENT MODELS or a bake bug -- investigate)"))

    print("\n%-10s" % "recall" + "".join("%12s%12s" % ("thr_" + k, "rej_" + k) for k in scores))
    for rc in [float(v) for v in a.recalls.split(",")]:
        row = "%-10.4f" % rc
        for k in scores:
            t, r = rej_at_recall(y, scores[k], rc)
            row += "%12.5f%12.4f" % (t, r)
        print(row)

    print("\nat fixed thresholds (realKeep / fakeKeep):")
    for t in [float(v) for v in a.thresholds.split(",")]:
        row = "  thr=%-10.6f" % t
        for k in scores:
            s = scores[k]
            row += "  %-8s realKeep=%.5f fakeKeep=%.5f" % (
                k, float((s[y == 1] >= t).mean()), float((s[y == 0] >= t).mean()))
        print(row)

    # k=1 means recall equals per-TP survival; the per-bin view catches a population-biased working point.
    if scores:
        s = scores.get("pt", scores.get("header"))
        t = thr_at_recall(y, s, 0.99)
        per_bin_recall(y, s, np.abs(te.tpEta.values), [0, 0.8, 1.6, 2.4, 3.0, 3.5, 4.5], t, "|tpEta|")
        per_bin_recall(y, s, te.nStubs.values, [0, 1, 2, 3, 4, 9], t, "nStubs")


if __name__ == "__main__":
    main()
