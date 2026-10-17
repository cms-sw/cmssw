#!/usr/bin/env python3
"""Measure a deployed compact forest on a feature cache and write its working-point profile: the
class-A (true-track) recall and class-C (fake) rejection the model has at a given threshold in
every pT, |eta| and |dxyBS| bin, in the JSON format train_merged_forest.py --wp-profile reads.

  python3 make_profile.py --cache <cache.npz ...> --model <deployed .bin> --threshold <t> \\
          --arm prompt|disp|merged --out <profile.json> [--split test|all] [--label mtv|legacy]
          [--feats 31|35 --abi-order ... --extra-cols ...]   (the model's feature vector, as in
                                                              train_merged_forest.py)

With that profile, train_merged_forest.py --wp-rule profile derives the working point at which
the new model matches or exceeds the deployed one in every bin of the same rows. The bin edges
are the trainer's defaults, so the labels line up bin for bin. By default the model is measured
on the trainer's test split (evt % 10 < 3), the rows the trainer compares against.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read_compact_tree as rd  # noqa: E402
import train_merged_forest as tmf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", required=True,
                    help="one or more nano_loader.py cache-arm / cache-merged .npz")
    ap.add_argument("--model", required=True, help="the compact .bin to measure")
    ap.add_argument("--threshold", type=float, required=True,
                    help="the score threshold the model is deployed at")
    ap.add_argument("--arm", default="prompt", help="recorded in the profile (prompt|disp|merged)")
    ap.add_argument("--out", required=True, help="output profile JSON")
    ap.add_argument("--name", default="deployed", help="a label recorded in the profile")
    ap.add_argument("--split", default="test", choices=["test", "all"],
                    help="rows the reference recall is measured on: 'test' = the trainer's own "
                         "event split (evt%%10 < 3), the rows it compares against; 'all' = every "
                         "cached row")
    ap.add_argument("--label", choices=("mtv", "legacy"), default="mtv",
                    help="the label the cache was built with (nano_loader.py --label)")
    ap.add_argument("--feats", type=int, choices=(31, 35), default=31,
                    help="the model's feature vector, as for train_merged_forest.py: 31 = the "
                         "per-iteration selectors; 35 = + the provenance block in --abi-order")
    ap.add_argument("--abi-order", default="nAttached,nOTExtra,iteration,ndof",
                    help="order of the 4 provenance columns in the selector's feature vector")
    ap.add_argument("--extra-cols", default="", dest="extra_cols", metavar="c1,c2,...",
                    help="extra cache columns appended after the ABI block (the merged selector's "
                         "seven pixel-cluster columns)")
    a = ap.parse_args()

    d = tmf.load_caches(a.cache, want_label=a.label)
    X, names, _ = tmf.build_abi_matrix(d["X"], d["feats"], a.feats, a.abi_order,
                                       [c for c in a.extra_cols.split(",") if c.strip()])
    y_eff = d["y_eff"]
    y_mtv = d["y_mtv"]
    dxyBS = d["dxyBS"]
    if a.split == "test":
        te = (d["evt"].astype(np.int64) % 10) < 3
        X = X[te]
        y_eff = y_eff[te]
        y_mtv = y_mtv[te]
        dxyBS = dxyBS[te]
        print("test split (evt%%10 < 3): %d of %d rows" % (te.sum(), len(te)))
    A = y_eff >= 0.5
    C = y_mtv < 0.5
    dxy = np.abs(dxyBS)
    pt = X[:, names.index("pt")].astype(np.float64)
    eta = np.abs(X[:, names.index("eta")].astype(np.float64))
    print("cache rows=%d feats=%d  A=%d C=%d  model=%s thr=%g"
          % (len(X), X.shape[1], A.sum(), C.sum(), os.path.basename(a.model), a.threshold))

    m = rd.load_compact(a.model)
    nf = int(m["feat"].max()) + 1
    if nf > X.shape[1]:
        raise SystemExit("the model reads %d features but the feature vector built here has %d: "
                         "pass the --feats / --abi-order / --extra-cols the model was trained "
                         "with" % (nf, X.shape[1]))
    Xs = np.ascontiguousarray(X.astype(np.float32))
    print("scoring on %d columns: %s ... %s" % (X.shape[1], names[:3], names[-1]))
    s = np.asarray(rd.score_compact(m, Xs), dtype=np.float64)
    print("scored: nTrees=%d nNodes=%d  mean score=%.4f  keep@thr=%.4f"
          % (m["nTrees"], m["nNodes"], s.mean(), float((s >= a.threshold).mean())))

    edges = {"pt": tmf.parse_edges(tmf.PT_BINS_DEFAULT, "--pt-bins"),
             "eta": tmf.parse_edges(tmf.ETA_BINS_DEFAULT, "--eta-bins"),
             "dxy": tmf.parse_edges(tmf.DXY_BINS_DEFAULT, "--dxy-bins")}
    bins = {"pt": tmf.make_bins(edges["pt"], underflow=True),
            "eta": tmf.make_bins(edges["eta"], underflow=False),
            "dxy": tmf.make_bins(edges["dxy"], underflow=False)}
    vals = {"pt": pt, "eta": eta, "dxy": dxy}
    axes = {}
    for ax in ("pt", "eta", "dxy"):
        tab = tmf.bin_table(bins[ax], vals[ax], A, C, s, a.threshold)
        axes[ax] = {"edges": edges[ax], "bins": tab}
        for lab, v in tab.items():
            print("  %-4s %-10s nA=%-9d nC=%-9d recall=%s rej=%s"
                  % (ax, lab, v["nA"], v["nC"],
                     ("%.6f" % v["recall"]) if np.isfinite(v["recall"]) else "nan",
                     ("%.6f" % v["rej"]) if np.isfinite(v["rej"]) else "nan"))
    glob_recall = float((s[A] >= a.threshold).mean()) if A.sum() else float("nan")
    glob_rej = float(1.0 - (s[C] >= a.threshold).mean()) if C.sum() else float("nan")
    print("GLOBAL recall=%.6f rejection=%.6f" % (glob_recall, glob_rej))
    out = {"model": os.path.abspath(a.model), "name": a.name, "threshold": a.threshold,
           "test_rows": int(len(X)), "arm": a.arm, "split": a.split, "feats": names,
           "reader": os.path.abspath(rd.__file__),
           "global": {"recall": glob_recall, "rej": glob_rej, "nA": int(A.sum()),
                      "nC": int(C.sum())},
           "caches": [os.path.abspath(p) for p in a.cache], "axes": axes}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
