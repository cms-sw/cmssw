#!/usr/bin/env python3
"""Working point of a retrained track DNN, at the recall of the bank it replaces, bin by bin.

usage: track_dnn_working_point.py --bank prompt|displaced --old-header H --pt MODEL.pt nano1 ...

One decision per track means track survival is track recall, so a recall rule is legitimate
here (the per-triplet gate compounds and needs a survival rule instead). The rule: on the test
split of the given loose dumps, measure how much of the matched (efficiency) population the
bank being replaced keeps at its own baked threshold, in every pT, |eta| and |dxy| bin; then
take the largest threshold at which the new model keeps at least that fraction in every
populated bin, which is the smallest of the per-bin thresholds. The global equal-recall
threshold is printed next to it for comparison; it is not the chosen value, because a single
overall quantile lets the populous bins pay for the sparse tails.

The last line is `CHOSEN per-bin threshold <x> ...`; retrain_track_dnn.sh reads it and bakes x.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_disp_nano as T                      # noqa: E402
import compare_track_dnn_banks as B              # noqa: E402

PT_EDGES = [0, 1, 2, 5, 10, 1e9]
ETA_EDGES = [0, 0.8, 1.6, 2.4, 3.0, 4.5]
DXY_EDGES = [0, 0.1, 0.5, 1.0, 5.0, 1e9]
MIN_N = 500          # a bin with fewer matched tracks than this does not constrain the threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nano", nargs="+", help="the loose track-level dumps of this run")
    ap.add_argument("--bank", choices=["prompt", "displaced"], required=True)
    ap.add_argument("--old-header", required=True, help="the header the bake is about to overwrite")
    ap.add_argument("--pt", required=True, help="the newly trained model")
    a = ap.parse_args()
    prefix = "TrkPrompt" if a.bank == "prompt" else "TrkDisp"
    df = T.load_nano(a.nano, prefix)
    _, _, te = T.split_by_event(df)
    X = te[T.FEATS].values.astype(np.float64)
    eff = te.label_eff.values.astype(int) == 1
    h = B.parse_header(a.old_header)
    s_old = B.score_header(h, X)
    s_new = B.score_pt(a.pt, X)
    thr_old = h["thr"]
    print("old bank %s at %.6f; test rows %d, matched %d"
          % (os.path.basename(a.old_header), thr_old, len(te), eff.sum()))
    axes = [("pT", te.tpPt.values.astype(float), PT_EDGES),
            ("|eta|", np.abs(te.tpEta.values.astype(float)), ETA_EDGES),
            ("|dxy|", np.abs(te.dXY.values.astype(float)), DXY_EDGES)]
    chosen, binding = np.inf, None
    print("%-6s %-14s %8s %10s %12s" % ("axis", "bin", "n", "oldKeep", "newThr@keep"))
    for name, col, edges in axes:
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = eff & (col >= lo) & (col < hi)
            n = int(m.sum())
            if n < MIN_N:
                print("%-6s [%g,%g) %8d  --" % (name, lo, hi, n))
                continue
            keep = float((s_old[m] >= thr_old).mean())
            t = float(np.quantile(s_new[m], 1.0 - keep)) if keep < 1.0 else float(s_new[m].min())
            flag = ""
            if t < chosen:
                chosen, binding, flag = t, "%s [%g,%g)" % (name, lo, hi), "  <- binding"
            print("%-6s [%g,%g) %8d %10.5f %12.6f%s" % (name, lo, hi, n, keep, t, flag))
    keep_all = float((s_old[eff] >= thr_old).mean())
    t_global = float(np.quantile(s_new[eff], 1.0 - keep_all))
    fake = ~(te.label.values.astype(int) == 1)
    for lbl, t in (("old bank", thr_old), ("new per-bin", chosen), ("new global", t_global)):
        s = s_old if lbl == "old bank" else s_new
        print("%-12s thr=%.6f  matchedKeep=%.5f  fakeKeep=%.5f"
              % (lbl, t, float((s[eff] >= t).mean()), float((s[fake] >= t).mean())))
    print("CHOSEN per-bin threshold %.6e (binding %s); global equal-recall %.6e"
          % (chosen, binding, t_global))


if __name__ == "__main__":
    main()
