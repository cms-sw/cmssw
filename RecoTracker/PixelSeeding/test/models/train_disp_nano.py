"""In-kernel track DNN (the loose -> tight promotion): train + bake, for either iteration.

The in-kernel CATrackDNN (Kernel_classifyTracks) is a SIMPLE high-recall loose->tight selector:
it retains real/loose efficiency out to large displacement rather than rejecting fakes, which is
the job of the final high-purity selector after reconstruction. So: 12 cheap features, a 32x16
MLP, plain BCE, a high-recall operating point.

Feature ABI = caTrackFeatures::fill / CATrackDNNWeights.h order (12), from the TrkDispCA nano table
by name. Truth from TrkDispTruth (MTV associator), joined per event (fitChi2 has no phi -> use
TrkDispCA 'phi' vs TrkDispTruth phi, fitPt).

  train:  python3 train_disp_nano.py train <nano.root> [...]
  bake:    python3 train_disp_nano.py bake --bank prompt --out ../../plugins/alpaka/CATrackDNNWeights_prompt.h
           [--bake-recall 0.995]  (default 0.99)

LABEL (--label, default `mtv`):
  target      = matchedAny (matched to ANY TrackingParticle -- MTV's fake definition; the fake axis)
  recall axis = matched (matched to a TP passing the arm's EFFICIENCY selection)
  class A = matched==1              -- the recall axis
  class B = matchedAny==1, matched==0 -- real, not an efficiency TP (low pT, secondary, large |tip|,
           out-of-time PU). A POSITIVE in training, in NEITHER metric axis.
  class C = matchedAny==0           -- the free-fake-rejection axis.
Training on `matched` instead (--label legacy) makes class B a negative, teaching the gate to
throw real tracks away. matchedAny needs a nano with NANO_MTV_LABEL=1 (retrain_common.sh sets it).

Artifacts (intermediate .pt/.json) -> $MODELS_WORKDIR (default: cwd).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import ROOT
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from nano_train_utils import MLPn, scores, thr_at_recall

ROOT.gROOT.SetBatch(True)

ARTDIR = os.environ.get("MODELS_WORKDIR", os.getcwd())  # intermediate model/result artifacts
HIDDEN = (32, 16)
NAME = "disp_stage1_12f"

# Trained ABI order, matching caTrackFeatures::fill.
FEATS = ["fitChi2", "psFrac", "r0", "nPS", "nh", "spanZ",
         "nVectorHits", "nl", "logChi2VectorHit", "kErr", "dcaEst", "nBarrel"]
TRUTH_COLS = ["matched", "duplicate", "tpPt", "tpEta", "dXY"]  # dXY = reco track |dxy|, the displacement weight input
# Present only on a nano dumped with the MTV label; read in addition to TRUTH_COLS under --label mtv.
MTV_COL = "matchedAny"


def _arr(t, name):
    return np.asarray(getattr(t, name), dtype=np.float64)


def load_nano(paths, prefix="TrkDisp", label="mtv"):
    """label='mtv' also reads TrkXTruth_matchedAny and makes it the training target, keeping
    `matched` as the recall axis; label='legacy' reads only `matched`."""
    cols = list(TRUTH_COLS) + ([MTV_COL] if label == "mtv" else [])
    rows = []
    for path in paths:
        f = ROOT.TFile.Open(path)
        t = f.Get("Events")
        if label == "mtv" and not t.GetBranch(f"{prefix}Truth_{MTV_COL}"):
            f.Close()
            raise SystemExit(
                f"{path}: {prefix}Truth has no branch '{MTV_COL}'. The MTV-true label needs a nano "
                f"dumped with the second, all-TrackingParticle association: NANO_MTV_LABEL=1 for "
                f"the prompt/displaced arms (retrain_common.sh sets it). Re-dump, or train with "
                f"--label legacy and know that you are then teaching the gate to reject real "
                f"non-efficiency tracks.")
        n = t.GetEntries()
        ntr = nj = 0
        for iev in range(n):
            t.GetEntry(iev)
            ca = {c: _arr(t, f"{prefix}CA_{c}") for c in FEATS}
            ca_phi = _arr(t, f"{prefix}CA_phi")
            tr_phi = _arr(t, f"{prefix}Truth_phi")
            truth = {c: _arr(t, f"{prefix}Truth_{c}") for c in cols}
            ntr += len(tr_phi)
            if len(tr_phi) == 0 or len(ca_phi) == 0:
                continue
            # fitPt is not in the feature set, so truth joins to CA by the 'phi' column of the same
            # tracks, through a sorted nearest match.
            order = np.argsort(ca_phi)
            ca_phi_sorted = ca_phi[order]
            for i in range(len(tr_phi)):
                pos = np.searchsorted(ca_phi_sorted, tr_phi[i])
                best, bestd = -1, 1e-4
                for pp in (pos - 1, pos):
                    if 0 <= pp < len(order):
                        d = abs(ca_phi_sorted[pp] - tr_phi[i])
                        if d < bestd:
                            bestd, best = d, order[pp]
                if best < 0:
                    continue
                nj += 1
                rows.append([iev] + [ca[c][best] for c in FEATS] + [truth[c][i] for c in cols])
        print(f"  {os.path.basename(path)}: events={n} truth={ntr} joined={nj} ({nj/max(1,ntr):.4f})", flush=True)
        f.Close()
    df = pd.DataFrame(rows, columns=["evt"] + FEATS + cols)
    # fitChi2 arrives already sanitized and log1p-compressed by caTrackFeatures::fill, so it is
    # consumed as is: transforming it again would double-log it and skew training against serving.
    df = df.dropna()
    # `label` is the training target and `label_eff` the recall axis; under --label legacy they are
    # the same column.
    df["label_eff"] = df["matched"].astype(np.float32)
    df["label"] = (df[MTV_COL] if label == "mtv" else df["matched"]).astype(np.float32)
    bad = int((df["label_eff"].values > df["label"].values).sum()) if label == "mtv" else 0
    if bad:
        print(f"  WARNING: {bad} rows have matched=1 with matchedAny=0 -- the two associations "
              f"disagree, which should be impossible; check tpSelectorAnyTP.", flush=True)
    print(f"total: {len(df)} rows, target({label})={df.label.mean():.4f} "
          f"A(matched)={df.label_eff.mean():.4f} "
          f"B(real,not eff TP)={(df.label - df.label_eff).mean():.4f}", flush=True)
    return df


def split_by_event(df):
    m = df.evt.values.astype(int) % 10
    return df[m >= 4], df[m == 3], df[m < 3]


def xy(d, mean=None, std=None):
    X = d[FEATS].values.astype(np.float32)
    if mean is None:
        mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32), d.label.values.astype(np.float32), mean, std


MAX_EPOCHS = 250
PATIENCE = 20
ALPHA = 3.0  # displacement weight strength: w = 1 + ALPHA*min(|dxy|,60)/20 on reals


def _wthr_at_recall(A, s, w, recall):
    """Threshold keeping `recall` of the displacement-WEIGHTED class-A mass (top scores).

    A is a BOOLEAN MASK of the recall axis -- `matched == 1`, not the training target. Under the
    MTV label the two differ by class B (real but not an efficiency TP), and quoting recall on the
    target instead would silently move the working point by the class-B fraction."""
    A = np.asarray(A, dtype=bool)
    ss = s[A]; ww = w[A]
    if len(ss) == 0:
        return float("nan")
    order = np.argsort(ss)
    cum = np.cumsum(ww[order])
    return float(ss[order][min(np.searchsorted(cum, (1.0 - recall) * ww.sum()), len(ss) - 1)])


# Companions to the AUCs, reported only: neither early stopping nor the baked threshold uses them.
_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def partial_prauc(y_pos, s, rmin=0.99):
    """Area under the precision-recall curve restricted to recall >= `rmin`, normalised by
    (1 - rmin) so it lands in [0, 1]. precision_recall_curve returns recall DECREASING with a
    trailing (precision=1, recall=0) point, so the arrays are re-sorted to increasing recall, the
    precision at exactly `rmin` is interpolated in, and the integral is a trapezoid over recall."""
    p, r, _ = precision_recall_curve(y_pos, s)
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    o = np.argsort(r, kind="stable")
    r, p = r[o], p[o]
    m = r >= rmin
    if int(m.sum()) < 2:
        return None
    rr, pp = r[m], p[m]
    if rr[0] > rmin:
        rr = np.concatenate(([rmin], rr))
        pp = np.concatenate(([float(np.interp(rmin, r, p))], pp))
    return float(_TRAPZ(pp, rr) / (1.0 - rmin))


def precision_at_thr(s, A, C, t):
    """Precision on the A-vs-C subset at threshold `t`: class-A rows kept / all rows kept, i.e.
    1 - (fake contamination of what the gate lets through)."""
    if not np.isfinite(t):
        return None
    nA = int((s[A] >= t).sum())
    kept = nA + int((s[C] >= t).sum())
    return float(nA / kept) if kept else None


def train(paths, device, alpha=ALPHA, name=NAME, prefix="TrkDisp", label="mtv"):
    df = load_nano(paths, prefix, label=label)
    tr, va, te = split_by_event(df)
    Xtr, ytr, mean, std = xy(tr)
    Xva, yva, _, _ = xy(va, mean, std)
    Xte, yte, _, _ = xy(te, mean, std)
    # Up-weight the rarer large-|dxy| reals so the loss and the recall accounting follow displaced
    # tracks rather than the abundant prompt-like ones; alpha = 0 gives plain unweighted BCE.
    dxw = lambda d: np.where(d.label.values == 1,
                             1.0 + alpha * np.minimum(np.abs(d.dXY.values), 60.0) / 20.0, 1.0).astype(np.float32)
    wtr = dxw(tr); wva = dxw(va); wte = dxw(te)
    # A is the recall axis (matched); C is the rejection axis (target == 0). Class B, a target-1 row
    # outside A, is a training positive in neither axis, so neither number can be read off `label`.
    Ava = va.label_eff.values == 1; Cva = yva == 0
    Ate = te.label_eff.values == 1; Cte = yte == 0
    print(f"  classes: train A={int((tr.label_eff.values == 1).sum())} "
          f"pos(target)={int((ytr == 1).sum())} C={int((ytr == 0).sum())} | "
          f"test A={int(Ate.sum())} B={int(((yte == 1) & ~Ate).sum())} C={int(Cte.sum())}", flush=True)
    torch.manual_seed(7); dev = torch.device(device)
    net = MLPn(len(FEATS), HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    Xt = torch.tensor(Xtr.tolist(), device=dev); yt = torch.tensor(ytr.tolist(), device=dev)
    wt = torch.tensor(wtr.tolist(), device=dev)
    # Early stopping selects on weighted fake rejection at 99 % weighted recall.
    best = (-1.0, None, -1)
    for ep in range(MAX_EPOCHS):
        net.train(); perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 65536):
            idx = perm[i:i + 65536]
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]), yt[idx], weight=wt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sv = scores(net, Xva, dev)
        thr = _wthr_at_recall(Ava, sv, wva, 0.99)
        metric = float(1.0 - (sv[Cva] >= thr).mean())  # class-C rejection at the class-A wrecall point
        sched.step(metric)
        if metric > best[0]:
            best = (metric, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
        if ep - best[2] >= PATIENCE:
            print(f"  early stop at ep{ep} (no val improvement for {PATIENCE})", flush=True)
            break
    net.load_state_dict(best[1]); s = scores(net, Xte, dev)
    tip = np.abs(te.dXY.values)
    AC = Ate | Cte
    res = {"name": name, "feats": FEATS, "hidden": list(HIDDEN), "best_epoch": best[2],
           # The label this model was trained with, so a bake or a comparison cannot mistake it.
           "label": label,
           "target": "matchedAny" if label == "mtv" else "matched",
           "recall_axis": "matched",
           "alpha": float(alpha), "prefix": prefix,
           "n_A_test": int(Ate.sum()), "n_B_test": int(((yte == 1) & ~Ate).sum()),
           "n_C_test": int(Cte.sum()),
           "test_auc": float(roc_auc_score(yte, s)),
           "test_auc_AC": (float(roc_auc_score(Ate[AC], s[AC]))
                           if AC.sum() and len(np.unique(Ate[AC])) > 1 else None)}
    # Same test arrays and positive class as above; a degenerate split yields None, not an exception.
    y_ac, s_ac = Ate[AC], s[AC]
    ac_ok = bool(AC.sum()) and len(np.unique(y_ac)) > 1
    try:
        res["prauc_target"] = (float(average_precision_score(yte, s))
                               if len(np.unique(yte)) > 1 else None)
    except Exception:
        res["prauc_target"] = None
    try:
        res["prauc_AC"] = float(average_precision_score(y_ac, s_ac)) if ac_ok else None
    except Exception:
        res["prauc_AC"] = None
    try:
        res["partial_prauc_AC_r99"] = partial_prauc(y_ac, s_ac, 0.99) if ac_ok else None
    except Exception:
        res["partial_prauc_AC_r99"] = None
    try:
        # Precision at exactly the 99.5 % displacement-weighted recall point (thr_wrecall995).
        res["precision_AC_at_recall995"] = precision_at_thr(
            s, Ate, Cte, _wthr_at_recall(Ate, s, wte, 0.995))
    except Exception:
        res["precision_AC_at_recall995"] = None
    # Both displacement-weighted recall working points are tabulated; 99 % is the deployed one.
    for rc in (0.99, 0.995):
        thr = _wthr_at_recall(Ate, s, wte, rc)
        keep = lambda lo, hi: float((s[Ate & (tip >= lo) & (tip < hi)] >= thr).mean())
        tag = f"{int(rc * 1000)}"  # '990' / '995'
        res[f"thr_wrecall{tag}"] = thr
        res[f"freeFakeRej_{tag}"] = float(1.0 - (s[Cte] >= thr).mean())
        res[f"recallA_{tag}"] = float((s[Ate] >= thr).mean())  # unweighted class-A recall achieved
        res[f"realKeep_0_10_{tag}"] = keep(0, 10)
        res[f"realKeep_10_30_{tag}"] = keep(10, 30)
        res[f"realKeep_30_60_{tag}"] = keep(30, 60)
    res["thr_wrecall99"] = res["thr_wrecall990"]  # deploy point (bake reads this key)
    os.makedirs(ARTDIR, exist_ok=True)
    torch.save({"state": best[1], "mean": mean.tolist(), "std": std.tolist(), "feats": FEATS},
               os.path.join(ARTDIR, f"model_{name}.pt"))
    json.dump(res, open(os.path.join(ARTDIR, f"result_{name}.json"), "w"), indent=1)
    _n = lambda v: ("%.5f" % v) if isinstance(v, float) and np.isfinite(v) else "n/a"
    print("  PR (info only)   prAUC=%s  prAUC(A/C)=%s  partial-prAUC(A/C, recall>=0.99)=%s  "
          "precision(A/C)@wrecallA=0.995 = %s"
          % (_n(res["prauc_target"]), _n(res["prauc_AC"]), _n(res["partial_prauc_AC_r99"]),
             _n(res["precision_AC_at_recall995"])), flush=True)
    print(json.dumps(res, indent=1))


def _fmt(name, arr):
    body = ", ".join("%.8ef" % v for v in np.asarray(arr, np.float64).ravel())
    return "  HOST_DEVICE_CONSTANT float %s[%d] = {%s};\n" % (name, np.asarray(arr).size, body)


def bake(threshold, out, bank="displaced", name=NAME):
    """Write CATrackDNNWeights_<bank>.h from model_<name>.pt. Self-contained (model .pt
    has weights + standardization); threshold is the high-recall operating point. bank =
    "displaced" | "prompt" selects the per-iteration weight bank (namespace caTrackDNN_<bank>);
    each bank is its own header so prompt/displaced bakes never clobber each other (the
    dispatcher CATrackDNNWeights.h includes both)."""
    assert bank in ("displaced", "prompt"), bank
    guard = "RecoTracker_PixelSeeding_plugins_alpaka_CATrackDNNWeights_%s_h" % bank
    ns = "caTrackDNN_%s" % bank
    blob = torch.load(os.path.join(ARTDIR, f"model_{name}.pt"), map_location="cpu")
    sd = blob["state"]; lin = [k for k in sd if k.endswith(".weight")]
    W = [np.asarray(sd[k].tolist(), np.float64) for k in lin]
    B = [np.asarray(sd[k.replace(".weight", ".bias")].tolist(), np.float64) for k in lin]
    with open(out, "w") as f:
        f.write("#ifndef %s\n" % guard)
        f.write("#define %s\n\n" % guard)
        f.write("// AUTO-GENERATED by test/models/train_disp_nano.py bake --bank %s -- do not edit by hand.\n" % bank)
        f.write("// Loose->tight track selector: 12 features, 32x16, high-recall point.\n\n")
        f.write("#include <alpaka/alpaka.hpp>\n\n")
        f.write('#include "FWCore/Utilities/interface/HostDeviceConstant.h"\n\n')
        f.write("namespace ALPAKA_ACCELERATOR_NAMESPACE::%s {\n\n" % ns)
        f.write(f"  constexpr int kNFeat = {len(FEATS)};\n")
        f.write(f"  constexpr int kNH1 = {HIDDEN[0]};\n  constexpr int kNH2 = {HIDDEN[1]};\n")
        f.write(f"  constexpr float kDefaultThreshold = {threshold:.6e}f;\n\n")
        f.write(_fmt("kMean", blob["mean"])); f.write(_fmt("kScale", blob["std"]))
        f.write(_fmt("kW1", W[0].T)); f.write(_fmt("kB1", B[0]))
        f.write(_fmt("kW2", W[1].T)); f.write(_fmt("kB2", B[1]))
        f.write(_fmt("kW3", W[2].T)); f.write(_fmt("kB3", B[2]))
        f.write("\n}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::%s\n\n#endif\n" % ns)
    print(f"baked {out}  (bank={bank}, kNFeat={len(FEATS)}, hidden={HIDDEN}, threshold={threshold:.6f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "bake"])
    ap.add_argument("nanos", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--recall", type=float, default=0.999)
    ap.add_argument("--threshold", type=float, default=None, help="bake: explicit threshold (else from result json)")
    ap.add_argument("--bake-recall", type=float, default=0.99, dest="bake_recall",
                    help="bake: which displacement-weighted recall working point of the result json "
                         "to take -- 0.99 (default, the deployed point) or 0.995")
    ap.add_argument("--bank", choices=["displaced", "prompt"], default="displaced",
                    help="per-iteration weight bank to bake (default: displaced)")
    ap.add_argument("--out", default=None,
                    help="output header (default: ../plugins/alpaka/CATrackDNNWeights_<bank>.h)")
    ap.add_argument("--prefix", default=None, help="nano table prefix (default: TrkDisp / TrkPrompt by bank)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="displacement-weight strength (default: 3.0 displaced / 0.0 prompt)")
    ap.add_argument("--name", default=None, help="model/result tag (default: <bank>_stage1_12f)")
    ap.add_argument("--label", choices=("mtv", "legacy"), default="mtv",
                    help="training target: 'mtv' (default) = matchedAny, the DQM/MTV fake "
                         "definition, with `matched` as the recall axis; 'legacy' = matched, "
                         "which makes real non-efficiency tracks negatives")
    args = ap.parse_args()
    # Per-bank defaults: displaced = TrkDisp + displacement weighting; prompt = TrkPrompt + plain BCE.
    prefix = args.prefix if args.prefix else ("TrkPrompt" if args.bank == "prompt" else "TrkDisp")
    alpha = args.alpha if args.alpha is not None else (0.0 if args.bank == "prompt" else ALPHA)
    name = args.name if args.name else (NAME if args.bank == "displaced" else "prompt_stage1_12f")
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../plugins/alpaka/CATrackDNNWeights_%s.h" % args.bank)
    if args.mode == "train":
        train(args.nanos, args.device, alpha=alpha, name=name, prefix=prefix, label=args.label)
    else:
        thr = args.threshold
        if thr is None:
            res = json.load(open(os.path.join(ARTDIR, f"result_{name}.json")))
            # train() writes one 'thr_wrecall<tag>' key per measured point, tag = int(recall*1000).
            key = "thr_wrecall%d" % round(args.bake_recall * 1000)
            if key not in res:
                have = sorted(k for k in res if k.startswith("thr_wrecall"))
                raise SystemExit("train_disp_nano: no working point at recall %g (%s) in "
                                 "result_%s.json; it holds %s" % (args.bake_recall, key, name, have))
            thr = res[key]
            print("baking the %.1f%% displacement-weighted recall point (%s = %.6g)"
                  % (100.0 * args.bake_recall, key, thr), flush=True)
        bake(thr, args.out, bank=args.bank, name=name)


if __name__ == "__main__":
    main()
