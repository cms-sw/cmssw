"""Final high-purity selector, NEURAL-NETWORK family: train and export.

The deployed final selectors are gradient-boosted forests (build_tree_model.py), which also
need far less device memory than a network of this shape. This trains the network candidates
of the comparison `retrain_prompt.sh hp` runs (--class mlp), or a displaced network.

Trains the MLP on Trk<prefix>Full features (+ the 10 hit/stub CA features with --feat-set
union) and Trk<prefix>Truth labels, then exports the drop-in TorchScript for
PixelTrackTorchHighPuritySelector.

  # prompt HP:  python3 train_prompt_hp_nano.py train <nano.root>; finalize <...> --thr-recall 0.99
  # displaced:  train <a.root>... --prefix TrkDisp --min-dxy 0.5 --name ...; finalize --prefix TrkDisp ...

Artifacts (intermediate + finalized .pt) -> $MODELS_WORKDIR. Copy the .pt into
data/PixelTrackTorchHighPuritySelector/ to deploy.
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from nano_train_utils import MLPn, ExportModule, scores, thr_at_recall

from nano_loader import FEATS17, FEATS_UNION, load_cache, load_nano

ARTDIR = os.environ.get("MODELS_WORKDIR", os.getcwd())  # intermediate + finalized artifacts
# Network shape of the Torch selector: 17 -> 64x4 -> 1. A wider net pushes the per-batch activation
# past Torch's 1 MB small/large-pool boundary, costing 20 MB per stream instead of 2 MB.
HIDDEN = (64, 64, 64, 64)


def feats_for(feat_set):
    """fit17 = the 17 fit/cov features; union = 17 fit + 10 hit/stub CA features
    (PixelTrackFeaturesSoA columns 1-27, the in-kernel add() order)."""
    return {"fit17": FEATS17, "union": FEATS_UNION}[feat_set]


def _load(paths, prefix, max_events, min_dxy, with_ca=False, cache=None, feats=None):
    """Concatenate one-or-more trackNano files for the given prefix, optionally restrict to
    the displaced population by |dxyBS| > min_dxy. with_ca=True also loads the 10 hit/stub CA features.

    cache: read a pre-built .npz instead of the ROOT nanos. Use this whenever the result has to
    be comparable to another model class -- it guarantees identical rows and an evt%10 split
    (loading from ROOT applies different NaN policies). paths is ignored when cache is given."""
    if cache is not None:
        df = load_cache(cache, feats)
    else:
        df = pd.concat([load_nano(p, prefix, max_events, with_ca=with_ca) for p in paths],
                       ignore_index=True)
    if min_dxy > 0.0:
        n0 = len(df)
        df = df[np.abs(df["dxyBS"].values) > min_dxy].reset_index(drop=True)
        print(f"  |dxyBS|>{min_dxy}: kept {len(df)}/{n0} rows  real={df.label.mean():.4f}", flush=True)
    return df


def split_by_event(df):
    m = df.evt.values.astype(int) % 10
    return df[m >= 4], df[m == 3], df[m < 3]  # train, val, test


def xy(d, feats, mean=None, std=None):
    X = d[feats].values.astype(np.float32)
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32), d.label.values.astype(np.float32), mean, std


def _wfocal(logits, y, gamma, w):
    """Per-sample-weighted (optionally focal) BCE. gamma=0 -> weighted BCE."""
    ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if gamma > 0:
        p = torch.sigmoid(logits)
        pt = torch.where(y == 1, p, 1 - p)
        ce = (1 - pt) ** gamma * ce
    return (w * ce).mean()


def train(paths, device, max_events, prefix, min_dxy, name, hidden=HIDDEN, gamma=0.0, disp_w=0.0,
          feat_set="fit17", cache=None):
    feats = feats_for(feat_set)
    df = _load(paths, prefix, max_events, min_dxy, with_ca=(feat_set == "union"),
               cache=cache, feats=feats if cache else None)
    tr, va, te = split_by_event(df)
    fake_frac = 1.0 - tr.label.mean()
    # per-sample weight = class balance (down-weight dominant reals) x displacement emphasis
    # (1 + disp_w*min(|dxyBS|,60)/20), which sharpens the high-|dxy| regime the displaced selector
    # works in. gamma>0 adds focal modulation.
    pw_val = fake_frac / max(1e-3, 1.0 - fake_frac)

    def wts(d):
        w = np.where(d.label.values == 1, pw_val, 1.0).astype(np.float32)
        if disp_w > 0.0:
            w = w * (1.0 + disp_w * np.minimum(np.abs(d.dxyBS.values), 60.0) / 20.0).astype(np.float32)
        return w

    print(f"population: train={len(tr)} val={len(va)} test={len(te)}  fake_frac={fake_frac:.4f} "
          f"pos_w={pw_val:.3f} gamma={gamma} disp_w={disp_w} hidden={tuple(hidden)}", flush=True)

    Xtr, ytr, mean, std = xy(tr, feats)
    Xva, yva, _, _ = xy(va, feats, mean, std)
    Xte, yte, _, _ = xy(te, feats, mean, std)

    torch.manual_seed(7)
    dev = torch.device(device)
    model = MLPn(len(feats), hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    Xtr_t = torch.tensor(Xtr.tolist(), device=dev)
    ytr_t = torch.tensor(ytr.tolist(), device=dev)
    wtr_t = torch.tensor(wts(tr).tolist(), device=dev)
    n = len(Xtr_t)
    best = (-1.0, None, -1)
    patience = 20
    for ep in range(250):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i0 in range(0, n, 65536):
            idx = perm[i0:i0 + 65536]
            loss = _wfocal(model(Xtr_t[idx]), ytr_t[idx], gamma, wtr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        s_va = scores(model, Xva, dev)
        thr = thr_at_recall(yva, s_va, 0.99)
        rej = float(1.0 - (s_va[yva == 0] >= thr).mean())
        auc = float(roc_auc_score(yva, s_va))
        sched.step(rej)
        mark = ""
        if rej > best[0]:
            best = (rej, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            mark = "  <- best"
        print(f"  ep{ep:02d}  val fakeRej@99%={rej:.4f}  val AUC={auc:.4f}{mark}", flush=True)
        if ep - best[2] >= patience:
            print(f"  early stop (no improvement for {patience} epochs)")
            break

    model.load_state_dict(best[1])
    s_te = scores(model, Xte, dev)
    res = {"name": name, "prefix": prefix, "min_dxy": min_dxy, "feats": feats, "feat_set": feat_set,
           "hidden": list(hidden), "gamma": gamma, "disp_w": disp_w, "pos_w": pw_val,
           "fake_frac": float(fake_frac), "val_rej99": best[0], "best_epoch": best[2],
           "test_auc": float(roc_auc_score(yte, s_te))}
    # operating points: recall = fraction of real tracks kept; fake-in-accepted reported too
    for rc in (0.999, 0.995, 0.99, 0.98):
        thr = thr_at_recall(yte, s_te, rc)
        acc = s_te >= thr
        res[f"rej@r{rc}"] = float(1.0 - (s_te[yte == 0] >= thr).mean())
        res[f"thr@r{rc}"] = float(thr)
        res[f"fakeInAcc@r{rc}"] = float((yte[acc] == 0).mean())
    os.makedirs(ARTDIR, exist_ok=True)
    torch.save({"state": best[1], "mean": mean.tolist(), "std": std.tolist(), "feats": feats,
                "feat_set": feat_set, "hidden": list(hidden)},
               os.path.join(ARTDIR, f"model_{name}.pt"))
    json.dump(res, open(os.path.join(ARTDIR, f"result_{name}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def finalize(paths, thr_recall, max_events, prefix, min_dxy, name, ts_name, precision="fp16",
             cache=None):
    blob = torch.load(os.path.join(ARTDIR, f"model_{name}.pt"), map_location="cpu")
    mean = np.asarray(blob["mean"], dtype=np.float32)
    std = np.asarray(blob["std"], dtype=np.float32)
    feats = blob.get("feats", FEATS17)  # exact trained feature order/dim
    feat_set = blob.get("feat_set", "fit17")
    net = MLPn(len(feats), tuple(blob.get("hidden", HIDDEN)))
    net.load_state_dict(blob["state"])
    net.eval()
    mod = ExportModule(net.net, mean, std).eval()

    df = _load(paths, prefix, max_events, min_dxy, with_ca=(feat_set == "union"),
               cache=cache, feats=feats if cache else None)
    _, _, te = split_by_event(df)
    te = te[np.isfinite(te[feats].values).all(axis=1)]  # drop NaN-feature rows (as train does)
    Xraw = te[feats].values.astype(np.float32)
    y = te.label.values.astype(np.float32)
    # Both precisions are exported; fp16 is the default. The producer runs forward(kHalf), so fp16
    # storage matches the runtime dtype, drops the per-forward cast and keeps the per-batch activation
    # in Torch's small-allocation pool (<1 MB -> 2 MB blocks instead of 20 MB/stream). The fp32 file
    # is written alongside (same name + _fp32/_fp16 suffix) for inferenceHalf=false.
    def _script(half):
        m = copy.deepcopy(mod).eval()
        return torch.jit.script(m.half() if half else m)
    sm = {"fp16": _script(True), "fp32": _script(False)}
    other = "fp32" if precision == "fp16" else "fp16"
    base, ext = os.path.splitext(ts_name)
    out = os.path.join(ARTDIR, ts_name)                       # primary (deployed) = --precision
    out_other = os.path.join(ARTDIR, f"{base}_{other}{ext}")
    sm[precision].save(out)
    sm[other].save(out_other)
    print(f"saved TorchScript -> {out} ({precision}, deployed)  +  {out_other} ({other})")
    scripted = sm["fp32"]  # the validation below feeds fp16 input to the fp32 graph
    # ExportModule returns [N,1] (producer's required output shape); squeeze for metrics.
    with torch.no_grad():
        s32 = np.asarray(mod(torch.tensor(Xraw.tolist(), dtype=torch.float32)).tolist()).squeeze(-1)
        s16 = np.asarray(mod(torch.tensor(Xraw.tolist(), dtype=torch.float16)).tolist()).squeeze(-1)
        sjs = np.asarray(scripted(torch.tensor(Xraw.tolist(), dtype=torch.float16)).tolist()).squeeze(-1)
    # FP16 overflows to NaN on rare extreme-pt tracks (deployment treats NaN as below-thr,
    # i.e. dropped -> benign); mask non-finite for the offline metrics.
    m = np.isfinite(s16) & np.isfinite(s32)
    nbad = int((~m).sum())
    print(f"FP16 non-finite scores: {nbad}/{len(s16)} (dropped from metrics); "
          f"max |f32-f16in|(finite)={np.abs(s32[m] - s16[m]).max():.3e}  "
          f"|eager-scripted|={np.abs(s16[m] - sjs[m]).max():.3e}")
    s16m, ym = s16[m], y[m]
    print(f"test AUC (half input) = {roc_auc_score(ym, s16m):.4f}")
    # report the 99% and 99.5% recall points, and the chosen deploy thr-recall
    for rc in sorted({0.99, 0.995, thr_recall}):
        thr = float(thr_at_recall(ym, s16m, rc))
        acc = s16m >= thr
        tag = "  <- deploy" if abs(rc - thr_recall) < 1e-9 else ""
        print(f"  recall {rc:.1%}: scoreThreshold={thr:.4f}  "
              f"fakeRej={float(1.0 - (s16m[ym == 0] >= thr).mean()):.4f}  "
              f"fakeInAcc={float((ym[acc] == 0).mean()):.4f}{tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "finalize"])
    ap.add_argument("nano", nargs="*", default=[])
    ap.add_argument("--cache", default=None,
                    help="read a pre-built .npz feature cache instead of the ROOT nanos "
                         "(nano_loader.py cache-prompt27 / cache-prompt). REQUIRED for a fair "
                         "bake-off: it pins identical rows + identical evt%%10 split across model "
                         "classes. When given, the positional nano args are ignored.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--thr-recall", type=float, default=0.99)
    ap.add_argument("--prefix", default="TrkPrompt", help="table prefix: TrkPrompt | TrkDisp")
    ap.add_argument("--min-dxy", type=float, default=0.0, help="restrict to reco |dxyBS|>this (cm)")
    ap.add_argument("--name", default="prompt_hp64_nano", help="model/result artifact name")
    ap.add_argument("--ts-name", default="pixel_track_classifier_newfit.pt",
                    help="finalize: TorchScript output filename (under ARTDIR)")
    ap.add_argument("--hidden", default="64,64,64,64",
                    help="comma-separated hidden layer widths (baseline/deployed = 64,64,64,64)")
    ap.add_argument("--gamma", type=float, default=0.0, help="focal-loss gamma (0 = plain weighted BCE)")
    ap.add_argument("--disp-weight", type=float, default=0.0,
                    help="displacement loss weight beta: per-sample x (1 + beta*min(|dxyBS|,60)/20)")
    ap.add_argument("--feat-set", default="fit17", choices=["fit17", "union"],
                    help="fit17 = 17 fit/cov feats; union = 17 fit + 10 hit/stub CA feats (TrkDisp only)")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                    help="finalize: precision of the primary --ts-name file (default fp16, deployed); "
                         "the other precision is always written alongside with an _fp32/_fp16 suffix")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))
    if args.mode == "train":
        train(args.nano, args.device, args.max_events, args.prefix, args.min_dxy, args.name,
              hidden, args.gamma, args.disp_weight, args.feat_set, args.cache)
    else:
        finalize(args.nano, args.thr_recall, args.max_events, args.prefix, args.min_dxy,
                 args.name, args.ts_name, args.precision, args.cache)


if __name__ == "__main__":
    main()
