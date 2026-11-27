#!/usr/bin/env python3
"""Final high-purity selector of the baseline pixel chain: retrain the Torch network.

The baseline chain (no process modifier) runs PixelTrackTorchHighPuritySelector@alpaka with the
TorchScript file pixel_track_classifier_FP16.pt. This script retrains that network with its
architecture, its preprocessing and its 17 inputs unchanged, and derives its scoreThreshold. The
vector hit chain runs a forest instead (train_merged_forest.py).

  check   show that the classes below are the deployed model: they are loaded with the weights
          of the deployed file and compared with it (parameters, buffers, TorchScript code of every
          submodule, scores on the cache rows)
  train   fit the network on a 17-feature cache, derive the working points against the deployed
          file, export the new TorchScript file in FP16 (deployed) and FP32

  python3 train_torch_hp_dnn.py check --reference <deployed .pt> [--cache <cache.npz>]
  python3 train_torch_hp_dnn.py train --cache <cache.npz> [more.npz] --out <dir> \\
          --reference <deployed .pt> --reference-threshold <its scoreThreshold> \\
          [--recall 0.995] [--date YYYYMMDD]

The cache is written by `nano_loader.py cache-arm <out.npz> <datasets> --prefix TrkPrompt
--label mtv --feats 17` from track datasets dumped on the baseline chain at quality 'tight'. The
deployed file (--reference) is the architecture template and is scored on the same rows, at its
own threshold, as the reference row of the report.

THE MODEL (fixed; taken from the deployed file, never changed by a retrain)
  inputs     the 17 fit features of PixelTrackFitFeaturesSoA, in FEATS17 order
  preprocess log10(|x| + 1e-8) on the features of LOG_FEATURES, then a clamp to [clip_min,
             clip_max] on the features of CLIPPED_FEATURES (the others are not clamped), then
             (x - mean) / (std + 1e-8); NaN mapped to numbers after it
  network    4 x (Linear 64, LayerNorm, ReLU, Dropout 0.05), Linear 1, sigmoid; output [N, 1]
  precision  FP16 parameters and buffers (the producer runs the forward pass in half precision)
A retrain recomputes the numbers only: mean, std and the clamp limits from the training rows (the
clamp limits are the CLIP_QUANTILES of each clamped feature), and the weights.

TRAINING (the defaults are the recipe shared with train_merged_forest.py, see README.md)
  target     y_mtv (matchedAny: matched to any TrackingParticle, the track validation's "not fake")
  split      evt % 10: >= 4 train, == 3 validation, < 3 test, as every trainer here
  recall     on the non-duplicate efficiency tracks: what the track validation counts as
             efficiency, a particle found once
  loss       binary cross entropy with these weights, multiplied:
             class balance: true tracks x fake_frac / (1 - fake_frac);
             --pos-scale K (4): every true track x K;
             --region-pt-weight A (4): every track x (1 + A log10(max(pT, 0.9) / 0.9));
             --eff-weight A (1): the non-duplicate efficiency tracks x the same factor;
             --dup-as-fake W (1): duplicates are negatives with weight W (0 = they stay positive)
  optimiser  Adam, learning rate 1e-3 halved on a plateau of the validation metric; early stop
             after --patience epochs without improvement; the best epoch is kept
  metric     fake rejection on the validation split at --recall

WORKING POINT. One scoreThreshold: the threshold at which the recall of the non-duplicate
efficiency tracks is --recall (0.995). Rejection is on the rows with y_mtv = 0. It is computed on
the test split, with the exported FP16 file scored the way the producer scores it (inputs and
model in half precision), and reported with the recall and rejection in pT, |eta| and |dxyBS|
bins (the bins of train_merged_forest.py).

OUTPUTS in --out: pixel_track_classifier_FP16_<date>.pt (deployed),
pixel_track_classifier_FP32_<date>.pt, result.json (recipe, preprocessing, metrics, the working
point with its per-bin tables), thrmap.txt (threshold against recall and rejection).
"""
import argparse
import copy
import datetime
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nano_loader import FEATS17  # noqa: E402
import train_merged_forest as tmf  # noqa: E402

HIDDEN = 64
NLAYERS = 4
DROPOUT = 0.05
# Taken from the deployed pixel_track_classifier_FP16.pt; `check` verifies them against a file.
LOG_FEATURES = ["dzError", "dxyError", "phiError", "pt", "qOverPtError", "cotThetaError",
                "covCotThetaDz", "covDxyQOverPt", "covPhiDxy", "covPhiQOverPt"]
CLIPPED_FEATURES = ["chi2", "dzError", "dxyError", "phiError", "pt", "covDxyQOverPt",
                    "covPhiDxy", "covPhiQOverPt"]
CLIP_QUANTILES = (0.001, 0.999)


def _aslist(v, kind=float):
    return [kind(x) for x in np.asarray(v).reshape(-1).tolist()]


# ---- the model: the class and attribute names are those of the deployed TorchScript file ----
class FeaturePreprocessing(nn.Module):
    def __init__(self, mean, std, clip_min, clip_max, do_log):
        super().__init__()
        # Through lists: the CMSSW torch is built without the NumPy bridge.
        self.register_buffer("mean", torch.tensor(_aslist(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(_aslist(std), dtype=torch.float32))
        self.register_buffer("clip_min", torch.tensor(_aslist(clip_min), dtype=torch.float32))
        self.register_buffer("clip_max", torch.tensor(_aslist(clip_max), dtype=torch.float32))
        self.register_buffer("do_log", torch.tensor(_aslist(do_log, bool), dtype=torch.bool))

    def _match_shape(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if p.numel() == 0:
            return p
        while p.dim() < x.dim():
            p = p.unsqueeze(0)
        return p.expand_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_log.numel() > 0:
            log_mask = self._match_shape(self.do_log, x) != 0
            x = torch.where(log_mask, torch.log10(torch.abs(x) + 1e-8), x)
        if self.clip_min.numel() > 0 or self.clip_max.numel() > 0:
            clip_min: Optional[torch.Tensor] = (self._match_shape(self.clip_min, x)
                                                if self.clip_min.numel() > 0 else None)
            clip_max: Optional[torch.Tensor] = (self._match_shape(self.clip_max, x)
                                                if self.clip_max.numel() > 0 else None)
            x.clamp_(clip_min, clip_max)
        if self.mean.numel() > 0 and self.std.numel() > 0:
            mean = self._match_shape(self.mean, x)
            std = self._match_shape(self.std, x)
            x.sub_(mean)
            x.div_(std + 1e-8)
        return x


class NetBTrackOnly(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        layers, prev = [], nfeat
        for _ in range(NLAYERS):
            layers += [nn.Linear(prev, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = HIDDEN
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, track_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(track_features).squeeze(-1)


class TrackOnlyClassifier(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        self.netB = NetBTrackOnly(nfeat)

    def forward(self, track_features: torch.Tensor) -> torch.Tensor:
        return self.netB(track_features)


class TrackOnlyClassifierInference(nn.Module):
    def __init__(self, pre_track, base):
        super().__init__()
        self.base = base  # registered first, as in the deployed file
        self.pre_track = pre_track

    def forward(self, track_features: torch.Tensor) -> torch.Tensor:
        track_features = self.pre_track(track_features)
        track_features = torch.nan_to_num_(track_features)
        logits = self.base(track_features)
        probs = torch.sigmoid(logits)
        return probs.reshape(-1, 1)


def build(mean, std, clip_min, clip_max, do_log):
    return TrackOnlyClassifierInference(FeaturePreprocessing(mean, std, clip_min, clip_max, do_log),
                                        TrackOnlyClassifier(len(FEATS17)))


# ---- helpers ----
def load_reference(path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.jit.load's Python 3.14 notice
        return torch.jit.load(path, map_location="cpu")


def submodule_codes(m):
    """TorchScript code of every submodule, keyed by its name ('' = the top)."""
    out = {"": m.code}
    for name, sub in m.named_modules():
        if name and hasattr(sub, "code"):
            out[name] = sub.code
    out["pre_track._match_shape"] = m.pre_track._match_shape.code
    return out


def as_producer(X):
    """The producer converts the inputs to half precision before the forward pass."""
    return X.astype(np.float16).astype(np.float32)


def score(mod, X, device, half=True, bs=262144):
    """Score rows with a TorchScript (or eager) model, the way the producer does: model and inputs
    in half precision on the device. Non-finite scores (FP16 overflow) count as rejected."""
    mod = mod.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xt = torch.tensor(X[i:i + bs].tolist(), dtype=torch.float32, device=device)
            if half:
                xt = xt.half()
            out.append(np.asarray(mod(xt).float().cpu().tolist(), dtype=np.float64).reshape(-1))
    s = np.concatenate(out) if out else np.zeros(0)
    return np.where(np.isfinite(s), s, -1.0)


def load_cache17(paths):
    d = tmf.load_caches(paths, want_label="mtv")
    if d["feats"][:17] != FEATS17:
        raise SystemExit("the cache does not start with the 17 fit features in FEATS17 order: %s"
                         % d["feats"][:17])
    d["X"] = d["X"][:, :17]
    return d


def deployed_constants(ref):
    """The preprocessing buffers of a deployed file, as numpy arrays."""
    b = dict(ref.pre_track.named_buffers())
    return {k: np.asarray(b[k].float().tolist() if k != "do_log" else b[k].tolist()) for k in b}


# ---- check ----
def cmd_check(a):
    ref = load_reference(a.reference)
    c = deployed_constants(ref)
    do_log = [FEATS17[i] for i in range(17) if c["do_log"][i]]
    clipped = [FEATS17[i] for i in range(17)
               if np.isfinite(c["clip_min"][i]) or np.isfinite(c["clip_max"][i])]
    ok = True
    print("deployed file: %s" % a.reference)
    print("  log10 features : %s" % do_log)
    print("  clamped        : %s" % clipped)
    if do_log != LOG_FEATURES or clipped != CLIPPED_FEATURES:
        print("  MISMATCH with LOG_FEATURES / CLIPPED_FEATURES of this script")
        ok = False
    mine = build(c["mean"], c["std"], c["clip_min"], c["clip_max"], c["do_log"])
    ref_sd = ref.state_dict()
    my_sd = mine.state_dict()
    if [(k, tuple(v.shape)) for k, v in ref_sd.items()] != [(k, tuple(v.shape)) for k, v in my_sd.items()]:
        print("  MISMATCH of parameter and buffer names or shapes:")
        print("    deployed: %s" % [(k, tuple(v.shape)) for k, v in ref_sd.items()])
        print("    here    : %s" % [(k, tuple(v.shape)) for k, v in my_sd.items()])
        return 1
    mine.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ref_sd.items()})
    scripted = torch.jit.script(copy.deepcopy(mine).eval().half())
    print("  parameters and buffers: %d, names, shapes and dtypes %s" % (
        len(ref_sd), "identical" if all(ref_sd[k].dtype == v.dtype for k, v in
                                         scripted.state_dict().items()) else "DIFFER"))
    rc, mc = submodule_codes(ref), submodule_codes(scripted)
    for k in rc:
        same = rc[k] == mc.get(k)
        ok &= same
        print("  TorchScript code %-22s %s" % (k or "<top>", "identical" if same else "DIFFERS"))
        if not same and a.verbose:
            print("---- deployed\n%s\n---- here\n%s" % (rc[k], mc.get(k)))
    rn = [(n, m.original_name) for n, m in ref.named_modules()]
    mn = [(n, m.original_name) for n, m in scripted.named_modules()]
    ok &= rn == mn
    print("  submodule class names: %s  (%s)" % ("identical" if rn == mn else "DIFFER",
                                                ", ".join(sorted({c for _, c in rn}))))
    if a.cache:
        X = as_producer(load_cache17(a.cache)["X"])[: a.max_rows]
    else:
        X = np.random.default_rng(1).normal(size=(4096, 17)).astype(np.float32)
    for dev in [d for d in ("cpu", "cuda:0") if d == "cpu" or torch.cuda.is_available()]:
        try:
            s_ref = score(ref, X, dev)
            s_me = score(scripted, X, dev)
        except RuntimeError as e:
            print("  scores on %s: not available (%s)" % (dev, str(e).splitlines()[0]))
            continue
        diff = np.abs(s_ref - s_me)
        print("  scores on %s, %d rows: max |deployed - rebuilt| = %.3g" % (dev, len(X), diff.max()))
        ok &= bool(diff.max() == 0.0)
    print("RESULT: %s" % ("the rebuilt classes are the deployed model" if ok else "DIFFERENCES above"))
    return 0 if ok else 1


# ---- train ----
def preprocessing_constants(Xtr):
    """Clamp limits (CLIP_QUANTILES of each clamped feature, after the log), mean and std, from the
    training rows, in the order the forward pass applies them."""
    do_log = np.array([f in LOG_FEATURES for f in FEATS17])
    Z = Xtr.astype(np.float64).copy()
    Z[:, do_log] = np.log10(np.abs(Z[:, do_log]) + 1e-8)
    cmin = np.full(17, -np.inf)
    cmax = np.full(17, np.inf)
    for i, f in enumerate(FEATS17):
        if f in CLIPPED_FEATURES:
            cmin[i], cmax[i] = np.quantile(Z[:, i], CLIP_QUANTILES)
    Z = np.clip(Z, cmin, cmax)
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std[std == 0] = 1.0
    return mean, std, cmin, cmax, do_log


def thrmap(s, A, C, n=200):
    lines = ["# threshold  recall(y_eff)  fake_rejection(y_mtv=0)"]
    for t in np.unique(np.quantile(s[A], np.linspace(0.0, 0.05, n))):
        lines.append("%.6f %.6f %.6f" % (t, (s[A] >= t).mean(), 1.0 - (s[C] >= t).mean()))
    return "\n".join(lines) + "\n"


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    date = a.date or datetime.date.today().strftime("%Y%m%d")
    d = load_cache17(a.cache)
    X = as_producer(d["X"])
    y = d["y_mtv"].astype(np.float32)
    A_all = d["y_eff"] >= 0.5
    C_all = d["y_mtv"] < 0.5
    dup_all = d["duplicate"] >= 0.5
    if a.dup_as_fake > 0:
        # Duplicates (true, but a second track of a TrackingParticle another track found) are
        # negatives of the loss, as train_merged_forest.py --dup-as-fake does; they do not join
        # the fakes of the metric.
        y = np.where(dup_all, 0.0, y).astype(np.float32)
        print("dup-as-fake: %d duplicates are negatives, loss weight x%.2f"
              % (int(dup_all.sum()), a.dup_as_fake), flush=True)
    # Recall on the non-duplicate efficiency tracks: the validation counts a particle found once,
    # so removing a duplicate costs no efficiency.
    A_all = A_all & ~dup_all
    print("recall axis: non-duplicate efficiency tracks (%d)" % int(A_all.sum()), flush=True)
    tr, va, te = tmf.make_split(d["evt"], "evt10", 0.1, 0.3, 0)
    print("rows %d: train %d / val %d / test %d; fake fraction (y_mtv = 0) %.4f"
          % (len(X), tr.sum(), va.sum(), te.sum(), C_all.mean()), flush=True)

    mean, std, cmin, cmax, do_log = preprocessing_constants(X[tr])
    for i, f in enumerate(FEATS17):
        print("  %-14s log=%d clip=[%9.4f, %9.4f] mean=%9.4f std=%8.4f"
              % (f, do_log[i], cmin[i], cmax[i], mean[i], std[i]))
    model = build(mean, std, cmin, cmax, do_log)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    model = model.to(dev)

    fake_frac = float(1.0 - y[tr].mean())
    pos_w = fake_frac / max(1e-3, 1.0 - fake_frac)
    print("true-track weight %.4f (fake fraction %.4f) x pos-scale %.2f"
          % (pos_w, fake_frac, a.pos_scale), flush=True)
    w_np = np.where(y[tr] > 0.5, pos_w * a.pos_scale, 1.0)
    if a.dup_as_fake > 0:
        w_np = w_np * np.where(dup_all[tr], a.dup_as_fake, 1.0)
    # From the full-precision pT, as the forest trainer computes it: X holds the inputs rounded
    # to half precision, where a fitted pT above 65504 GeV is infinite.
    ptw = 1.0 + np.log10(np.maximum(d["X"][tr][:, FEATS17.index("pt")].astype(np.float64), 0.9) / 0.9)
    if not np.isfinite(ptw).all():
        raise SystemExit("non-finite pT in the training rows")
    if a.eff_weight > 0:
        w_np = w_np * np.where(A_all[tr], 1.0 + a.eff_weight * (ptw - 1.0), 1.0)
        print("eff-weight: A=%.2f on efficiency tracks" % a.eff_weight, flush=True)
    if a.region_pt_weight > 0:
        w_np = w_np * (1.0 + a.region_pt_weight * (ptw - 1.0))
        print("region-pt-weight: A=%.2f on every row" % a.region_pt_weight, flush=True)

    def tens(m):
        return torch.tensor(X[m].tolist(), dtype=torch.float32, device=dev)
    # Preprocess once: the constants are fixed, only the network trains.
    with torch.no_grad():
        Ptr = torch.nan_to_num_(model.pre_track(tens(tr)))
        Pva = torch.nan_to_num_(model.pre_track(tens(va)))
    ytr = torch.tensor(y[tr].tolist(), device=dev)
    wtr = torch.tensor(w_np.astype(np.float32).tolist(), device=dev)
    A_va, C_va = A_all[va], C_all[va]

    net = model.base
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best = (-1.0, None, -1)
    n = len(Ptr)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i0 in range(0, n, a.batch):
            idx = perm[i0:i0 + a.batch]
            loss = (wtr[idx] * bce(net(Ptr[idx]), ytr[idx])).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            s_va = np.asarray(torch.sigmoid(net(Pva)).cpu().tolist(), dtype=np.float64)
        t = tmf.thr_at_recall(s_va, A_va, a.recall)
        rej = tmf.rej_at_thr(s_va, C_va, t)
        sched.step(rej)
        mark = ""
        if rej > best[0]:
            best = (rej, copy.deepcopy(net.state_dict()), ep)
            mark = "  <- best"
        print("  epoch %3d  loss %.5f  val fake rejection at %.1f%% recall %.5f  lr %.2e%s"
              % (ep, tot / n, 100 * a.recall, rej, opt.param_groups[0]["lr"], mark), flush=True)
        if ep - best[2] >= a.patience:
            print("  early stop: no improvement for %d epochs" % a.patience)
            break
    net.load_state_dict(best[1])
    model = model.cpu().eval()

    # Export: FP16 is the deployed file, FP32 the sibling.
    out16 = os.path.join(a.out, "pixel_track_classifier_FP16_%s.pt" % date)
    out32 = os.path.join(a.out, "pixel_track_classifier_FP32_%s.pt" % date)
    torch.jit.script(copy.deepcopy(model).half()).save(out16)
    torch.jit.script(copy.deepcopy(model).float()).save(out32)
    print("exported %s and %s" % (out16, out32))

    # Working points on the test split, scored like the producer: the exported FP16 file and the
    # deployed file, both in half precision.
    sdev = a.device if torch.cuda.is_available() else "cpu"
    new = load_reference(out16)
    ref = load_reference(a.reference)
    rc, nc = submodule_codes(ref), submodule_codes(new)
    same_code = all(rc[k] == nc.get(k) for k in rc)
    print("TorchScript code of the export against the deployed file: %s"
          % ("identical in every submodule" if same_code else "DIFFERS"))
    Xte = X[te]
    s_new = score(new, Xte, sdev)
    s_ref = score(ref, Xte, sdev)
    A, C = A_all[te], C_all[te]
    vals = {"pt": Xte[:, FEATS17.index("pt")].astype(np.float64),
            "eta": np.abs(Xte[:, FEATS17.index("eta")].astype(np.float64)),
            "dxy": np.abs(d["dxyBS"][te].astype(np.float64))}
    edges = {"pt": tmf.parse_edges(tmf.PT_BINS_DEFAULT, "pt"),
             "eta": tmf.parse_edges(tmf.ETA_BINS_DEFAULT, "eta"),
             "dxy": tmf.parse_edges(tmf.DXY_BINS_DEFAULT, "dxy")}
    bins = {"pt": tmf.make_bins(edges["pt"], True), "eta": tmf.make_bins(edges["eta"], False),
            "dxy": tmf.make_bins(edges["dxy"], False)}

    def tables(s, t):
        return {ax: tmf.bin_table(bins[ax], vals[ax], A, C, s, t) for ax in ("pt", "eta", "dxy")}

    wp_name = "global_%g" % a.recall
    wp_thr = tmf.thr_at_recall(s_new, A, a.recall)

    D_te = dup_all[te]

    def summary(s, t):
        return {"threshold": float(t), "recall": tmf.recall_at_thr(s, A, t),
                "rejection": tmf.rej_at_thr(s, C, t),
                "duplicate_rejection": tmf.rej_at_thr(s, D_te, t), "tables": tables(s, t)}

    res = {"date": date, "reference": os.path.abspath(a.reference),
           "reference_threshold": a.reference_threshold,
           "caches": [os.path.abspath(p) for p in a.cache], "rows": int(len(X)),
           "split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "recipe": {"lr": a.lr, "batch": a.batch, "epochs_max": a.epochs, "patience": a.patience,
                      "seed": a.seed, "best_epoch": best[2], "recall": a.recall,
                      "val_rej_at_recall": best[0], "pos_weight": pos_w,
                      "pos_scale": a.pos_scale, "eff_weight": a.eff_weight,
                      "region_pt_weight": a.region_pt_weight, "dup_as_fake": a.dup_as_fake,
                      "recall_axis": "A_nodup", "target": "y_mtv",
                      "clip_quantiles": CLIP_QUANTILES},
           "preprocessing": {"features": FEATS17, "do_log": do_log.tolist(),
                             "clip_min": cmin.tolist(), "clip_max": cmax.tolist(),
                             "mean": mean.tolist(), "std": std.tolist()},
           "export": {"fp16": out16, "fp32": out32, "code_identical_to_reference": same_code},
           "reference_at_its_threshold": summary(s_ref, a.reference_threshold),
           "reference_at_same_recall": summary(s_ref, tmf.thr_at_recall(s_ref, A, a.recall)),
           "working_points": {wp_name: summary(s_new, wp_thr)}}
    print("\nTest split: %d rows, %d efficiency tracks, %d fakes" % (len(Xte), A.sum(), C.sum()))
    print("%-22s %10s %9s %9s   pT bins: recall / rejection" % ("model", "threshold", "recall",
                                                                "rejection"))

    def line(name, sm):
        pt = " ".join("%s:%.4f/%.3f" % (lab, v["recall"], v["rej"])
                      for lab, v in sm["tables"]["pt"].items())
        print("%-22s %10.5f %9.5f %9.5f   %s" % (name, sm["threshold"], sm["recall"],
                                                 sm["rejection"], pt))
    line("reference file", res["reference_at_its_threshold"])
    line("reference, same recall", res["reference_at_same_recall"])
    line("new " + wp_name, res["working_points"][wp_name])
    res["working_point"] = dict(res["working_points"][wp_name], name=wp_name)
    print("\nWORKING POINT (%s): scoreThreshold = %.6f" % (wp_name, wp_thr))
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(a.out, "thrmap.txt"), "w") as f:
        f.write(thrmap(s_new, A, C))
    print("wrote %s/result.json and thrmap.txt" % a.out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "train"])
    ap.add_argument("--reference", required=True,
                    help="the deployed TorchScript file (pixel_track_classifier_FP16.pt)")
    ap.add_argument("--reference-threshold", type=float, default=None,
                    help="train: the scoreThreshold the deployed file runs at")
    ap.add_argument("--cache", nargs="+", default=None,
                    help="17-feature cache(s): nano_loader.py cache-arm ... --feats 17")
    ap.add_argument("--out", default=None, help="train: output directory")
    ap.add_argument("--date", default=None, help="date in the file name (default today)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--recall", type=float, default=0.995,
                    help="recall of the non-duplicate efficiency tracks at the working point and "
                         "in the early-stopping metric")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pos-scale", type=float, default=4.0, dest="pos_scale", metavar="K",
                    help="every true track x K after the class balance")
    ap.add_argument("--region-pt-weight", type=float, default=4.0, dest="region_pt_weight",
                    metavar="A", help="every track x (1 + A log10(max(pT, 0.9) / 0.9))")
    ap.add_argument("--eff-weight", type=float, default=1.0, dest="eff_weight", metavar="A",
                    help="non-duplicate efficiency tracks x (1 + A log10(max(pT, 0.9) / 0.9))")
    ap.add_argument("--dup-as-fake", type=float, default=1.0, dest="dup_as_fake", metavar="W",
                    help="duplicates are negatives of the loss with weight W; 0 = they stay "
                         "positives")
    ap.add_argument("--max-rows", type=int, default=200000, dest="max_rows",
                    help="check: rows scored")
    ap.add_argument("--verbose", action="store_true", help="check: print differing code")
    a = ap.parse_args()
    if a.mode == "check":
        return cmd_check(a)
    if not a.cache or not a.out or a.reference_threshold is None:
        ap.error("train needs --cache, --out and --reference-threshold")
    return cmd_train(a)


if __name__ == "__main__":
    sys.exit(main())
