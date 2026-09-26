#!/usr/bin/env python3
"""Train the high-purity forest of one pixel-track collection and export it in the compact
binary PixelTrackForestHighPuritySelector loads.

One code path serves the three collections:

  --cache <prompt cache>   nano_loader.py cache-arm --prefix TrkPrompt      the prompt selector
  --cache <disp cache>     nano_loader.py cache-arm --prefix TrkDisp        the displaced selector
  --cache <merged cache>   nano_loader.py cache-merged                      the merged selector,
                           which scores the output of hltPhase2PixelTracksSoAMerger

INPUT. Every cache carries the 31 features of PixelTrackFeaturesSoA columns 1-31 as a prefix, in
that order, and two labels: y_mtv (matched to any TrackingParticle, the training target) and
y_eff (matched to a TrackingParticle passing the efficiency selection, the recall axis). A merged
cache also carries the four provenance columns iteration, ndof, nOTExtra, nAttached and, when the
dataset was produced with NANO_CLUSTER=1, the pixel-cluster columns. Several caches are
concatenated; the event-level split is taken from evt % 10, so caches whose event numbering
restarts still keep every event inside one split.

CLASSES. The rows are grouped by the two labels:
  A = y_eff == 1                 real, and matched to an efficiency-selection TrackingParticle.
                                 The recall axis: a working point's recall is quoted on A only.
  C = y_mtv == 0                 matched to no TrackingParticle at all, what the validation calls
                                 a fake. The rejection axis.
  B = y_mtv == 1 and y_eff == 0  real, but not an efficiency-selection TrackingParticle (low pT,
                                 secondary, large impact parameter, out-of-time pile-up). On
                                 neither axis, but present in training as a positive.
The training target is y_mtv (A and B positive, C negative). Training on y_eff instead teaches
the model to reject class B; --label legacy does that, for a comparison with an older model.

FEATURE VECTOR. The .bin stores a split-feature index per node, which is a position in the vector
the selector kernel assembles; there is no name lookup at load time and no header that would
catch a mismatch, so the training matrix must be in the selector's column order.
  --feats 31              columns 0..30, the per-iteration selectors' input (the cache prefix, no
                          permutation needed)
  --feats 35              adds the provenance block as columns 31..34, permuted into the order
                          named by --abi-order (the cache order is iteration, ndof, nOTExtra,
                          nAttached; the selector's is nAttached, nOTExtra, iteration, ndof)
  --extra-cols c1,c2,...  appends further cache columns after the ABI block, in the order given:
                          the merged selector reads the seven pixel-cluster columns this way,
                          for a 42-wide vector
The column order used is recorded in result.json (feats_abi).

RECIPE (the defaults). Gradient-boosted trees, depth 12, up to 450 trees, learning rate 0.05,
min_child_weight 20, reg_lambda 4, subsample 0.8, colsample_bytree 0.8, tree_method hist.
Positives are weighted to balance the classes; --dxy-alpha, --eff-weight and --fake-region-weight
add per-row training weights on top (they multiply). Two passes: after the first fit, positives
scoring below --tail-band times the pass-1 threshold at --recall get their weight multiplied by
--tailw, and the model is fitted again. Early stopping (--es-rounds) minimises 1 - (class-C
rejection at the run's working point) on the validation split, so it optimises the model at the
threshold it will be deployed at. Event-level split evt % 10: train >= 4, validation == 3, test
< 3. The booster is sliced to the early-stopping optimum before export.

WORKING POINT (--wp-rule). The rule that turns scores into a threshold; it is the run's own
("wp") working point and drives early stopping:
  uniform   (default) the largest threshold at which class-A recall is at least --recall in
            every pT, |eta| and |dxyBS| bin of the --pt-bins / --eta-bins / --dxy-bins edges.
            Bins with fewer than --pt-bin-min-a class-A rows do not constrain it; if no bin is
            usable the global rule applies. It depends on no reference model and protects the
            sparse tails (high pT, forward, displaced) that a global quantile lets the populous
            bins pay for.
  global    the threshold at which the overall class-A recall is --recall.
  profile   the threshold at which every bin reaches the recall a reference model has in that
            bin at its own threshold, plus --wp-margin, capped at --wp-target-cap: a retrain
            that may not lose efficiency in any bin against that model. The profile is written
            by make_profile.py and passed with --wp-profile (which selects this rule).
The threshold printed and written to result.json is an offline starting point: the deployed value
is chosen by re-scanning the full reconstruction and the track validation.

OUTPUTS in --out:
  model.json                        booster sliced to the early-stopping optimum (the exported one)
  model_full.json                   booster as fitted, all --ntrees trees
  <arm>_tree<nf>_<tag>_<date>.bin   compact binary for the selector
  result.json                       metrics, the working point, the feature list and the recipe
  thrmap.txt                        score threshold against class-A recall, with class-C rejection

After the export the .bin is read back with read_compact_tree.py and re-scored on test rows
against xgboost. With --no-fp16 the traversal must reproduce xgboost to 1e-5. With the default
fp16 rounding the split thresholds are rounded too, so a row sitting between the fp32 and the fp16
threshold takes the other branch: that gate is on the median difference and on how often a row
moves, never on the maximum.

Usage (in the CMSSW environment, whose xgboost is the one the deployed models were trained with):
  python3 train_merged_forest.py --cache <cache.npz> --out <dir> [options]
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_compact_tree  # noqa: E402
import read_compact_tree  # noqa: E402

# Layout constants; must match nano_loader.py (FEATS31 + CA_PROV).
FEATS31 = [
    "chi2", "dzError", "dxyError", "eta", "nHits", "phi", "phiError", "pt", "qOverPtError",
    "dzBS", "dxyBS", "nLayers", "cotThetaError", "covCotThetaDz", "covDxyQOverPt", "covPhiDxy",
    "covPhiQOverPt", "fitChi2", "psFrac", "r0", "nPS", "spanZ", "nStubs", "logChi2Stub", "kErr",
    "dcaEst", "nBarrel", "rzChi2", "meanStubKappa", "leverArm", "rMax",
]
CA_PROV = ["iteration", "ndof", "nOTExtra", "nAttached"]
PROV_ALIAS = {"iterationid": "iteration", "iteration": "iteration", "ndof": "ndof",
              "notextra": "nOTExtra", "notextras": "nOTExtra", "notextrahits": "nOTExtra",
              "nattached": "nAttached"}
# pixelTrack::Iteration
ITER_NAME = {0: "promptHighPt", 1: "promptLowPt", 2: "displaced", 3: "notIteration", -1: "invalid"}
ITER_GROUP = {0: "prompt", 1: "prompt", 2: "displaced", 3: "other", -1: "other"}
DXY_BINS = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, float("inf"))]

# Per-bin recall axes: the track validation quotes efficiency per pT, |eta| and vertex-position
# bin, so thresholds are constrained bin by bin rather than globally. Class-A recall is the offline
# proxy for efficiency and |dxyBS| stands in for the vertex position.
PT_BINS_DEFAULT = "0.9,1.5,3,10,30,1000"
ETA_BINS_DEFAULT = "0,1.5,2.5,4"
DXY_BINS_DEFAULT = "0,0.1,0.5,1,2,5,100"
WP_AXES = ("pt", "eta", "dxy")


# Metric primitives: recall on class A, rejection on class C.
def thr_at_recall(s, A, rc):
    """Score threshold retaining `rc` of class A: an index into the sorted class-A scores, the
    same convention as nano_train_utils.thr_at_recall."""
    r = s[A]
    if len(r) == 0:
        return float("nan")
    i = min(int((1.0 - rc) * len(r)), len(r) - 1)
    # np.partition gives the same value as np.sort(r)[i] in O(n): early stopping re-derives one
    # threshold per bin per round.
    return float(np.partition(r, i)[i])


def rej_at_thr(s, C, t):
    """Fraction of class C rejected (scored below t)."""
    if C.sum() == 0 or not np.isfinite(t):
        return float("nan")
    return float(1.0 - (s[C] >= t).mean())


def recall_at_thr(s, A, t):
    if A.sum() == 0 or not np.isfinite(t):
        return float("nan")
    return float((s[A] >= t).mean())


# Precision/recall companions to the AUCs, reported only: on a test split that is mostly positive a
# ROC-AUC of 0.99 leaves the fake contamination unquoted, while precision quotes it. Nothing selects,
# stops or thresholds on them.
_TRAPZ = getattr(np, "trapezoid", np.trapz)


def partial_prauc(y_pos, s, rmin=0.99):
    """Area under the precision-recall curve restricted to recall >= `rmin`, normalised by
    (1 - rmin) so it lands in [0, 1]. sklearn's precision_recall_curve returns recall decreasing
    with a trailing (precision=1, recall=0) point, so the arrays are re-sorted to increasing
    recall, the precision at exactly `rmin` is interpolated in, and the integral is a trapezoid
    rule over recall."""
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
    """Precision on the A-vs-C subset at threshold `t`: class-A rows kept / all rows kept. At the
    working point this is 1 - (fake contamination of the accepted sample)."""
    if not np.isfinite(t):
        return None
    nA = int((s[A] >= t).sum())
    kept = nA + int((s[C] >= t).sum())
    return float(nA / kept) if kept else None


# Binned metric primitives: the per-bin recall constraint.
def parse_edges(spec, what):
    """Comma-separated, strictly increasing bin edges."""
    try:
        e = [float(v) for v in str(spec).split(",") if v.strip() != ""]
    except ValueError:
        raise SystemExit("%s: not a comma-separated list of numbers: %r" % (what, spec))
    if len(e) < 2:
        raise SystemExit("%s needs at least two edges, got %r" % (what, spec))
    if any(e[i + 1] <= e[i] for i in range(len(e) - 1)):
        raise SystemExit("%s edges must be strictly increasing, got %r" % (what, spec))
    return e


def make_bins(edges, underflow):
    """[(label, lo, hi)] covering the whole axis, so every row lands in exactly one bin and no
    class-A track can escape the per-bin constraint by sitting off the end of the edge list.

    underflow=True  (pT):  an explicit [-inf, edges[0]) bin -- reconstructed pT can fall below the
                           efficiency-selection cut, and those tracks are still class A.
    underflow=False (|eta|, |dxyBS|): the axis starts at edges[0] = 0.
    The top edge is always opened to +inf: an edge list ending in 1000 (pT) or 100 (|dxyBS|) means
    'and everything above', never 'drop what is above'."""
    out = []
    if underflow:
        out.append(("lt%g" % edges[0], float("-inf"), edges[0]))
    for i in range(len(edges) - 2):
        out.append(("%g-%g" % (edges[i], edges[i + 1]), edges[i], edges[i + 1]))
    out.append(("%g-inf" % edges[-2], edges[-2], float("inf")))
    return out


def bin_masks(bins, v):
    return [(lab, (v >= lo) & (v < hi)) for lab, lo, hi in bins]


def axis_masks(entry):
    """An `axes` entry is (name, values, bins) or (name, values, bins, precomputed masks). The
    early-stopping metric re-derives the working point every round on fixed rows, so the masks are
    built once and carried along rather than recomputed every round."""
    if len(entry) > 3 and entry[3] is not None:
        return entry[3]
    return bin_masks(entry[2], entry[1])


def thr_at_recall_binned(yA, s, pt, edges, R, min_n=100, underflow=True):
    """The threshold at which every pT bin has class-A recall >= R: the minimum over bins of the
    per-bin thr_at_recall. A global quantile lets a populous bin pay for a sparse one, which is
    how efficiency gets redistributed in pT. Bins with fewer than `min_n` class-A rows are skipped
    (their quantile is one track wide); with no usable bin it falls back to the global quantile."""
    ts = []
    for _, m in bin_masks(make_bins(edges, underflow), pt):
        mA = yA & m
        if int(mA.sum()) < min_n:
            continue
        t = thr_at_recall(s, mA, R)
        if np.isfinite(t):
            ts.append(t)
    if not ts:
        return thr_at_recall(s, yA, R)
    return float(min(ts))


def thr_at_profile(yA, s, axes, profile, margin, cap, min_n=100, fallback_recall=None):
    """The threshold at which every bin of every axis reaches the target class-A recall of that
    bin, plus `margin`. `axes` is [(name, values, bins)]; `profile` is {axis: {label: recall}},
    either a reference model's own per-bin recall (the match-or-exceed rule) or the same number
    in every bin (the uniform rule). Per-bin targets are capped at `cap` so one unlucky track in
    a sparse bin cannot drag the whole working point to the floor. Bins with fewer than `min_n`
    class-A rows do not constrain it. With no usable bin at all the global quantile at
    `fallback_recall` is returned when one is given, otherwise the run stops.
    Returns (threshold, per-bin detail)."""
    detail, ts = {}, []
    for entry in axes:
        name = entry[0]
        prof = profile.get(name, {})
        for lab, m in axis_masks(entry):
            mA = yA & m
            nA = int(mA.sum())
            ref = prof.get(lab)
            if ref is None or not np.isfinite(ref) or nA < min_n:
                detail["%s:%s" % (name, lab)] = {"nA": nA, "ref_recall": ref, "used": False}
                continue
            tgt = min(float(ref) + margin, cap)
            t = thr_at_recall(s, mA, tgt)
            detail["%s:%s" % (name, lab)] = {"nA": nA, "ref_recall": float(ref),
                                             "target": tgt, "thr": t, "used": True}
            if np.isfinite(t):
                ts.append(t)
    if not ts:
        if fallback_recall is not None:
            return thr_at_recall(s, yA, fallback_recall), detail
        raise SystemExit("--wp-profile matched no usable bin -- check that its edges match this "
                         "run's --pt-bins / --eta-bins / --dxy-bins")
    return float(min(ts)), detail


def thr_at_recall_axes(yA, s, axes, R, min_n=100):
    """The threshold at which every bin of every axis reaches class-A recall R -- the uniform
    counterpart of thr_at_profile, reported as a comparison working point."""
    ts = []
    for entry in axes:
        for _, m in axis_masks(entry):
            mA = yA & m
            if int(mA.sum()) < min_n:
                continue
            t = thr_at_recall(s, mA, R)
            if np.isfinite(t):
                ts.append(t)
    return float(min(ts)) if ts else thr_at_recall(s, yA, R)


def bin_table(bins, v, A, C, s, thr):
    """Class-A recall and class-C rejection per bin at one threshold."""
    out = {}
    for lab, m in bin_masks(bins, v):
        out[lab] = {"nA": int((m & A).sum()), "nC": int((m & C).sum()),
                    "recall": recall_at_thr(s, m & A, thr), "rej": rej_at_thr(s, m & C, thr)}
    return out


def min_bin_recall(tables, min_n=100):
    """Worst per-bin class-A recall over every axis, ignoring bins too sparse to measure."""
    worst, where = float("inf"), None
    for axis, tab in tables.items():
        for lab, v in tab.items():
            if v["nA"] < min_n or not np.isfinite(v["recall"]):
                continue
            if v["recall"] < worst:
                worst, where = v["recall"], "%s:%s" % (axis, lab)
    return (None, None) if where is None else (worst, where)


# Region re-weighting of class C in the training loss, never in the metric: --fake-region-weight
# lets the loss lean on the regions where the residual fakes live without touching the rejection
# metric, the recall axis or the working-point rule.
#
# Comma-separated terms <column>:<lo>:<hi>:<weight>, e.g. "abseta:1.2:1.8:3,absdxy:1:inf:3".
# Ranges are half-open [lo, hi) and accept "inf"/"-inf"; terms multiply where they overlap. Only
# class-C rows are touched, so this cannot trade away recall.
# Columns, all read from the cache: abseta|eta -> abs(feature "eta"); absdxy|dxy -> abs(cache
# column "dxyBS"), the array the --dxy-bins recall axis uses; pt -> feature "pt", signed.
FAKE_REGION_COLS = {"abseta": "abseta", "eta": "abseta", "absdxy": "absdxy", "dxy": "absdxy",
                    "pt": "pt"}


def parse_fake_region_weight(spec):
    """[(column, lo, hi, weight)] from a --fake-region-weight SPEC string."""
    terms = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 4:
            raise SystemExit("--fake-region-weight term %r is not <column>:<lo>:<hi>:<weight> "
                             "(e.g. abseta:1.2:1.8:3)" % part)
        col = FAKE_REGION_COLS.get(bits[0].strip().lower())
        if col is None:
            raise SystemExit("--fake-region-weight: unknown column %r (known: %s)"
                             % (bits[0], ", ".join(sorted(set(FAKE_REGION_COLS)))))
        try:
            lo, hi, wv = float(bits[1]), float(bits[2]), float(bits[3])
        except ValueError:
            raise SystemExit("--fake-region-weight term %r: lo/hi/weight must be numbers "
                             "('inf' is allowed for hi)" % part)
        if not (hi > lo):
            raise SystemExit("--fake-region-weight term %r: needs hi > lo" % part)
        if not (wv > 0):
            raise SystemExit("--fake-region-weight term %r: weight must be > 0" % part)
        terms.append((col, lo, hi, wv))
    if not terms:
        raise SystemExit("--fake-region-weight parsed to no terms: %r" % spec)
    return terms


def fake_region_weights(terms, vals, C_mask):
    """Multiplicative per-row weight (1.0 outside every region, and 1.0 on every non-C row)."""
    fw = np.ones(len(C_mask), np.float64)
    detail = []
    for col, lo, hi, wv in terms:
        v = vals[col]
        m = (v >= lo) & (v < hi) & C_mask
        fw[m] *= wv
        detail.append({"column": col, "lo": lo, "hi": hi, "weight": wv,
                       "n_C_rows": int(m.sum()),
                       "frac_of_C": float(m.sum() / max(1, int(C_mask.sum())))})
    return fw, detail


# inputs
def load_caches(paths, want_label="mtv"):
    """Concatenate caches written by nano_loader.py cache-merged / cache-arm.

    The y_is guard: every such cache records which array `y` is. A cache built with
    `--label legacy` has y = y_eff, the narrow efficiency label, under which real tracks that
    fail an efficiency cut are negatives. Training on that silently produces a model that rejects
    real tracks, and nothing in the resulting .bin or result.json would say so. So the target is
    taken from the named array (y_mtv / y_eff), never from `y`, and a cache whose y_is contradicts
    --label is refused rather than quietly reinterpreted."""
    Xs, ymtv, yeff, dup, dxy, evt = [], [], [], [], [], []
    feats = None
    per_file = []
    meta = {"prefix": set(), "min_dxy": set(), "y_is": set()}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        f = [str(v) for v in z["feats"]]
        if feats is None:
            feats = f
        elif f != feats:
            raise SystemExit("cache column mismatch:\n  %s\n  has %s\n  expected %s" % (p, f, feats))
        for key in ("X", "y_mtv", "y_eff", "evt"):
            if key not in z:
                raise SystemExit("%s has no '%s' -- it is not a cache-merged / cache-arm npz. "
                                 "Rebuild it with nano_loader.py cache-merged (merged row space) "
                                 "or cache-arm --prefix TrkPrompt|TrkDisp (per-arm)." % (p, key))
        if "y_is" not in z:
            raise SystemExit(
                "%s carries no 'y_is' metadata, so which label it was built against cannot be "
                "established. That is exactly the silent narrow-label fallback this check exists "
                "to stop. Rebuild it with the current nano_loader.py." % p)
        y_is = str(z["y_is"])
        want = "y_mtv" if want_label == "mtv" else "y_eff"
        if y_is != want:
            raise SystemExit(
                "%s was built with y = %s but --label %s asks for y = %s. The named arrays y_mtv / "
                "y_eff are both present, so this is only a mismatch of intent -- rebuild the cache "
                "with 'nano_loader.py ... --label %s', or pass the matching --label."
                % (p, y_is, want_label, want, want_label))
        meta["y_is"].add(y_is)
        meta["prefix"].add(str(z["prefix"]) if "prefix" in z else "unknown")
        meta["min_dxy"].add(float(z["min_dxy"]) if "min_dxy" in z else -1.0)
        Xs.append(z["X"].astype(np.float32))
        ymtv.append(z["y_mtv"].astype(np.float32))
        yeff.append(z["y_eff"].astype(np.float32))
        dup.append(z["duplicate"].astype(np.float32) if "duplicate" in z
                   else np.zeros(len(z["X"]), np.float32))
        dxy.append(z["dxyBS"].astype(np.float32))
        evt.append(z["evt"].astype(np.int64))
        per_file.append({"path": os.path.abspath(p), "rows": int(len(z["X"]))})
    if len(meta["prefix"]) > 1 or len(meta["min_dxy"]) > 1:
        raise SystemExit("the caches disagree about the row space they came from: prefix=%s "
                         "min_dxy=%s. Mixing row spaces in one forest is never intended."
                         % (sorted(meta["prefix"]), sorted(meta["min_dxy"])))
    out = dict(X=np.concatenate(Xs), y_mtv=np.concatenate(ymtv), y_eff=np.concatenate(yeff),
               duplicate=np.concatenate(dup), dxyBS=np.concatenate(dxy), evt=np.concatenate(evt),
               feats=feats, per_file=per_file,
               prefix=sorted(meta["prefix"])[0], min_dxy=sorted(meta["min_dxy"])[0],
               y_is=sorted(meta["y_is"])[0])
    return out


def build_abi_matrix(X, feats, nfeat, abi_order_str, extra_cols=None):
    """Return (Xabi, names, prov_by_name). Columns 0..30 = the 31 deployed features (the cache
    prefix), columns 31..34 = the provenance block permuted into `abi_order_str`, then any
    --extra-cols appended verbatim, in the order given, after the ABI block."""
    if feats[:31] != FEATS31:
        raise SystemExit("cache does not start with the 31 deployed features in FEATSEL order.\n"
                         "  got     : %s\n  expected: %s" % (feats[:31], FEATS31))
    prov_by_name = {}
    for i, n in enumerate(feats[31:], start=31):
        prov_by_name[n] = X[:, i]
    extra = [e.strip() for e in (extra_cols or []) if e.strip()]
    miss = [e for e in extra if e not in prov_by_name]
    if miss:
        raise SystemExit("--extra-cols: the cache has no column(s) %s (it has %s)"
                         % (miss, sorted(prov_by_name)))
    dup_e = [e for e in extra if extra.count(e) > 1]
    if dup_e:
        raise SystemExit("--extra-cols lists %s more than once" % sorted(set(dup_e)))

    def _append_extra(Xb, nb):
        if not extra:
            return Xb, nb
        cols = [Xb] + [prov_by_name[e].reshape(-1, 1) for e in extra]
        return np.concatenate(cols, axis=1).astype(np.float32), nb + extra

    if nfeat == 31:
        return _append_extra(X[:, :31].copy(), list(FEATS31)) + (prov_by_name,)

    want = [w.strip() for w in abi_order_str.split(",") if w.strip()]
    canon = []
    for w in want:
        k = PROV_ALIAS.get(w.lower())
        if k is None:
            raise SystemExit("--abi-order: unknown provenance column %r (known: %s)"
                             % (w, ", ".join(CA_PROV)))
        canon.append(k)
    if sorted(canon) != sorted(CA_PROV):
        raise SystemExit("--abi-order must be a permutation of %s, got %s" % (CA_PROV, canon))
    missing = [c for c in canon if c not in prov_by_name]
    if missing:
        raise SystemExit(
            "--feats 35 asks for the provenance columns %s but the cache has %s. A per-arm cache "
            "(cache-arm) normally carries none of them: `iteration` is only meaningful on the "
            "merged collection, where the two arms become indistinguishable. Train the per-arm "
            "forests with --feats 31; --feats 35 is for the merged cache."
            % (missing, list(prov_by_name) or "none"))
    cols = [X[:, :31]] + [prov_by_name[c].reshape(-1, 1) for c in canon]
    Xb = np.concatenate(cols, axis=1).astype(np.float32)
    Xb, nb = _append_extra(Xb, list(FEATS31) + canon)
    return Xb, nb, prov_by_name


def make_split(evt, mode, valfrac, testfrac, seed):
    """train / val / test boolean masks. evt10 = the event-level split."""
    if mode == "evt10":
        m = evt % 10
        return (m >= 4), (m == 3), (m < 3)
    if mode == "rows":
        # deterministic row-level fallback for caches with too few distinct events for evt10
        rng = np.random.default_rng(seed)
        u = rng.random(len(evt))
        te = u < testfrac
        va = (u >= testfrac) & (u < testfrac + valfrac)
        tr = u >= testfrac + valfrac
        return tr, va, te
    raise SystemExit("unknown --split %r" % mode)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", required=True,
                    help="one or more nano_loader.py cache-arm / cache-merged .npz (concatenated)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--feats", type=int, choices=(31, 35), default=31,
                    help="31 = the per-iteration selectors' input; 35 = + the provenance block in "
                         "--abi-order (the merged selector, together with --extra-cols). "
                         "Default 31.")
    ap.add_argument("--label", choices=("mtv", "legacy"), default="mtv",
                    help="training target: mtv = y_mtv (matched to any TrackingParticle, "
                         "default); legacy = the narrow efficiency-selected label")
    ap.add_argument("--recall", type=float, default=0.995,
                    help="target class-A recall for early stopping and the working point")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--ntrees", type=int, default=450)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--mcw", type=float, default=20.0, help="min_child_weight")
    ap.add_argument("--lam", type=float, default=4.0, help="reg_lambda")
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8, help="colsample_bytree")
    ap.add_argument("--tailw", type=float, default=4.0,
                    help="two-pass tail re-weight factor on positives; 0 disables the second pass")
    ap.add_argument("--tail-band", type=float, default=4.0,
                    help="tail band in units of pass-1 thr@recall (positives with s0 < band*t0)")
    ap.add_argument("--prune-tol", type=float, default=0.0, dest="prune_tol",
                    help="after the fit, keep the smallest number of trees whose working-point "
                         "fake rejection on the validation split is within this tolerance of the "
                         "best prefix (a size chosen by the data instead of a fixed budget); 0 = off")
    ap.add_argument("--from-full", default=None, dest="from_full", metavar="model_full.json",
                    help="skip the fit and start from a saved full booster (then prune/export)")
    ap.add_argument("--es-rounds", type=int, default=60,
                    help="early_stopping_rounds; 0 disables early stopping and keeps all --ntrees trees")
    ap.add_argument("--split", choices=("evt10", "rows"), default="evt10",
                    help="evt10 = event-level evt%%10 (train m>=4, val m==3, test m<3); "
                         "rows = deterministic row-level fallback for tiny caches")
    ap.add_argument("--valfrac", type=float, default=0.1, help="--split rows only")
    ap.add_argument("--testfrac", type=float, default=0.3, help="--split rows only")
    ap.add_argument("--extra-cols", default="", dest="extra_cols", metavar="c1,c2,...",
                    help="extra cache columns appended after the ABI block, in the order given "
                         "(the merged selector reads the NANO_CLUSTER=1 pixel-cluster block "
                         "minCharge,meanCharge,minChargeNorm,maxSizeY,meanSizeY,maxSizeX,"
                         "nLowCharge this way). A .bin trained with them needs a selector fed "
                         "those columns in exactly this order -- there is no name lookup in the "
                         "compact reader. Recorded in result.json as `extra_cols`.")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="wp")
    ap.add_argument("--arm", choices=("prompt", "disp", "merged"), default=None,
                    help="names the .bin (<arm>_tree<nf>_<tag>_<date>.bin). Default: taken from "
                         "the cache's own `prefix` metadata (TrkPrompt->prompt, TrkDisp->disp, "
                         "TrkMerged->merged), so it cannot disagree with the rows.")
    ap.add_argument("--dxy-alpha", type=float, default=0.0, dest="dxy_alpha",
                    help="displacement re-weighting of the positives, w *= 1 + alpha*min(|dxyBS|,"
                         "60)/20, the same rule the displaced track DNN uses. 0 (default) = off. "
                         "It changes only the training weights: every metric axis stays "
                         "unweighted, so the recall and rejection numbers remain comparable "
                         "across runs.")
    # the working-point rule and its bins
    ap.add_argument("--wp-rule", choices=("uniform", "global", "profile"), default=None,
                    dest="wp_rule",
                    help="how the working point (and the early-stopping objective) is derived: "
                         "uniform = class-A recall >= --recall in every pT/|eta|/|dxyBS| bin "
                         "(default); global = overall class-A recall = --recall; profile = "
                         "match or exceed a reference model's per-bin recall (needs "
                         "--wp-profile, which also selects this rule on its own)")
    ap.add_argument("--pt-bins", default=PT_BINS_DEFAULT, dest="pt_bins",
                    help="pT reporting/constraint edges (default %s); an explicit underflow bin "
                         "below the first edge is always added" % PT_BINS_DEFAULT)
    ap.add_argument("--eta-bins", default=ETA_BINS_DEFAULT, dest="eta_bins",
                    help="|eta| reporting/constraint edges (default %s)" % ETA_BINS_DEFAULT)
    ap.add_argument("--dxy-bins", default=DXY_BINS_DEFAULT, dest="dxy_bins",
                    help="|dxyBS| (vertex-position proxy) reporting/constraint edges (default %s)"
                         % DXY_BINS_DEFAULT)
    ap.add_argument("--pt-bin-min-a", type=int, default=100, dest="pt_bin_min_a",
                    help="a bin with fewer class-A rows than this is reported but never allowed "
                         "to set the working point -- its recall quantile is one track wide")
    ap.add_argument("--wp-profile", default=None, dest="wp_profile", metavar="JSON",
                    help="the profile of --wp-rule profile: a file written by make_profile.py "
                         "holding a reference model's own class-A recall in every "
                         "pT/|eta|/|dxyBS| bin at its threshold. The working point becomes the "
                         "threshold at which every bin reaches that recall + --wp-margin, so no "
                         "bin can regress against the reference. Giving it selects the rule.")
    ap.add_argument("--wp-margin", type=float, default=0.001, dest="wp_margin",
                    help="profile rule only: headroom added to every reference per-bin recall "
                         "(default 0.001; pass 0 for an exact match-or-exceed)")
    ap.add_argument("--wp-target-cap", type=float, default=0.9990, dest="wp_target_cap",
                    help="per-bin recall targets are capped here (default 0.9990). A reference bin "
                         "sitting at recall 1.0 would otherwise demand the single lowest-scoring "
                         "class-A track in that bin and collapse the whole working point.")
    ap.add_argument("--eff-weight", type=float, default=0.0, dest="eff_weight", metavar="A",
                    help="class-A positive re-weighting w(pt) = 1 + A*log10(max(pt,0.9)/0.9), so "
                         "the loss stops trading away the rare high-pT reals for the abundant "
                         "low-pT ones. Composes with --dxy-alpha and the two-pass tail re-weight "
                         "(they multiply). Training weights only: every metric axis stays "
                         "unweighted. 0 (default) = off.")
    ap.add_argument("--recall-axis", choices=("A", "A_nodup"), default="A", dest="recall_axis",
                    help="which rows the recall is measured on -- for the working point, for the "
                         "per-bin / --wp-profile rule and for the early-stopping metric. "
                         "A (default) = every class-A row. A_nodup = class-A rows with "
                         "duplicate==0: the validation counts TrackingParticles found, not tracks "
                         "kept, so removing a duplicate costs no efficiency. Rejection stays on "
                         "class C either way, and both recalls are reported at the working point.")
    ap.add_argument("--fake-region-weight", default=None, dest="fake_region_weight", metavar="SPEC",
                    help="up-weight class-C rows in chosen regions in the training loss only "
                         "(never in the metric, the recall axis or the working-point rule). SPEC "
                         "is comma-separated <column>:<lo>:<hi>:<weight>, half-open [lo,hi), "
                         "'inf' allowed, terms multiply where they overlap; columns abseta|eta "
                         "(abs of feature 'eta'), absdxy|dxy (abs of cache column 'dxyBS'), pt. "
                         "Example: \"abseta:1.2:1.8:3,absdxy:1:inf:3\". Multiplies with the "
                         "existing class weight, --dxy-alpha, --eff-weight and the tail re-weight.")
    ap.add_argument("--dup-as-fake", nargs="?", type=float, const=1.0, default=None,
                    dest="dup_as_fake", metavar="W",
                    help="treat duplicate==1 rows (real, but a second reconstruction of a "
                         "TrackingParticle another track already found) as negatives in the "
                         "training target, with weight W (default 1.0). They are removed from the "
                         "recall axis automatically (so --recall-axis A_nodup is implied) and they "
                         "are not added to class C, so every rejection number stays on exactly "
                         "the same rows and remains comparable across runs. The duplicate "
                         "rejection at the working point is reported for every run either way.")
    ap.add_argument("--date", default=None, help="YYYYMMDD stamp in the .bin name (default: today)")
    ap.add_argument("--abi-order", default="nAttached,nOTExtra,iteration,ndof",
                    help="order of the 4 provenance columns in the selector's feature vector "
                         "(cache order is iteration,ndof,nOTExtra,nAttached)")
    ap.add_argument("--no-fp16", action="store_true",
                    help="keep the split and leaf values in fp32 (the prompt and merged files); "
                         "by default they are rounded to fp16 and stored upcast (the displaced "
                         "file)")
    ap.add_argument("--emit-npz", action="store_true", help="also write the compact .npz sidecar")
    ap.add_argument("--verify-rows", type=int, default=5000,
                    help="rows used to cross-check the .bin against xgboost (0 = header only)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    date = a.date or datetime.date.today().strftime("%Y%m%d")
    t_start = time.time()

    d = load_caches(a.cache, want_label=a.label)
    arm = a.arm or {"TrkPrompt": "prompt", "TrkDisp": "disp",
                    "TrkMerged": "merged"}.get(d["prefix"], "merged")
    X, names, prov = build_abi_matrix(d["X"], d["feats"], a.feats, a.abi_order,
                                      [c for c in a.extra_cols.split(",") if c.strip()])
    y_mtv, y_eff = d["y_mtv"], d["y_eff"]
    dxy, evt = np.abs(d["dxyBS"]), d["evt"]
    y = y_mtv if a.label == "mtv" else y_eff
    A_all, C_all = (y_eff >= 0.5), (y_mtv < 0.5)
    # the recall axis (rejection always stays on class C)
    dup_all = d["duplicate"] >= 0.5
    Anod_all = A_all & ~dup_all
    ra_nodup = (a.recall_axis == "A_nodup") or (a.dup_as_fake is not None)
    RA_all = Anod_all if ra_nodup else A_all
    recall_axis_eff = "A_nodup" if ra_nodup else "A"
    # the three per-bin recall axes (pT and |eta| come out of the feature matrix itself)
    ptv = X[:, names.index("pt")].astype(np.float64)
    etav = np.abs(X[:, names.index("eta")].astype(np.float64))
    pt_edges = parse_edges(a.pt_bins, "--pt-bins")
    eta_edges = parse_edges(a.eta_bins, "--eta-bins")
    dxy_edges = parse_edges(a.dxy_bins, "--dxy-bins")
    PTB = make_bins(pt_edges, underflow=True)
    ETAB = make_bins(eta_edges, underflow=False)
    DXYB = make_bins(dxy_edges, underflow=False)
    axis_vals = {"pt": ptv, "eta": etav, "dxy": dxy}
    axis_bins = {"pt": PTB, "eta": ETAB, "dxy": DXYB}

    rule = a.wp_rule or ("profile" if a.wp_profile else "uniform")
    if rule == "profile" and not a.wp_profile:
        raise SystemExit("--wp-rule profile needs --wp-profile <json> (see make_profile.py)")
    if rule != "profile" and a.wp_profile:
        raise SystemExit("--wp-profile is only read under --wp-rule profile (got --wp-rule %s)"
                         % rule)
    profile = None
    if rule == "profile":
        with open(a.wp_profile) as fh:
            pj = json.load(fh)
        want = {"pt": pt_edges, "eta": eta_edges, "dxy": dxy_edges}
        profile = {}
        for ax in WP_AXES:
            got = pj.get("axes", {}).get(ax)
            if got is None:
                raise SystemExit("--wp-profile %s has no axis %r" % (a.wp_profile, ax))
            if [float(v) for v in got["edges"]] != want[ax]:
                raise SystemExit(
                    "--wp-profile %s was built with %s edges %s but this run uses %s. The bin "
                    "labels would not line up, so the match-or-exceed targets would be applied to "
                    "the wrong bins." % (a.wp_profile, ax, got["edges"], want[ax]))
            profile[ax] = {k: v["recall"] for k, v in got["bins"].items()}
        print("wp-profile %s (ref %s @ thr %s): match-or-exceed + %.4f, target cap %.4f"
              % (a.wp_profile, pj.get("model", "?"), pj.get("threshold", "?"),
                 a.wp_margin, a.wp_target_cap), flush=True)
    B_all = (y_mtv >= 0.5) & (y_eff < 0.5)
    bad = int((y_eff > y_mtv).sum())
    dxycut = ("full |dxyBS| range" if d["min_dxy"] < 0 else "|dxyBS| > %g" % d["min_dxy"])
    print("rows %d  feats %d  arm=%s (cache prefix=%s, %s)  label=%s (y=%s)"
          % (len(X), X.shape[1], arm, d["prefix"], dxycut, a.label, d["y_is"]), flush=True)
    print("classes: A(y_eff==1)=%d  B(real,not eff TP)=%d  C(y_mtv==0)=%d%s"
          % (A_all.sum(), B_all.sum(), C_all.sum(),
             "   WARNING %d impossible rows y_eff>y_mtv" % bad if bad else ""), flush=True)
    print("target-real %.4f (y_mtv %.4f / y_eff %.4f)" % (y.mean(), y_mtv.mean(), y_eff.mean()),
          flush=True)

    tr, va, te = make_split(evt, a.split, a.valfrac, a.testfrac, a.seed)
    print("split[%s] train %d / val %d / test %d  (%d distinct evt)"
          % (a.split, tr.sum(), va.sum(), te.sum(), len(np.unique(evt))), flush=True)
    for nm, msk in (("train", tr), ("val", va), ("test", te)):
        if msk.sum() == 0:
            raise SystemExit("the %s split is EMPTY under --split %s. With only %d distinct events "
                             "the evt%%10 split cannot fill all three; use --split rows."
                             % (nm, a.split, len(np.unique(evt))))

    A_va, C_va = A_all[va], C_all[va]
    A_te, C_te = A_all[te], C_all[te]
    RA_va, RA_te = RA_all[va], RA_all[te]
    Anod_te, dup_te = Anod_all[te], dup_all[te]
    if a.dup_as_fake is not None and a.recall_axis == "A":
        print("--dup-as-fake implies the non-duplicate recall axis; using A_nodup", flush=True)
    print("recall axis: %s (test rows A=%d, A_nodup=%d, duplicates=%d)"
          % (recall_axis_eff, A_te.sum(), Anod_te.sum(), dup_te.sum()), flush=True)
    if RA_te.sum() == 0 or RA_all[tr].sum() == 0:
        raise SystemExit("the %s recall axis is EMPTY -- the cache carries no non-duplicate "
                         "class-A rows." % recall_axis_eff)

    def dup_stats(sc, t):
        """Duplicate rejection at a threshold: duplicates are real tracks that cost no
        efficiency, so how many of them a working point removes is reported in its own right."""
        if dup_te.sum() == 0 or not np.isfinite(t):
            return {"nDup": int(dup_te.sum()), "dup_kept": None, "dup_rej": None}
        kept = float((sc[dup_te] >= t).mean())
        return {"nDup": int(dup_te.sum()), "dup_kept": kept, "dup_rej": 1.0 - kept}
    if A_te.sum() == 0 or C_te.sum() == 0:
        print("WARNING: test split has A=%d C=%d -- metrics will be degenerate"
              % (A_te.sum(), C_te.sum()), flush=True)

    def axes_for(msk):
        out = []
        for nm in WP_AXES:
            v = axis_vals[nm][msk]
            out.append((nm, v, axis_bins[nm], bin_masks(axis_bins[nm], v)))
        return out

    AX_VA, AX_TE = axes_for(va), axes_for(te)
    wp_mode = {"profile": "match-or-exceed-profile", "uniform": "uniform-per-bin",
               "global": "global"}[rule]
    # The uniform rule is the profile rule with the same target in every bin and no margin, so
    # it goes through the same code; no profile file is involved.
    uniform = {ax: {lab: a.recall for lab, _, _ in axis_bins[ax]} for ax in WP_AXES}

    def wp_threshold(s, A, ax):
        """The run's working-point rule, in one place, so early stopping optimises exactly the
        threshold the model will be deployed at. Returns (threshold, per-bin detail or None)."""
        if rule == "profile":
            return thr_at_profile(A, s, ax, profile, a.wp_margin, a.wp_target_cap, a.pt_bin_min_a)
        if rule == "uniform":
            return thr_at_profile(A, s, ax, uniform, 0.0, a.wp_target_cap, a.pt_bin_min_a,
                                  fallback_recall=a.recall)
        return thr_at_recall(s, A, a.recall), None

    # early-stopping metric: 1 - rejection at the working point, on the val split's A/C axes
    def es_metric(y_true, y_pred):
        s = np.asarray(y_pred, dtype=np.float64)
        if len(s) == len(A_va):
            t = wp_threshold(s, RA_va, AX_VA)[0]
            Cm = C_va
        else:  # xgboost handed us something other than the val set; fall back to the target label
            yt = np.asarray(y_true)
            Am, Cm = (yt >= 0.5), (yt < 0.5)
            t = thr_at_recall(s, Am, a.recall)
        r = rej_at_thr(s, Cm, t)
        return 1.0 if not np.isfinite(r) else 1.0 - r

    print("working point rule: %s (early stopping optimises 1 - rejC at that threshold)" % wp_mode,
          flush=True)

    def make_clf():
        return xgb.XGBClassifier(
            tree_method="hist", max_depth=a.depth, n_estimators=a.ntrees, learning_rate=a.lr,
            subsample=a.subsample, colsample_bytree=a.colsample, min_child_weight=a.mcw,
            reg_lambda=a.lam, n_jobs=a.threads, random_state=a.seed,
            eval_metric=es_metric, early_stopping_rounds=(a.es_rounds if a.es_rounds > 0 else None))

    ytr = y[tr]
    yva_fit = y[va]
    dup_tr = dup_all[tr]
    dupfake = None
    if a.dup_as_fake is not None:
        n_flip = int((dup_tr & (ytr >= 0.5)).sum())
        dupfake = {"weight": float(a.dup_as_fake), "n_train_duplicates": int(dup_tr.sum()),
                   "n_train_rows_flipped_to_negative": n_flip,
                   "note": "duplicates are negatives in the LOSS only; class C, the rejection "
                           "metric and the recall axis are unchanged"}
        ytr = np.where(dup_tr, 0.0, ytr).astype(np.float32)
        yva_fit = np.where(dup_all[va], 0.0, y[va]).astype(np.float32)
        print("dup-as-fake: %d train duplicates -> negatives (%d were positives), loss weight x%.2f"
              % (int(dup_tr.sum()), n_flip, a.dup_as_fake), flush=True)
    pw = (1.0 - ytr.mean()) / max(1e-3, ytr.mean())
    w = np.where(ytr == 1, pw, 1.0).astype(np.float32)
    if a.dup_as_fake is not None:
        w = (w * np.where(dup_tr, float(a.dup_as_fake), 1.0)).astype(np.float32)
    # Displacement re-weighting of the positives (alpha 0 = off), before the two-pass tail
    # re-weight, so the two multiply.
    if a.dxy_alpha > 0:
        dxw = 1.0 + a.dxy_alpha * np.minimum(dxy[tr], 60.0) / 20.0
        w = (w * np.where(ytr == 1, dxw, 1.0)).astype(np.float32)
        print("dxy re-weight: alpha=%.2f -> positive weights x%.2f..x%.2f (mean %.2f)"
              % (a.dxy_alpha, dxw[ytr == 1].min() if (ytr == 1).any() else 0.0,
                 dxw[ytr == 1].max() if (ytr == 1).any() else 0.0,
                 dxw[ytr == 1].mean() if (ytr == 1).any() else 0.0), flush=True)

    # Class-A high-pT re-weighting: the abundant 0.9-1.5 GeV reals otherwise dominate the loss.
    # Class A only (class B is a positive but not on the recall axis), before the tail re-weight.
    A_tr = RA_all[tr]   # the recall axis, so --recall-axis A_nodup does not up-weight duplicates
    if a.eff_weight > 0:
        ptw = 1.0 + a.eff_weight * np.log10(np.maximum(ptv[tr], 0.9) / 0.9)
        w = (w * np.where(A_tr, ptw, 1.0)).astype(np.float32)
        print("eff-weight: A=%.2f -> class-A weights x%.2f..x%.2f (mean %.2f)"
              % (a.eff_weight, ptw[A_tr].min(), ptw[A_tr].max(), ptw[A_tr].mean()), flush=True)

    frw = None
    if a.fake_region_weight:
        terms = parse_fake_region_weight(a.fake_region_weight)
        C_tr = C_all[tr]
        rvals = {"abseta": etav[tr], "absdxy": dxy[tr], "pt": ptv[tr]}
        fw, fdetail = fake_region_weights(terms, rvals, C_tr)
        w = (w * fw).astype(np.float32)
        touched = int((fw > 1.0).sum())
        frw = {"spec": a.fake_region_weight, "terms": fdetail,
               "n_C_train": int(C_tr.sum()), "n_C_reweighted": touched,
               "frac_C_reweighted": float(touched / max(1, int(C_tr.sum()))),
               "max_weight": float(fw.max()),
               "columns": {"abseta": "abs(feature 'eta', FEATS31 index 3)",
                           "absdxy": "abs(cache column 'dxyBS') = abs(feature 'dxyBS', "
                                     "FEATS31 index 10)",
                           "pt": "feature 'pt' (FEATS31 index 7)"}}
        print("fake-region-weight %s: %d of %d train class-C rows re-weighted (max x%.2f)"
              % (a.fake_region_weight, touched, int(C_tr.sum()), fw.max()), flush=True)
        for t_ in fdetail:
            print("    %s [%g, %g) x%.2f -> %d class-C rows (%.4f of C)"
                  % (t_["column"], t_["lo"], t_["hi"], t_["weight"], t_["n_C_rows"],
                     t_["frac_of_C"]), flush=True)

    class _BoosterClf:
        """predict_proba / get_booster of a saved booster, for --from-full."""
        def __init__(self, path):
            self.bst = xgb.Booster(); self.bst.load_model(path)
        def get_booster(self):
            return self.bst
        def predict_proba(self, Xm, iteration_range=None):
            p1 = self.bst.predict(xgb.DMatrix(Xm), iteration_range=iteration_range or (0, 0))
            return np.stack([1.0 - p1, p1], axis=1)

    # pass 1, with the tail re-weight
    pass1 = None
    if a.from_full:
        print("starting from the saved booster %s (no fit)" % a.from_full, flush=True)
    elif a.tailw > 0:
        c0 = make_clf()
        c0.fit(X[tr], ytr, sample_weight=w, eval_set=[(X[va], yva_fit)], verbose=False)
        n0 = int(c0.best_iteration) + 1 if a.es_rounds > 0 else a.ntrees
        s0 = c0.predict_proba(X[tr], iteration_range=(0, n0))[:, 1]
        t0 = thr_at_recall(s0, RA_all[tr], a.recall)
        tail = (ytr == 1) & (s0 < a.tail_band * t0)
        w = w * np.where(tail, a.tailw, 1.0).astype(np.float32)
        pass1 = {"n_trees": n0, "thr_at_recall_train": float(t0),
                 "tail_band": a.tail_band, "n_reweighted": int(tail.sum()),
                 "frac_positives_reweighted": float(tail.sum() / max(1, int((ytr == 1).sum())))}
        print("pass1: trees=%d thr@%.4f(train,A)=%.6f  tail x%.1f on %d positives (%.3f)"
              % (n0, a.recall, t0, a.tailw, tail.sum(), pass1["frac_positives_reweighted"]),
              flush=True)
        if pass1["frac_positives_reweighted"] > 0.9:
            pass1["degenerate"] = True
            print("  WARNING: the tail band %.2f x %.6f = %.3f covers essentially every positive, "
                  "so the second pass is only a uniform class-weight change, not a tail re-weight. "
                  "This means pass 1 did not separate (too few trees / too little data)."
                  % (a.tail_band, t0, a.tail_band * t0), flush=True)

    if a.from_full:
        clf = _BoosterClf(a.from_full)
        nt = int(clf.get_booster().num_boosted_rounds())
        a.ntrees = nt
    else:
        clf = make_clf()
        clf.fit(X[tr], ytr, sample_weight=w, eval_set=[(X[va], yva_fit)], verbose=False)
        nt = int(clf.best_iteration) + 1 if a.es_rounds > 0 else a.ntrees
    size_curve = None
    if a.prune_tol > 0:
        # Size chosen by the data: the working-point fake rejection on the validation split for
        # every prefix of the forest, and the smallest prefix within the tolerance of the best.
        grid = sorted(set(list(range(25, nt, 25)) + [nt]))
        curve = []
        for n in grid:
            sv = clf.predict_proba(X[va], iteration_range=(0, n))[:, 1].astype(np.float64)
            tv = wp_threshold(sv, RA_va, AX_VA)[0]
            rv = rej_at_thr(sv, C_va, tv)
            curve.append((n, float(rv) if np.isfinite(rv) else 0.0))
        best = max(r for _, r in curve)
        chosen = min(n for n, r in curve if r >= best - a.prune_tol)
        size_curve = {"grid": curve, "best_rejection": best, "tolerance": a.prune_tol,
                      "chosen_trees": chosen, "fitted_trees": nt}
        print("size curve (val rejection at the working point): "
              + " ".join("%d:%.4f" % (n, r) for n, r in curve), flush=True)
        print("size chosen by the data: %d of %d trees (best %.4f, tolerance %.4f)"
              % (chosen, nt, best, a.prune_tol), flush=True)
        nt = chosen
    print("pass2: best_iteration+1 = %d of %d trees" % (nt, a.ntrees), flush=True)

    booster = clf.get_booster()
    full_path = os.path.join(a.out, "model_full.json")
    booster.save_model(full_path)
    sliced = booster[0:nt]
    model_path = os.path.join(a.out, "model.json")
    sliced.save_model(model_path)

    s = clf.predict_proba(X[te], iteration_range=(0, nt))[:, 1].astype(np.float64)
    yte = y[te]
    res = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": d["per_file"],
        "n_feats": int(X.shape[1]),
        "feats_abi": names,
        "abi_order": names[31:35] if a.feats == 35 else [],
        "extra_cols": names[a.feats:],
        "label": a.label,
        "y_is": d["y_is"],
        "target_column": "matchedAny" if a.label == "mtv" else "matched",
        "recall_axis_column": "matched",
        "recall_axis": recall_axis_eff,
        "recall_axis_requested": a.recall_axis,
        "recall_axis_rows_test": {"A": int(A_te.sum()), "A_nodup": int(Anod_te.sum()),
                                  "duplicates": int(dup_te.sum())},
        "fake_region_weight": frw,
        "dup_as_fake": dupfake,
        "arm": arm,
        "cache_prefix": d["prefix"],
        "cache_min_dxy": d["min_dxy"],
        "dxy_alpha": a.dxy_alpha,
        "recipe": dict(depth=a.depth, ntrees=a.ntrees, lr=a.lr, min_child_weight=a.mcw,
                       reg_lambda=a.lam, subsample=a.subsample, colsample_bytree=a.colsample,
                       tree_method="hist", es_recall=a.recall, es_rounds=a.es_rounds,
                       tailw=a.tailw, tail_band=a.tail_band, split=a.split, seed=a.seed,
                       eff_weight=a.eff_weight, wp_rule=rule,
                       recall_axis=recall_axis_eff, extra_cols=a.extra_cols,
                       fake_region_weight=a.fake_region_weight,
                       dup_as_fake=a.dup_as_fake,
                       wp_profile=(os.path.abspath(a.wp_profile) if a.wp_profile else None),
                       wp_margin=a.wp_margin, wp_target_cap=a.wp_target_cap,
                       pt_bin_min_a=a.pt_bin_min_a,
                       pt_edges=pt_edges, eta_edges=eta_edges, dxy_edges=dxy_edges),
        "pass1": pass1,
        "n_trees_used": nt,
        "size_curve": size_curve,
        "n_trees_saved": a.ntrees,
        "rows": {"total": int(len(X)), "train": int(tr.sum()), "val": int(va.sum()),
                 "test": int(te.sum())},
        "classes_test": {"A": int(A_te.sum()), "B": int(B_all[te].sum()), "C": int(C_te.sum())},
        "classes_all": {"A": int(A_all.sum()), "B": int(B_all.sum()), "C": int(C_all.sum())},
        "impossible_rows_yeff_gt_ymtv": bad,
        "train_s": round(time.time() - t_start),
    }
    res["auc_target"] = (float(roc_auc_score(yte, s)) if len(np.unique(yte)) > 1 else None)
    AC = A_te | C_te
    res["auc_AC"] = (float(roc_auc_score(A_te[AC], s[AC]))
                     if AC.sum() and len(np.unique(A_te[AC])) > 1 else None)

    # PR-AUC companions, report only. A degenerate split (one class only) yields None.
    y_ac, s_ac = A_te[AC], s[AC]
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
        # the threshold the global working point uses, at 99.5 % class-A recall (res["thr@0.9950"])
        res["precision_AC_at_recall995"] = precision_at_thr(s, A_te, C_te,
                                                            thr_at_recall(s, A_te, 0.995))
    except Exception:
        res["precision_AC_at_recall995"] = None

    for rc in (0.999, a.recall, 0.995, 0.99, 0.98):
        t = thr_at_recall(s, RA_te, rc)
        res["thr@%.4f" % rc] = t
        res["rej@%.4f" % rc] = rej_at_thr(s, C_te, t)
    # both recall axes are always scanned, so runs made under different --recall-axis settings
    # stay comparable without re-scoring
    res["axis_scan"] = {}
    for axname, axmask in (("A", A_te), ("A_nodup", Anod_te)):
        blk = {}
        for rc in (0.999, a.recall, 0.995, 0.99, 0.98):
            t = thr_at_recall(s, axmask, rc)
            e = {"thr": t, "rejC": rej_at_thr(s, C_te, t),
                 "recall_on_A": recall_at_thr(s, A_te, t),
                 "recall_on_A_nodup": recall_at_thr(s, Anod_te, t)}
            e.update(dup_stats(s, t))
            blk["%.4f" % rc] = e
        res["axis_scan"][axname] = blk

    # Three thresholds, always all three, so runs made under different rules stay comparable:
    #   wp       the run's own rule (--wp-rule)
    #   uniform  --recall in every bin of every axis (equal to wp under the default rule)
    #   global   the plain --recall quantile over all class A (what a flat threshold scan gives)
    thr_wp, wp_detail = wp_threshold(s, RA_te, AX_TE)
    thr_unif = thr_at_recall_axes(RA_te, s, AX_TE, a.recall, a.pt_bin_min_a)
    thr_glob = thr_at_recall(s, RA_te, a.recall)
    res["wp_mode"] = wp_mode
    res["working_point"] = {"recall": a.recall, "mode": wp_mode, "threshold": thr_wp,
                            "rejection": rej_at_thr(s, C_te, thr_wp),
                            "recall_achieved": recall_at_thr(s, RA_te, thr_wp),
                            "recall_axis": recall_axis_eff,
                            "recall_on_A": recall_at_thr(s, A_te, thr_wp),
                            "recall_on_A_nodup": recall_at_thr(s, Anod_te, thr_wp)}
    res["working_point"].update(dup_stats(s, thr_wp))
    if wp_detail is not None:
        res["working_point"]["per_bin_targets"] = wp_detail
        binding = [(v["thr"], k) for k, v in wp_detail.items()
                   if v.get("used") and np.isfinite(v.get("thr", float("nan")))]
        if binding:
            res["working_point"]["binding_bin"] = min(binding)[1]

    tables = {}
    for tag, t in (("wp", thr_wp), ("uniform%.4f" % a.recall, thr_unif),
                   ("global%.4f" % a.recall, thr_glob)):
        tab = {"pt": bin_table(PTB, ptv[te], A_te, C_te, s, t),
               "eta": bin_table(ETAB, etav[te], A_te, C_te, s, t),
               "dxy": bin_table(DXYB, dxy[te], A_te, C_te, s, t)}
        mn, where = min_bin_recall(tab, a.pt_bin_min_a)
        res["at_%s" % tag] = {"threshold": t, "rejection": rej_at_thr(s, C_te, t),
                              "recall": recall_at_thr(s, A_te, t),
                              "recall_on_A": recall_at_thr(s, A_te, t),
                              "recall_on_A_nodup": recall_at_thr(s, Anod_te, t),
                              "min_bin_recall": mn, "min_bin": where, "bins": tab}
        res["at_%s" % tag].update(dup_stats(s, t))
        tables[tag] = tab
    res["thresholds"] = {"wp": thr_wp, "uniform": thr_unif, "global": thr_glob}
    if profile is not None:
        # did the run actually clear the reference in every bin?
        margins = {}
        worst = None
        for ax in WP_AXES:
            for lab, v in tables["wp"][ax].items():
                ref = profile[ax].get(lab)
                if ref is None or v["nA"] < a.pt_bin_min_a or not np.isfinite(v["recall"]):
                    continue
                dm = v["recall"] - float(ref)
                margins["%s:%s" % (ax, lab)] = dm
                if worst is None or dm < worst[0]:
                    worst = (dm, "%s:%s" % (ax, lab))
        res["profile_margins@wp"] = margins
        res["profile_worst_margin@wp"] = {"delta_recall": worst[0], "bin": worst[1]} if worst else None
        res["profile_no_regression@wp"] = bool(worst is not None and worst[0] >= 0.0)

    # per-iteration-origin breakdown
    it_te = prov["iteration"][te].astype(np.int64) if "iteration" in prov else None
    if it_te is not None:
        by_iter, by_group = {}, {}
        for rc in (a.recall, 0.99):
            t = thr_at_recall(s, A_te, rc)
            for v in sorted(np.unique(it_te).tolist()):
                m = it_te == v
                key = "%d_%s" % (v, ITER_NAME.get(v, "iter%d" % v))
                by_iter.setdefault(key, {})["@%.4f" % rc] = {
                    "nA": int((m & A_te).sum()), "nC": int((m & C_te).sum()),
                    "recall": recall_at_thr(s, m & A_te, t), "rej": rej_at_thr(s, m & C_te, t)}
            for g in ("prompt", "displaced", "other"):
                m = np.isin(it_te, [k for k, gg in ITER_GROUP.items() if gg == g])
                if m.sum() == 0:
                    continue
                by_group.setdefault(g, {})["@%.4f" % rc] = {
                    "nA": int((m & A_te).sum()), "nC": int((m & C_te).sum()),
                    "recall": recall_at_thr(s, m & A_te, t), "rej": rej_at_thr(s, m & C_te, t)}
        res["by_iteration"] = by_iter
        res["by_iteration_group"] = by_group

    # per |dxyBS| bin at the working point
    dte = dxy[te]
    by_dxy = {}
    for lo, hi in DXY_BINS:
        m = (dte >= lo) & (dte < hi)
        key = "%.1f-%s" % (lo, "inf" if hi == float("inf") else "%.1f" % hi)
        by_dxy[key] = {"nA": int((m & A_te).sum()), "nC": int((m & C_te).sum()),
                       "recall": recall_at_thr(s, m & A_te, thr_wp),
                       "rej": rej_at_thr(s, m & C_te, thr_wp)}
    res["by_dxyBS@wp"] = by_dxy

    # feature importance (gain), top 15; keys are f<index> because training uses a bare ndarray
    gain = sliced.get_score(importance_type="gain")
    imp = sorted(((names[int(k[1:])] if k[1:].isdigit() and int(k[1:]) < len(names) else k,
                   float(v)) for k, v in gain.items()), key=lambda kv: -kv[1])
    res["importance_gain_top15"] = [{"feat": f, "gain": g} for f, g in imp[:15]]
    res["feats_unused"] = [n for i, n in enumerate(names) if ("f%d" % i) not in gain]

    rcs = sorted(set([round(v, 5) for v in np.arange(0.97, 0.99751, 0.0025)]
                     + [0.998, 0.9985, 0.999, a.recall]))
    binned = rule != "global"
    thrmap = os.path.join(a.out, "thrmap.txt")
    with open(thrmap, "w") as f:
        f.write("# %s HP forest working points -- test split\n" % arm)
        f.write("# recall on class A (y_eff==1, nA=%d), rejection on class C (y_mtv==0, nC=%d)\n"
                % (A_te.sum(), C_te.sum()))
        f.write("# model: %d feats, %d trees, label=%s\n" % (X.shape[1], nt, a.label))
        if binned:
            f.write("# threshold = the PER-BIN threshold: the min over pT bins of the per-bin\n"
                    "# thr_at_recall, i.e. the cut at which EVERY pT bin reaches recallA. thr_glob\n"
                    "# is the plain global quantile. The trailing columns are the class-A recall\n"
                    "# each pT bin actually has at `threshold`.\n")
            f.write("# %-8s %-12s %-10s %-10s %-12s %s\n"
                    % ("recallA", "threshold", "rejC", "recall_ach", "thr_glob",
                       " ".join("%-9s" % lab for lab, _, _ in PTB)))
        else:
            f.write("# %-8s %-12s %-10s %-10s\n"
                    % ("recallA", "threshold", "rejC", "recall_ach"))
        if recall_axis_eff != "A":
            f.write("# recall AXIS = %s: thresholds are derived on class-A rows with "
                    "duplicate==0 (nA_nodup=%d); the recall_ach column stays on ALL of class A, "
                    "and recall_nodup is the axis the threshold was set on.\n"
                    % (recall_axis_eff, int(Anod_te.sum())))
        for rc in rcs:
            tg = thr_at_recall(s, RA_te, rc)
            t = (thr_at_recall_binned(RA_te, s, ptv[te], pt_edges, rc, a.pt_bin_min_a)
                 if binned else tg)
            row = "  %-8.4f %-12.6f %-10.5f %-10.5f" % (
                rc, t, rej_at_thr(s, C_te, t), recall_at_thr(s, A_te, t))
            if binned:
                row += " %-12.6f " % tg
                row += " ".join("%-9.5f" % recall_at_thr(s, A_te & m, t)
                                for _, m in bin_masks(PTB, ptv[te]))
            f.write(row + "\n")
        f.write("#\n# working point [%s, axis %s]: thr=%.6f rejC=%.5f recallA=%.5f "
                "recallA_nodup=%.5f dupRej=%s\n"
                % (wp_mode, recall_axis_eff, thr_wp, res["working_point"]["rejection"],
                   res["working_point"]["recall_on_A"],
                   res["working_point"]["recall_on_A_nodup"],
                   "%.5f" % res["working_point"]["dup_rej"]
                   if res["working_point"]["dup_rej"] is not None else "n/a"))
        for tag in ("wp", "uniform%.4f" % a.recall, "global%.4f" % a.recall):
            e = res["at_%s" % tag]
            f.write("# %-16s thr=%-10.6f rejC=%-8.5f min-bin recall=%s (%s)\n"
                    % (tag, e["threshold"], e["rejection"],
                       "%.5f" % e["min_bin_recall"] if e["min_bin_recall"] is not None else "n/a",
                       e["min_bin"]))
    print("wrote %s" % thrmap, flush=True)

    exp = export_compact_tree
    exp_path = os.path.abspath(exp.__file__)
    feat_a, val_a, left_a, right_a, roots_a, base_logit, n_out = exp.build_compact(sliced)
    fp16 = not a.no_fp16
    val_out = val_a.astype(np.float16).astype(np.float32) if fp16 else val_a
    if feat_a.size and int(feat_a.max()) >= X.shape[1]:
        raise SystemExit("exported feature index %d >= n_feats %d -- ABI mapping is wrong"
                         % (int(feat_a.max()), X.shape[1]))
    bin_name = "%s_tree%d_%s_%s.bin" % (arm, X.shape[1], a.tag, date)
    bin_path = os.path.join(a.out, bin_name)
    size = exp.write_bin(bin_path, feat_a, val_out, left_a, right_a, roots_a, base_logit, n_out)
    res["export"] = {"bin": os.path.abspath(bin_path), "nNodes": int(feat_a.shape[0]),
                     "nTrees": int(n_out), "baseLogit": float(base_logit), "size_bytes": int(size),
                     "val_precision": "fp16->fp32" if fp16 else "fp32",
                     "max_feature_index": int(feat_a.max()) if feat_a.size else -1,
                     "exporter": exp_path}
    print("wrote %s: nNodes=%d nTrees=%d baseLogit=%.8f size=%d (%s)"
          % (bin_path, feat_a.shape[0], n_out, base_logit, size, res["export"]["val_precision"]),
          flush=True)
    if a.emit_npz:
        npz_path = bin_path[:-4] + ".npz"
        exp.write_npz(npz_path, feat_a, val_a, left_a, right_a, roots_a, base_logit, n_out, fp16)
        res["export"]["npz"] = os.path.abspath(npz_path)

    rd = read_compact_tree
    rd_path = os.path.abspath(rd.__file__)
    m = rd.load_compact(bin_path)
    ver = {"reader": rd_path,
           "nNodes": int(m["nNodes"]), "nTrees": int(m["nTrees"]),
           "baseLogit": float(m["baseLogit"]),
           "n_leaves": int((m["feat"] < 0).sum()),
           "feat_index_min": int(m["feat"][m["feat"] >= 0].min()) if (m["feat"] >= 0).any() else -1,
           "feat_index_max": int(m["feat"].max()),
           "nNodes_matches": int(m["nNodes"]) == int(feat_a.shape[0]),
           "nTrees_matches": int(m["nTrees"]) == int(n_out),
           "feat_index_in_range": bool(int(m["feat"].max()) < X.shape[1])}
    if a.verify_rows > 0 and te.sum():
        k = min(a.verify_rows, int(te.sum()))
        Xv = X[te][:k]
        sb = np.asarray(rd.score_compact(m, Xv))
        sx = clf.predict_proba(Xv, iteration_range=(0, nt))[:, 1].astype(np.float64)
        ad = np.abs(sb - sx)
        # With fp16 values the split thresholds are rounded too, so a row between the fp32 and the
        # fp16 threshold takes the other branch and its score jumps: the fp16 gate is therefore on
        # the median and on the flip rate, not on the max. With --no-fp16 the traversal must
        # reproduce xgboost essentially exactly.
        ver.update(score_check_rows=k, score_max_abs_diff=float(ad.max()),
                   score_median_abs_diff=float(np.median(ad)),
                   score_p99_abs_diff=float(np.quantile(ad, 0.99)),
                   score_frac_over_1e3=float((ad > 1e-3).mean()))
        ver["score_ok"] = bool(ver["score_median_abs_diff"] < 1e-4
                               and ver["score_frac_over_1e3"] < 0.02) if fp16 \
            else bool(ver["score_max_abs_diff"] < 1e-5)
    res["verify"] = ver
    print("read back [%s]: nNodes=%d nTrees=%d leaves=%d featIdx %d..%d (<%d: %s)"
          % (os.path.basename(ver["reader"]), ver["nNodes"], ver["nTrees"], ver["n_leaves"],
             ver["feat_index_min"], ver["feat_index_max"], X.shape[1],
             ver["feat_index_in_range"]), flush=True)
    if "score_max_abs_diff" in ver:
        print("score cross-check on %d test rows: max|d|=%.3e median|d|=%.3e -> %s"
              % (ver["score_check_rows"], ver["score_max_abs_diff"],
                 ver["score_median_abs_diff"], "OK" if ver["score_ok"] else "MISMATCH"), flush=True)

    res["train_s"] = round(time.time() - t_start)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)

    print("\nFINAL trees=%d feats=%d AUC(A/C)=%s thr@%.4f=%.6f rej=%.5f | rej@0.99=%.5f "
          "rej@0.98=%.5f (%ds)"
          % (nt, X.shape[1],
             "%.5f" % res["auc_AC"] if res["auc_AC"] is not None else "n/a",
             a.recall, thr_wp, res["working_point"]["rejection"],
             res["rej@0.9900"], res["rej@0.9800"], res["train_s"]), flush=True)
    _n = lambda v: ("%.5f" % v) if isinstance(v, float) and np.isfinite(v) else "n/a"
    print("  PR (info only)   prAUC=%s  prAUC(A/C)=%s  partial-prAUC(A/C, recall>=0.99)=%s  "
          "precision(A/C)@recallA=0.995 = %s"
          % (_n(res["prauc_target"]), _n(res["prauc_AC"]), _n(res["partial_prauc_AC_r99"]),
             _n(res["precision_AC_at_recall995"])), flush=True)
    for tag in ("wp", "uniform%.4f" % a.recall, "global%.4f" % a.recall):
        e = res["at_%s" % tag]
        print("  %-16s thr=%-10.6f rejC=%-8.5f recallA=%-8.5f recallA_nodup=%-8.5f dupRej=%-8s "
              "min-bin=%s (%s)"
              % (tag, e["threshold"], e["rejection"], e["recall"], e["recall_on_A_nodup"],
                 "%.5f" % e["dup_rej"] if e["dup_rej"] is not None else "n/a",
                 "%.5f" % e["min_bin_recall"] if e["min_bin_recall"] is not None else "n/a",
                 e["min_bin"]), flush=True)
    if "profile_worst_margin@wp" in res and res["profile_worst_margin@wp"]:
        wm = res["profile_worst_margin@wp"]
        print("  vs reference profile: worst per-bin margin %+.5f in %s -> no-regression %s"
              % (wm["delta_recall"], wm["bin"], res["profile_no_regression@wp"]), flush=True)
    if "by_iteration_group" in res:
        for g, v in res["by_iteration_group"].items():
            k = "@%.4f" % a.recall
            print("  %-10s nA=%-8d nC=%-7d recall=%.4f rej=%.5f"
                  % (g, v[k]["nA"], v[k]["nC"], v[k]["recall"], v[k]["rej"]), flush=True)
    print("wrote %s" % os.path.join(a.out, "result.json"), flush=True)


if __name__ == "__main__":
    main()
