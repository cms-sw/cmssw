#!/usr/bin/env python3
"""Trainer for the in-kernel per-triplet DNN gate (torch backend).

Consumes the triplet dataset from triplet_dump_cfg.py: per built triplet, 18 raw dump
features + signed 'curvature' + the truth label (set when the triplet's three hits all
point at the same TrackingParticle).

Features: derived features (pulls/residuals/log-compressions); optional one-hot layer-id block;
weighted-BCE/focal losses; POPULATION-BIAS-AWARE weighting (--weight-mode, per-(|tip|,pT) cell
occupancy); threshold rules (--threshold-rule: global recall, or WORST-GROUP survival/recall);
early stopping on the operating metric; monitoring plots;
baked-header export + float32 numpy-forward self-check against the torch model.

Memory: ingestion is chunked into preallocated float32 columns, the shuffle is an index
permutation (not a frame copy), the split is event-level, and a background thread polls
/sys/fs/cgroup/memory.current (NOT free/MemAvailable, which lies inside a cgroup) and aborts
before the cap. See --chunk-events / --mem-guard-gb.

  train:   python3 train_triplet_dnn.py train --variant derived_wbce --device cuda:0 nano.root:tag ...
  finalize: python3 train_triplet_dnn.py finalize nano.root:tag ... --bank displaced [--out <header>]

--bank selects the nano table (displaced -> TrkDispTriplet, prompt -> TrkPromptTriplet). The
derived-feature formulas MUST stay in sync with CATripletCuts.h accept() (eps conventions included).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_FEATURES = [
    "absCurv", "tipXcurv", "dca",
    "curvatureStubs", "curvatureStubsErrSq", "curvature13",
    "dPhi12", "dPhi13", "dPhi23", "dr12", "dr13",
    "r1", "r2", "r3", "z1", "z2", "z3", "nStubs",
]
DERIVED = ["stubCirclePull", "stubCircleRatio", "curv13Resid", "rzResid", "cotTheta",
           "dPhiRatio", "logAbsCurv", "logErrSq", "logDca", "layGap12", "layGap23"]
# Appended features, indices 29/30.
# True logarithms of the triplet's own fitted |tip| and curvature. logDca and logAbsCurv are log1p with
# an argument scaled so that the displaced and high-pT regions fall in log1p's linear part, where the
# coordinate is blind to multiplicative changes; a true log restores uniform per-e-fold resolution there.
# Kept in a separate list so that `BASE_FEATURES + DERIVED` remains the 29-column vector the deployed
# headers are baked from. Opt in to the wider vector with --nfeat 31.
DERIVED_EXT = ["logTip", "logPt"]
LAYER_FEATURES = ["lay1", "lay2", "lay3"]
ALL_FEATS = BASE_FEATURES + DERIVED + DERIVED_EXT
LEGACY_NFEAT = len(BASE_FEATURES) + len(DERIVED)  # 29
FEATS = list(ALL_FEATS[:LEGACY_NFEAT])            # default = the 29-column width; set_nfeat() widens it
EPS = 1e-12

# Clamps of the appended log features. They double as sentinels and are physically monotone-consistent,
# so no NaN or Inf is reachable. They must match CATripletCuts.h accept() bit-for-bit.
#   1e-4 cm = 1 um: below the beamspot transverse size (~15 um) and below any 3-hit tip
#     resolution => "indistinguishable from zero". A degenerate (collinear) fit has
#     absCurvature == 0, hence dca == 0 by the in-kernel guard, and maps to ln(1e-4) = -9.2103404.
#   1e-6 cm^-1: R = 10 km ~ pT 11 TeV, above any physical track; also absorbs the exact-zero
#     collinear case -> -ln(1e-6) = +13.8155106 ("maximally straight").
TIP_CLAMP_CM = 1e-4
CURV_CLAMP_INVCM = 1e-6


def set_nfeat(n):
    """Restrict the feature vector to its leading n columns (APPEND-ONLY ABI, so n=29 is
    the deployed 29-column vector). Called BEFORE ingestion so the matrix is built at the
    requested width and fork workers inherit it (no post-hoc slice, no extra copy)."""
    global FEATS
    assert 1 <= n <= len(ALL_FEATS), "nfeat must be in [1, %d]" % len(ALL_FEATS)
    FEATS = list(ALL_FEATS[:n])
    return FEATS


def drop_features(names):
    """REMOVE named features from the active vector, preserving order (for ablation studies).
    Complement of set_nfeat() (which truncates). Called BEFORE ingestion so the matrix is built
    at the reduced width. Off by default.

    Deployment note: a dropped feature needs NO kernel change -- layer 1 is affine in the
    standardized inputs, so zero its kW1 column and fold the constant into kB1."""
    global FEATS
    miss = [n for n in names if n not in FEATS]
    assert not miss, "drop_features: not in the active vector: %s (have %s)" % (miss, FEATS)
    FEATS = [f for f in FEATS if f not in names]
    assert FEATS, "drop_features: refusing to drop every feature"
    return FEATS

# model/plot output directory. TRIPLET_DNN_OUTDIR redirects it (retrain_triplet_gate.sh points
# it at the working directory, so nothing is written into the source tree).
PLOTDIR = os.environ.get("TRIPLET_DNN_OUTDIR",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_tripletDNN"))

# nano table name per weight bank (triplet_dump_cfg.py DS_NTUPLE_ITER output).
TABLE_FOR_BANK = {"displaced": "TrkDispTriplet", "prompt": "TrkPromptTriplet"}

# name: (use_onehot, loss, weight_mode)
VARIANTS = {
    "derived_wbce": (False, "wbce", "none"),
    "derived_focal": (False, "focal", "none"),
    "onehot_wbce": (True, "wbce", "none"),
    "onehot_focal": (True, "focal", "none"),
    # population-bias-aware arms.
    # Same 29-feature 64x64 architecture: only the training objective (per-sample weights) and the
    # threshold rule change.
    "bal_cell1": (False, "wbce", "cell1"),     # inverse-frequency cell weights, alpha=1
    "bal_cell05": (False, "wbce", "cell05"),   # inverse-frequency cell weights, alpha=1/2
    "bal_dro": (False, "wbce", "dro"),         # adaptive group-DRO on the per-group recall deficit
    # Tail-only arms: the same weight map with --weight-floor 1 --weight-normalize bulk, so the
    # prompt/low-pT bulk keeps its real-vs-fake calibration and only the rare groups are upweighted.
    "bal_up1": (False, "wbce", "cell1"),
    "bal_up05": (False, "wbce", "cell05"),
}

# Pixel-only (nStubs == 0) sentinels, which must match CATripletCuts.h accept() bit-for-bit: at
# nStubs == 0 the kernel has no stub curvature, the weighted mean being 0/0, so it feeds the DNN
# curvatureStubs = 0 and curvatureStubsErrSquared = 1e6. Those two columns are overwritten for
# pixel-only rows before deriving, so such a training row is identical to what the kernel evaluates.
NO_STUB_CURV = 0.0       # curvatureStubs sentinel  (matches 0.f in accept())
NO_STUB_CURV_ERRSQ = 1e6  # curvatureStubsErrSquared sentinel (matches 1e6f in accept())

# Population groups. |tip| = |tipTimesCurvature| / |absCurvature| is the triplet's own fitted transverse
# impact parameter, the only displacement proxy in the dump; pT is the matched TP's pT, used only for the
# real rows, which are the only rows that get a non-unit weight.
TIP_EDGES = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 1e9])
PT_EDGES = np.array([0.0, 0.9, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 1e9])
ETA_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5])
NS_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
K_EDGES = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 9.5, 1e9])
NTIP, NPT = len(TIP_EDGES) - 1, len(PT_EDGES) - 1


def bin_of(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2).astype(np.int16)


# Memory guard: inside a cgroup, free and MemAvailable report the host and are meaningless. Poll
# /sys/fs/cgroup/memory.current, log, and abort before the cap is hit.
_MEM_PATH = "/sys/fs/cgroup/memory.current"
_MEM_STAT = "/sys/fs/cgroup/memory.stat"


def cgroup_mem_gb():
    """(memory.current, anonymous) in GiB. memory.current is the cgroup total, but a large part of it
    can be NFS page cache, which is reclaimable and must NOT trip the guard;
    the number that actually causes an OOM kill is the ANONYMOUS footprint. Both are logged; the
    guard trips on anon."""
    cur = anon = float("nan")
    try:
        cur = int(open(_MEM_PATH).read().strip()) / 2**30
        for line in open(_MEM_STAT):
            if line.startswith("anon "):
                anon = int(line.split()[1]) / 2**30
                break
    except Exception:
        pass
    return cur, anon


def start_mem_guard(limit_gb=260.0, period=60.0, log=None):
    if not os.path.exists(_MEM_PATH):
        print("[memguard] %s absent -- guard disabled" % _MEM_PATH, flush=True)
        return

    def loop():
        while True:
            cur, anon = cgroup_mem_gb()
            msg = ("[memguard] %s cgroup.current=%.0fG anon=%.0fG (trip>%.0fG of the 360G cap) "
                   "self_rss=%.0fG" % (time.strftime("%H:%M:%S"), cur, anon, limit_gb, _self_rss_gb()))
            print(msg, flush=True)
            if log:
                with open(log, "a") as f:
                    f.write(msg + "\n")
            if anon == anon and anon > limit_gb:
                print("[memguard] CGROUP GUARD TRIP (anon %.0fG > %.0fG) -- aborting" % (anon, limit_gb),
                      flush=True)
                sys.stdout.flush()
                os._exit(9)
            time.sleep(period)

    threading.Thread(target=loop, daemon=True).start()


# Worker-pool sizing. Every CPU-bound pass here is chunked, so it parallelises with a process pool over
# chunks and a deterministic reduce, chunks being consumed strictly in index order. The binding constraint
# is memory rather than cores: N_workers x (per-worker transient) <= MEM_BUDGET_GB, with per-worker
# transients of about 1.0 GB for ingestion and 0.5 GB for scoring. The pool uses fork, so the read-only
# feature matrix is shared copy-on-write.
MEM_BUDGET_GB = 200.0
PER_WORKER_GB = 2.5


def default_workers():
    ncpu = len(os.sched_getaffinity(0))
    return max(1, min(ncpu - 2, int(MEM_BUDGET_GB / PER_WORKER_GB), 24))


def _pool(workers):
    """Fork-based pool (so children inherit the parent's big read-only arrays without copying)."""
    return mp.get_context("fork").Pool(processes=workers)


def _self_rss_gb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return float("nan")


# The CMSSW py3-torch build is compiled without numpy support, so torch.from_numpy and Tensor.numpy()
# raise. torch.frombuffer over the array's own memoryview is a zero-copy substitute, which matters here
# because the feature matrix is tens of GiB.
_TORCH_DTYPE = {np.dtype("float32"): torch.float32, np.dtype("float64"): torch.float64,
                np.dtype("int64"): torch.int64, np.dtype("int32"): torch.int32,
                np.dtype("int16"): torch.int16, np.dtype("int8"): torch.int8}


def t_of(a):
    """Zero-copy torch view of a C-contiguous numpy array (the array must outlive the tensor)."""
    a = np.ascontiguousarray(a)
    t = torch.frombuffer(memoryview(a.reshape(-1)), dtype=_TORCH_DTYPE[a.dtype])
    return t.view(*a.shape) if a.ndim > 1 else t


def np_of(t, out=None):
    """Copy a torch tensor into a numpy array (device -> host), no NumPy interop required."""
    t = t.detach().cpu().contiguous()
    dt = {v: k for k, v in _TORCH_DTYPE.items()}[t.dtype]
    if out is None:
        out = np.empty(tuple(t.shape), dtype=dt)
    t_of(out).copy_(t)
    return out


def apply_pixel_only_sentinels(df):
    """Overwrite the stub-curvature columns for pixel-only (nStubs==0) rows with the in-kernel
    sentinels, mirroring the nStubs==0 branch of CATripletCuts.h accept()."""
    if "nStubs" not in df.columns:
        return df
    pix = df["nStubs"].values < 0.5  # nStubs is a float count; exact 0.0 in the dump
    if pix.any():
        df.loc[pix, "curvatureStubs"] = NO_STUB_CURV
        df.loc[pix, "curvatureStubsErrSq"] = NO_STUB_CURV_ERRSQ
    return df


def add_derived(df):
    """Derived features; formulas mirrored 1:1 in CATripletCuts.h accept()."""
    df = apply_pixel_only_sentinels(df)  # nStubs==0 sentinels first, so pulls/logs are bit-consistent
    err = np.sqrt(np.maximum(df.curvatureStubsErrSq.values, EPS))
    df["stubCirclePull"] = (df.curvatureStubs - df.curvature) / err
    df["stubCircleRatio"] = df.curvatureStubs / (np.abs(df.curvature) + EPS)
    df["curv13Resid"] = df.curvature13 - df.curvature
    dr13 = df.r3 - df.r1
    df["rzResid"] = df.z2 - (df.z1 + (df.r2 - df.r1) * (df.z3 - df.z1) / (dr13 + EPS))
    df["cotTheta"] = (df.z3 - df.z1) / (dr13 + EPS)
    denom = df.dPhi23 + np.sign(df.dPhi23) * EPS + (df.dPhi23 == 0) * EPS
    df["dPhiRatio"] = df.dPhi12 / denom
    df["logAbsCurv"] = np.log1p(np.abs(df.absCurv) * 1e3)
    df["logErrSq"] = np.log1p(df.curvatureStubsErrSq * 1e6)
    df["logDca"] = np.log1p(np.abs(df.dca))
    df["layGap12"] = df.lay2 - df.lay1
    df["layGap23"] = df.lay3 - df.lay2
    df["logTip"] = np.log(np.maximum(np.abs(df.dca.values), TIP_CLAMP_CM))
    df["logPt"] = -np.log(np.maximum(np.abs(df.absCurv.values), CURV_CLAMP_INVCM))
    return df


def derive_np(d):
    """numpy twin of add_derived (same formulas, same eps, same sentinels); operates on a dict of
    1-D float32 arrays keyed by the trainer feature names and returns the (n, 29) matrix."""
    pix = d["nStubs"] < 0.5
    d["curvatureStubs"] = np.where(pix, NO_STUB_CURV, d["curvatureStubs"]).astype(np.float32)
    d["curvatureStubsErrSq"] = np.where(pix, NO_STUB_CURV_ERRSQ, d["curvatureStubsErrSq"]).astype(np.float32)
    err = np.sqrt(np.maximum(d["curvatureStubsErrSq"], EPS))
    d["stubCirclePull"] = (d["curvatureStubs"] - d["curvature"]) / err
    d["stubCircleRatio"] = d["curvatureStubs"] / (np.abs(d["curvature"]) + EPS)
    d["curv13Resid"] = d["curvature13"] - d["curvature"]
    dr13 = d["r3"] - d["r1"]
    d["rzResid"] = d["z2"] - (d["z1"] + (d["r2"] - d["r1"]) * (d["z3"] - d["z1"]) / (dr13 + EPS))
    d["cotTheta"] = (d["z3"] - d["z1"]) / (dr13 + EPS)
    denom = d["dPhi23"] + np.sign(d["dPhi23"]) * EPS + (d["dPhi23"] == 0) * EPS
    d["dPhiRatio"] = d["dPhi12"] / denom
    d["logAbsCurv"] = np.log1p(np.abs(d["absCurv"]) * 1e3)
    d["logErrSq"] = np.log1p(d["curvatureStubsErrSq"] * 1e6)
    d["logDca"] = np.log1p(np.abs(d["dca"]))
    d["layGap12"] = d["lay2"] - d["lay1"]
    d["layGap23"] = d["lay3"] - d["lay2"]
    # true logs (indices 29/30); clamps == the in-kernel ones, so no NaN/Inf is reachable
    d["logTip"] = np.log(np.maximum(np.abs(d["dca"]), np.float32(TIP_CLAMP_CM)))
    d["logPt"] = -np.log(np.maximum(np.abs(d["absCurv"]), np.float32(CURV_CLAMP_INVCM)))
    return np.stack([d[f] for f in FEATS], axis=1).astype(np.float32)


_RENAME = {"absCurvature": "absCurv", "tipTimesCurvature": "tipXcurv",
           "curvatureStubsErrSquared": "curvatureStubsErrSq"}
_TRIP_COLS = ["absCurvature", "tipTimesCurvature", "dca", "curvatureStubs", "curvatureStubsErrSquared",
              "curvature13", "dPhi12", "dPhi13", "dPhi23", "dr12", "dr13", "r1", "r2", "r3",
              "z1", "z2", "z3", "nStubs", "curvature", "lay1", "lay2", "lay3", "h1", "h2", "h3"]


def _ingest_chunk(job):
    """Read events [a,b) of one nano file and return the compact per-chunk columns.
    Module-level and self-contained so a process pool can run it; each worker opens its own uproot
    handle. Pure function of (path, table, a, b, label_mode) -> deterministic."""
    path, table, a, b, label_mode = job
    tf = uproot.open(path)["Events"]
    trip = {c: tf["%s_%s" % (table, c)].array(library="np", entry_start=a, entry_stop=b)
            for c in _TRIP_COLS}
    tk = tf["HitTruth_tpKey"].array(library="np", entry_start=a, entry_stop=b)
    tpt = tf["HitTruth_tpPt"].array(library="np", entry_start=a, entry_stop=b)
    tet = tf["HitTruth_tpEta"].array(library="np", entry_start=a, entry_stop=b)
    out = {k: [] for k in ("X", "y", "nStubs", "tip", "tpPt", "tpEta", "lay", "ev", "tpKey")}
    for e in range(b - a):
        h1, h2, h3 = trip["h1"][e], trip["h2"][e], trip["h3"][e]
        kk = np.asarray(tk[e])
        n = len(kk)
        if n == 0 or len(h1) == 0:
            continue
        ok = (h1 < n) & (h2 < n) & (h3 < n)
        i1 = np.clip(h1, 0, n - 1)
        k1 = np.where(ok, kk[i1], -1)
        k2 = np.where(ok, kk[np.clip(h2, 0, n - 1)], -1)
        k3 = np.where(ok, kk[np.clip(h3, 0, n - 1)], -1)
        if label_mode == "2of3":
            lab = ok & (((k1 >= 0) & (k1 == k2)) | ((k1 >= 0) & (k1 == k3)) |
                        ((k2 >= 0) & (k2 == k3)))
        else:
            lab = ok & (k1 >= 0) & (k1 == k2) & (k2 == k3)
        d = {_RENAME.get(c, c): np.asarray(trip[c][e], dtype=np.float32) for c in _TRIP_COLS}
        lay = np.stack([d["lay1"], d["lay2"], d["lay3"]], axis=1).astype(np.int16)
        nst = d["nStubs"].copy()
        tip = np.abs(d["tipXcurv"]) / np.maximum(np.abs(d["absCurv"]), 1e-9)
        good = np.isfinite(tip)
        X = derive_np(d)
        good &= np.isfinite(X).all(axis=1)
        ng = int(good.sum())
        out["X"].append(X[good])
        out["y"].append(lab[good].astype(np.float32))
        out["nStubs"].append(nst[good])
        out["tip"].append(tip[good].astype(np.float32))
        out["tpPt"].append(np.where(lab, np.asarray(tpt[e], dtype=np.float32)[i1], -1.0)[good].astype(np.float32))
        out["tpEta"].append(np.where(lab, np.abs(np.asarray(tet[e], dtype=np.float32)[i1]), -1.0)[good].astype(np.float32))
        out["lay"].append(lay[good])
        out["ev"].append(np.full(ng, a + e, dtype=np.int32))
        out["tpKey"].append(np.where(lab, k1, -1).astype(np.int32)[good])
    empty = {"X": np.zeros((0, len(FEATS)), np.float32), "lay": np.zeros((0, 3), np.int16),
             "ev": np.zeros(0, np.int32), "tpKey": np.zeros(0, np.int32)}
    return {k: (np.concatenate(v, axis=0) if v else empty.get(k, np.zeros(0, np.float32)))
            for k, v in out.items()}


def load_arrays(specs, bank="displaced", label_mode="3of3", max_events=None, skip_events=0,
                chunk_events=8, quiet=False, workers=None):
    """CHUNKED, memory-bounded, PARALLEL ingestion. Returns a dict of numpy columns:

      X (n, nfeat) float32 | y (n,) float32 | nStubs, tip, tpPt, tpEta (n,) float32
      lay (n, 3) int16  | ev, tpKey (n,) int32 | sample (n,) int8 | sample_tags list

    Chunks are read by a process pool (see the WORKER-POOL SIZING RULE above) and reduced in strict
    chunk order, so the result is bit-identical for any --workers. The final concatenation is done
    COLUMN BY COLUMN with each column's chunk list freed as it goes, so the peak is steady state
    plus one column instead of the 2x of pd.concat. tpPt/tpEta/tpKey are the matched-TP quantities
    of hit h1 (meaningful for label==1 rows; -1 elsewhere); ev is a GLOBAL event index (offset per
    spec) so (ev, tpKey) identifies a TP."""
    table = TABLE_FOR_BANK[bank]
    workers = default_workers() if workers is None else workers
    cols = {k: [] for k in ("X", "y", "nStubs", "tip", "tpPt", "tpEta", "lay", "ev", "tpKey", "sample")}
    tags, ev_base = [], 0
    for si, sp in enumerate(specs):
        path, tag = sp.rsplit(":", 1)
        tags.append(tag)
        nall = uproot.open(path)["Events"].num_entries
        first = int(skip_events)
        last = nall if max_events is None else min(nall, first + int(max_events))
        jobs = [(path, table, a, min(a + chunk_events, last), label_mode)
                for a in range(first, last, chunk_events)]
        n_tot = n_real = 0

        def consume(r):
            nonlocal n_tot, n_real
            nrow = len(r["y"])
            if nrow == 0:
                return
            for k in ("X", "y", "nStubs", "tip", "tpPt", "tpEta", "lay", "tpKey"):
                cols[k].append(r[k])
            cols["ev"].append(r["ev"] + ev_base)
            cols["sample"].append(np.full(nrow, si, dtype=np.int8))
            n_tot += nrow
            n_real += int(r["y"].sum())

        if workers > 1:
            # imap, not imap_unordered: strict chunk order, so the reduce is deterministic.
            with _pool(workers) as pool:
                for r in pool.imap(_ingest_chunk, jobs, chunksize=1):
                    consume(r)
        else:
            for j in jobs:
                consume(_ingest_chunk(j))
        ev_base += nall
        if not quiet:
            print("  %15s: %9d triplets (%d real, %.4f) events [%d,%d) workers=%d"
                  % (tag, n_tot, n_real, n_real / max(1, n_tot), first, last, workers), flush=True)
    out = {}
    for k in list(cols):
        out[k] = np.concatenate(cols[k], axis=0)
        cols[k] = None  # free the chunk list immediately -- peak = steady + 1 column
    out["sample_tags"] = tags
    return out


def load(specs, bank="displaced", rng=17, label_mode="3of3", max_events=None):
    """Pandas view of load_arrays for external tooling. Builds the frame from the already
    contiguous numpy columns with copy=False and shuffles by index, so it never pays a
    pd.concat / DataFrame.sample double copy."""
    D = load_arrays(specs, bank=bank, label_mode=label_mode, max_events=max_events)
    data = {f: D["X"][:, i] for i, f in enumerate(FEATS)}
    for j, c in enumerate(LAYER_FEATURES):
        data[c] = D["lay"][:, j].astype(np.float32)
    data["label"] = D["y"]
    data["sample"] = np.asarray(D["sample_tags"], dtype=object)[D["sample"]]
    df = pd.DataFrame(data, copy=False)
    return df.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def event_split(ev, val_frac=0.1, seed=17):
    """Split at EVENT level (triplets of one event are strongly correlated; a row-level split
    leaks). Returns boolean masks (train, val)."""
    uev = np.unique(ev)
    rs = np.random.RandomState(seed)
    isval = rs.rand(len(uev)) < val_frac
    val_set = set(uev[isval].tolist())
    m_val = np.fromiter((e in val_set for e in ev), dtype=bool, count=len(ev))
    return ~m_val, m_val


# population-bias-aware weights
def group_ids(D):
    """(tip-group, pT-group) ids for every row; pT id is -1 for fakes (no matched TP)."""
    gt = bin_of(D["tip"], TIP_EDGES)
    gp = bin_of(D["tpPt"], PT_EDGES)
    gp = np.where(D["y"] > 0.5, gp, -1).astype(np.int16)
    return gt, gp


def marginal_factors(D, mask, alpha=1.0, min_count=200, verbose=True):
    """Raw (uncapped) inverse-frequency factor per |tip| group and per pT group, referenced to the
    MEDIAN group occupancy. Product-of-marginals rather than a raw 81-cell inverse frequency on
    purpose: the joint (|tip|,pT) cells in the far tails hold O(100) triplets, and a raw 1/N there
    is pure noise amplified by 10^5. The marginals are the two axes the loss was actually diagnosed
    on and each is populated to >=10^4 rows. alpha=1 is full inverse frequency, alpha=0.5 the
    classic sqrt / 'effective-number' softening (Cui et al., CVPR 2019)."""
    y = D["y"]
    gt, gp = group_ids(D)
    real = (y > 0.5) & mask
    out = []
    for g, ng, name in ((gt, NTIP, "tip"), (gp, NPT, "pT ")):
        cnt = np.bincount(g[real], minlength=ng).astype(np.float64)
        ok = cnt >= min_count
        ref = np.median(cnt[ok]) if ok.any() else max(cnt.max(), 1.0)
        f = (ref / np.maximum(cnt, min_count)) ** alpha
        if verbose:
            print("  [w] %s counts : %s" % (name, np.array2string(cnt, precision=0, max_line_width=250)),
                  flush=True)
            print("  [w] %s raw fac: %s" % (name, np.array2string(f, precision=2, max_line_width=250)),
                  flush=True)
        out.append(f)
    return out[0], out[1]


def apply_weights(D, mask, ft, fp, floor=0.2, cap=30.0, verbose=True, normalize="total"):
    """Per-sample weight for REAL triplets = clip(w_tip(i) * w_pT(j), floor, cap), renormalised
    so the TOTAL real weight is unchanged (only the DISTRIBUTION changes; fakes keep weight 1).

    The clip is on the PRODUCT: an uncapped product spans ~10^5, which would let a handful of
    >30 GeV triplets dominate every minibatch. The cap (~30) covers the per-group threshold
    deficit of an unweighted model: under weighted BCE a real-class weight lambda shifts the
    logit by log(lambda). The floor limits how far the over-served prompt/low-pT bulk is pushed
    down."""
    y = D["y"]
    gt, gp = group_ids(D)
    real = (y > 0.5) & mask
    w = np.ones(len(y), dtype=np.float32)
    prod = ft[gt[real]].astype(np.float64) * fp[gp[real]]
    prod /= max(np.median(prod), 1e-30)   # BULK -> 1, so floor/cap are relative to the bulk
    prod = np.clip(prod, floor, cap)
    w[real] = prod.astype(np.float32)
    if normalize == "total":
        # Rescale so the total real weight is unchanged, preserving the global real/fake balance. The
        # factor n_real/sum(w) is below 1 whenever the tails are upweighted, so it also downweights the
        # bulk; per-TP survival goes like recall^k with k about 10 real triplets per track, so a small
        # bulk recall loss costs more survival than the tails gain.
        w[real] *= float(real.sum()) / float(w[real].sum())
    # normalize == "bulk": leave the bulk at 1, so its real-vs-fake calibration is unchanged and the
    # weighting is purely additive on the tails. The total real weight rises, a global scale the
    # threshold absorbs.
    if verbose:
        q = np.percentile(w[real], [0, 1, 50, 90, 99, 100])
        print("  [w] real weights: min=%.3g p1=%.3g med=%.3g p90=%.3g p99=%.3g max=%.3g (mean %.3f)"
              % (q[0], q[1], q[2], q[3], q[4], q[5], w[real].mean()), flush=True)
    return w


class MLP(nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def scores_of(model, X, dev, bs=1 << 20):
    """Score a (n, nf) float32 numpy array or torch tensor, in blocks -> float32 numpy array."""
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            xb = xb if torch.is_tensor(xb) else t_of(xb)
            np_of(torch.sigmoid(model(xb.to(dev, non_blocking=True))).float(), out=out[i:i + bs])
    return out


def thr_at_recall(y, s, recall):
    real = np.sort(s[y == 1])
    if len(real) == 0:
        return 0.0
    return float(real[min(int((1.0 - recall) * len(real)), len(real) - 1)])


def fakerej_at_recall(y, s, recall=0.99):
    thr = thr_at_recall(y, s, recall)
    return 1.0 - (s[y == 0] >= thr).mean(), thr


def worst_group_threshold(y, s, gt, gp, recall=0.99, min_count=5000):
    """THE SUB-POPULATION THRESHOLD RULE. Returns the LARGEST threshold at which EVERY sufficiently
    populated (|tip| or pT) group of real triplets still keeps `recall` of its members, i.e.
    min over groups of that group's (1-recall) score quantile. A global recall rule instead
    prices the working point off the prompt / low-pT bulk."""
    thr, worst = np.inf, None
    for g, ng, nm in ((gt, NTIP, "tip"), (gp, NPT, "pT")):
        for i in range(ng):
            m = (y == 1) & (g == i)
            n = int(m.sum())
            if n < min_count:
                continue
            t = thr_at_recall(np.ones(n), s[m], recall)
            if t < thr:
                thr, worst = t, "%s[%d] n=%d" % (nm, i, n)
    if not np.isfinite(thr):
        thr = thr_at_recall(y, s, recall)
        worst = "global (no group above min_count)"
    return float(thr), worst


def group_recalls(y, s, g, ng, thr):
    """Per-group real-triplet recall. Single vectorised bincount pass (a per-group mask loop over
    O(10^8) rows dominates the scoring wall time)."""
    sel = y == 1
    gg = g[sel]
    cnt = np.bincount(gg, minlength=ng).astype(np.float64)
    num = np.bincount(gg, weights=(s[sel] >= thr).astype(np.float64), minlength=ng)
    return np.divide(num, cnt, out=np.full(ng, np.nan), where=cnt > 0), cnt


def tp_min_scores(D, s):
    """Per real TrackingParticle (key = global event index, tpKey): the MINIMUM score over all of its
    real triplets, plus that TP's group ids and real-triplet multiplicity.

    A TP 'survives' a gate at threshold t iff EVERY one of its real triplets passes, i.e. iff its
    minimum score >= t. Reducing to the min once makes survival a plain 1-D threshold problem: the
    survival-vs-threshold curve is the complementary CDF of these minima, and the threshold that
    delivers a target survival is just a quantile of them (no re-sort per threshold)."""
    m = D["y"] > 0.5
    ev = D["ev"][m].astype(np.int64)
    key = D["tpKey"][m].astype(np.int64)
    k64 = ev * (int(key.max()) + 2) + key
    order = np.argsort(k64, kind="stable")
    k64s = k64[order]
    ss = s[m][order]
    _, first, cnt = np.unique(k64s, return_index=True, return_counts=True)
    smin = np.minimum.reduceat(ss, first)
    gt = bin_of(D["tip"][m], TIP_EDGES)[order][first]
    gp = bin_of(D["tpPt"][m], PT_EDGES)[order][first]
    ge = bin_of(D["tpEta"][m], ETA_EDGES)[order][first]
    return {"smin": smin, "tip": gt, "pt": gp, "eta": ge, "k": bin_of(cnt.astype(np.float64), K_EDGES)}


def thr_at_survival(TP, target):
    """Largest threshold whose GLOBAL per-TP survival is >= target."""
    v = np.sort(TP["smin"])
    i = min(int((1.0 - target) * len(v)), len(v) - 1)
    return float(v[i])


def survival_ref(TP, floor=None, ref_json=None):
    """Build the per-group survival targets used by the default threshold rule.
    `ref_json` (a dict 'tip'/'pt' -> list) pins each group against a REFERENCE model's survival --
    used when the acceptance criterion is relative ("match or beat what we have today"). Otherwise
    every group gets the same absolute `floor`."""
    if ref_json:
        return {k: np.asarray(v, dtype=float) for k, v in ref_json.items()}
    return {"tip": np.full(NTIP, floor), "pt": np.full(NPT, floor)}


def worst_group_survival_threshold(TP, ref, min_count=2000):
    """WORST-GROUP SURVIVAL rule. `ref` maps 'tip'/'pt' -> per-group reference per-TP survivals.
    Returns the largest threshold at which EVERY sufficiently populated group's per-TP survival
    is at least its reference value.

    Why survival not per-triplet recall: a track is lost as soon as ONE of its real triplets is
    gated away, so per-TP survival ~ recall^k with k ~ 10 real triplets/TP. A 0.99 per-triplet
    recall floor is only ~0.99^10 ~ 0.90 per-TP survival. Equalising per-triplet recall at 0.99
    across groups hands the tails their recall by taking it back from the over-served prompt
    bulk (most of the population): no net efficiency. The rule is on the compounded quantity,
    which maps to MTV efficiency."""
    thr, worst = np.inf, None
    for nm, ng in (("tip", NTIP), ("pt", NPT)):
        g = TP[nm]
        for i in range(ng):
            m = g == i
            n = int(m.sum())
            if n < min_count or not np.isfinite(ref[nm][i]):
                continue
            v = np.sort(TP["smin"][m])
            t = float(v[min(int((1.0 - ref[nm][i]) * n), n - 1)])
            if t < thr:
                thr, worst = t, "%s[%d] n=%d ref_surv=%.4f" % (nm, i, n, ref[nm][i])
    if not np.isfinite(thr):
        thr, worst = 0.0, "none (no group above min_count)"
    return float(thr), worst


def survival_tables(TP, thr):
    """Per-TP survival at thr: globally and per |tip| / pT / |eta| / multiplicity-k group."""
    ok = (TP["smin"] >= thr).astype(np.float64)
    out = {"ALL": (float(ok.mean()), len(ok))}
    for nm, ng in (("tip", NTIP), ("pt", NPT), ("eta", len(ETA_EDGES) - 1), ("k", len(K_EDGES) - 1)):
        g = TP[nm]
        den = np.bincount(g, minlength=ng).astype(np.float64)
        num = np.bincount(g, weights=ok, minlength=ng)
        out[nm] = (np.divide(num, den, out=np.full(ng, np.nan), where=den > 0), den)
    return out


def per_tp_survival(D, s, thr, mask=None):
    """Survival tables at a single threshold (wrapper over tp_min_scores + survival_tables)."""
    return survival_tables(tp_min_scores(D, s), thr)


def _fmt_edges(edges):
    return ["[%g,%s)" % (edges[i], "inf" if edges[i + 1] >= 1e8 else "%g" % edges[i + 1])
            for i in range(len(edges) - 1)]


def report_bins(title, edges, rec, cnt, extra=None, extra_name=""):
    print("\n%s" % title)
    hdr = "  %-14s %12s %9s" % ("bin", "n_real", "recall")
    if extra is not None:
        hdr += " %9s" % extra_name
    print(hdr)
    for i, lab in enumerate(_fmt_edges(edges)):
        line = "  %-14s %12.0f %9.4f" % (lab, cnt[i], rec[i])
        if extra is not None:
            line += " %9.4f" % extra[i]
        print(line)


def evaluate(D, s, thr, tag="", full=True, TP=None):
    """Full operating report of a scored dataset at a single deployed threshold."""
    y = D["y"]
    gt, gp = group_ids(D)
    ge = bin_of(D["tpEta"], ETA_EDGES)
    gn = bin_of(D["nStubs"], NS_EDGES)
    real, fake = y == 1, y == 0
    rec = float((s[real] >= thr).mean())
    rej = float(1.0 - (s[fake] >= thr).mean())
    print("\n===== [%s] thr=%.6g : recall=%.4f fakeRej=%.4f  (real=%d fake=%d) ====="
          % (tag, thr, rec, rej, real.sum(), fake.sum()), flush=True)
    if not full:
        return {"thr": thr, "recall": rec, "fakeRej": rej}
    res = {"thr": thr, "recall": rec, "fakeRej": rej}
    for nm, g, edges in (("tip", gt, TIP_EDGES), ("pt", gp, PT_EDGES),
                         ("eta", ge, ETA_EDGES), ("nstubs", gn, NS_EDGES)):
        r, c = group_recalls(y, s, g, len(edges) - 1, thr)
        res["recall_" + nm] = r.tolist()
        res["n_" + nm] = c.tolist()
        report_bins("REAL-TRIPLET RECALL vs %s" % nm, edges, r, c)
    sv = survival_tables(TP if TP is not None else tp_min_scores(D, s), thr)
    res["survival_all"] = sv["ALL"][0]
    print("\nPER-TP SURVIVAL (all real triplets of the TP pass): ALL = %.4f  (nTP=%d)"
          % (sv["ALL"][0], sv["ALL"][1]))
    for nm, edges in (("tip", TIP_EDGES), ("pt", PT_EDGES), ("eta", ETA_EDGES), ("k", K_EDGES)):
        r, c = sv[nm]
        res["survival_" + nm] = r.tolist()
        res["nTP_" + nm] = c.tolist()
        report_bins("PER-TP SURVIVAL vs %s" % nm, edges, r, c)
    return res


def standardize_inplace(X, mean, std):
    for i in range(0, len(X), 1 << 23):
        X[i:i + (1 << 23)] -= mean
        X[i:i + (1 << 23)] /= std


def masked_mean_std(X, mask, ncol, block=1 << 23):
    """Column mean/std over the masked rows WITHOUT materialising X[mask] (a boolean fancy-index of
    the O(10 GiB) matrix costs a full copy, and several concurrent trainings of that exhaust the
    cgroup). Streams in row blocks and accumulates first and second moments in float64."""
    n = 0
    s1 = np.zeros(ncol, dtype=np.float64)
    s2 = np.zeros(ncol, dtype=np.float64)
    for i in range(0, len(X), block):
        m = mask[i:i + block]
        if not m.any():
            continue
        b = X[i:i + block, :ncol][m].astype(np.float64)
        n += len(b)
        s1 += b.sum(axis=0)
        s2 += (b * b).sum(axis=0)
        del b
    mean = s1 / max(n, 1)
    std = np.sqrt(np.maximum(s2 / max(n, 1) - mean * mean, 0.0))
    std[std == 0] = 1.0
    return mean, std


def train_variant(name, D, dev, args):
    use_onehot, loss_kind, wmode = VARIANTS[name]
    assert not use_onehot or "lay" in D, "one-hot needs layer columns"
    m_tr, m_va = event_split(D["ev"], val_frac=args.val_frac, seed=args.seed)
    y = D["y"]
    X = D["X"]
    if use_onehot:
        cats = [np.unique(D["lay"][:, j]) for j in range(3)]
        oh = np.concatenate([(D["lay"][:, j:j + 1] == c[None, :]).astype(np.float32)
                             for j, c in enumerate(cats)], axis=1)
        X = np.concatenate([X, oh], axis=1)
        del oh
    else:
        cats = [np.unique(D["lay"][:, j]) for j in range(3)]

    nf_cont = len(FEATS)
    mean, std = masked_mean_std(X, m_tr, nf_cont)
    standardize_inplace(X[:, :nf_cont], mean.astype(np.float32), std.astype(np.float32))

    pos_w = min(float((y[m_tr] == 0).sum() / max(1, (y[m_tr] == 1).sum())), 50.0)
    floor, cap = args.weight_floor, args.weight_cap
    if wmode in ("cell1", "cell05"):
        ft, fp = marginal_factors(D, m_tr, alpha=1.0 if wmode == "cell1" else 0.5)
        w = apply_weights(D, m_tr, ft, fp, floor, cap, normalize=args.weight_normalize)
    elif wmode == "dro":
        ft, fp = np.ones(NTIP), np.ones(NPT)
        w = apply_weights(D, m_tr, ft, fp, floor, cap, verbose=False,
                          normalize=args.weight_normalize)
    else:
        ft, fp = np.ones(NTIP), np.ones(NPT)
        w = np.ones(len(y), dtype=np.float32)
    w = w.astype(np.float32)
    w[y > 0.5] *= pos_w  # fold pos_weight in -> wmode='none' is plain class-balanced BCE

    idx_tr = np.flatnonzero(m_tr).astype(np.int64)
    idx_va = np.flatnonzero(m_va).astype(np.int64)
    print("[%s] nfeat=%d train=%d val=%d pos_w=%.1f wmode=%s dev=%s"
          % (name, X.shape[1], len(idx_tr), len(idx_va), pos_w, wmode, dev), flush=True)

    gpu = dev.startswith("cuda") and torch.cuda.is_available()
    nbytes = X.nbytes / 2**30
    resident = gpu and nbytes < args.gpu_resident_gb
    print("[%s] X = %.1f GiB -> %s" % (name, nbytes, "GPU-resident" if resident else "host-streamed"),
          flush=True)
    Xd = t_of(X).to(dev) if resident else t_of(X)
    yd = t_of(y).to(dev)
    wd = t_of(w).to(dev)
    itr = t_of(idx_tr).to(dev)
    iva = t_of(idx_va).to(dev)

    torch.manual_seed(args.seed)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    y_va = y[idx_va]
    gt_all, gp_all = group_ids(D)
    gt_va, gp_va = gt_all[idx_va], gp_all[idx_va]

    hist = {"train_loss": [], "val_rej99": [], "val_metric": [], "val_thr": [], "lr": []}
    best = {"metric": -1.0, "epoch": -1, "state": None}
    n, bs = len(itr), args.batch
    for ep in range(args.max_epochs):
        model.train()
        perm = torch.randperm(n, device=itr.device)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            sel = itr[perm[i:i + bs]]
            xb = Xd[sel].to(dev, non_blocking=True) if not resident else Xd[sel]
            yb, wb = yd[sel], wd[sel]
            logits = model(xb)
            if loss_kind == "wbce":
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            else:  # focal, gamma=2
                ce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                p = torch.sigmoid(logits)
                pt = p * yb + (1 - p) * (1 - yb)
                loss = ((1 - pt) ** 2 * ce * wb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        s_va = scores_of(model, Xd[iva] if resident else X[idx_va], dev)
        rej99, _ = fakerej_at_recall(y_va, s_va, args.recall)
        thr_wg, worst = worst_group_threshold(y_va, s_va, gt_va, gp_va, args.recall, args.group_min_count)
        metric = float(1.0 - (s_va[y_va == 0] >= thr_wg).mean())
        hist["train_loss"].append(tot / max(1, nb))
        hist["val_rej99"].append(float(rej99))
        hist["val_metric"].append(metric)
        hist["val_thr"].append(thr_wg)
        hist["lr"].append(opt.param_groups[0]["lr"])
        sel_metric = metric if args.select_metric == "worstgroup" else float(rej99)
        sched.step(sel_metric)
        marker = ""
        if sel_metric > best["metric"]:
            best = {"metric": sel_metric, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            marker = " *"
        print("[%s] ep%3d loss=%.4f val_rej99=%.4f  worstgroup thr=%.5g rej=%.4f (%s) lr=%.2e%s"
              % (name, ep, hist["train_loss"][-1], rej99, thr_wg, metric, worst, hist["lr"][-1], marker),
              flush=True)
        if wmode == "dro" and ep < args.dro_freeze_epoch:
            # The DRO objective is non-stationary: the weights move every epoch, so the validation
            # metric is not comparable across epochs and early stopping on it is ill-defined. The weights
            # ramp for a fixed number of epochs, then freeze, and the run converges against a fixed
            # objective. The update is exponentiated-gradient group reweighting on the recall deficit at
            # the common threshold, the group-DRO update of Sagawa et al. (ICLR 2020) specialised to a
            # per-group recall floor; groups already at the floor are untouched.
            t_gl = thr_at_recall(y_va, s_va, args.recall)
            rt, _ = group_recalls(y_va, s_va, gt_va, NTIP, t_gl)
            rp, _ = group_recalls(y_va, s_va, gp_va, NPT, t_gl)
            ft = np.clip(ft * np.exp(args.dro_eta * np.nan_to_num(args.recall - rt, nan=0.0).clip(0)),
                         1.0, args.weight_cap)
            fp = np.clip(fp * np.exp(args.dro_eta * np.nan_to_num(args.recall - rp, nan=0.0).clip(0)),
                         1.0, args.weight_cap)
            w = apply_weights(D, m_tr, ft, fp, floor, cap, verbose=False,
                              normalize=args.weight_normalize).astype(np.float32)
            w[y > 0.5] *= pos_w
            wd = t_of(w).to(dev)
            print("  [dro] tip=%s pT=%s" % (np.array2string(ft, precision=2, max_line_width=250),
                                            np.array2string(fp, precision=2, max_line_width=250)),
                  flush=True)
        if ep - best["epoch"] >= args.patience:
            print("[%s] early stop at ep%d (best ep%d metric=%.4f)"
                  % (name, ep, best["epoch"], best["metric"]), flush=True)
            break

    model.load_state_dict(best["state"])
    res = {"variant": name, "use_onehot": use_onehot, "loss": loss_kind, "weight_mode": wmode,
           "nfeat": int(X.shape[1]), "feats": list(FEATS), "best_epoch": best["epoch"],
           "val_metric": best["metric"],
           "val_rej99": hist["val_rej99"][best["epoch"]],
           "val_worstgroup_rej": hist["val_metric"][best["epoch"]],
           "val_worstgroup_thr": hist["val_thr"][best["epoch"]],
           "pos_w": pos_w, "weight_cap": args.weight_cap, "weight_floor": args.weight_floor,
           "select_metric": args.select_metric,
           "recall": args.recall, "group_min_count": args.group_min_count,
           "w_tip": ft.tolist(), "w_pt": fp.tolist(),
           "categories": [c.tolist() for c in cats],
           "train_events": [int(D["ev"].min()), int(D["ev"].max())]}
    os.makedirs(PLOTDIR, exist_ok=True)
    torch.save({"state": best["state"], "mean": mean, "std": std, "res": res},
               os.path.join(PLOTDIR, "model_%s.pt" % name))
    with open(os.path.join(PLOTDIR, "result_%s.json" % name), "w") as f:
        json.dump({**res, "hist": hist}, f)
    plot_training(name, hist)
    del Xd, yd, wd, itr, iva
    if gpu:
        torch.cuda.empty_cache()

    # End-of-training scoring (train + val): the full per-(|tip|,pT,|eta|,nStubs) recall tables,
    # per-TP survival and fake rejection, pinned by the same sub-population rule that pins the
    # deployed threshold. This makes a population-biased working point visible at training time.
    s_full = scores_of(model, X, dev)   # one blocked GPU pass; X is never fancy-indexed (full copy)
    for split_name, idx in (("val", idx_va), ("train", idx_tr)):
        Dv = {k: D[k][idx] for k in ("y", "tip", "tpPt", "tpEta", "nStubs", "ev", "tpKey")}
        sv = s_full[idx]
        TPv = tp_min_scores(Dv, sv)
        ref = survival_ref(TPv, floor=args.survival_floor)
        t, wb = worst_group_survival_threshold(TPv, ref, max(500, args.group_min_count // 4))
        print("\n[%s] %s split: worst-group survival rule (floor %.4g) -> thr=%.6g (binding: %s)"
              % (name, split_name, args.survival_floor, t, wb), flush=True)
        res["eval_" + split_name] = evaluate(Dv, sv, t, tag="%s/%s" % (name, split_name), TP=TPv)
        del Dv, sv, TPv
    del s_full
    with open(os.path.join(PLOTDIR, "result_%s.json" % name), "w") as f:
        json.dump({**res, "hist": hist}, f)
    print("[%s] DONE val_metric=%.4f (ep %d)" % (name, best["metric"], best["epoch"]), flush=True)
    return res


def plot_training(name, hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].set_title("%s: loss" % name)
    ax[1].plot(hist["val_rej99"], label="global rej@99")
    ax[1].plot(hist["val_metric"], label="rej @ worst-group rule")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val fake rejection"); ax[1].legend()
    ax[1].set_title("%s: val metric" % name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, "training_%s.png" % name), dpi=110)
    plt.close(fig)


def final_plots(name, y, s, sample, thr):
    from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                                 precision_recall_curve, roc_curve, auc)
    os.makedirs(PLOTDIR, exist_ok=True)
    sub = np.random.RandomState(5).choice(len(y), min(len(y), 20_000_000), replace=False)
    y, s, sample = y[sub], s[sub], sample[sub]
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label="AUC=%.4f" % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("fake efficiency (FPR)"); ax.set_ylabel("real efficiency (TPR)")
    ax.set_title("%s: ROC (test)" % name); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "final_roc.png"), dpi=110); plt.close(fig)
    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(rec, prec)
    ax.set_xlabel("recall (real)"); ax.set_ylabel("precision"); ax.set_title("%s: PR (test)" % name)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "final_precision_recall.png"), dpi=110)
    plt.close(fig)
    cm = confusion_matrix(y, (s >= thr).astype(int), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["fake", "real"]).plot(ax=ax, values_format=".3f",
                                                                    colorbar=False)
    ax.set_title("%s: confusion @thr=%.4g (row-norm)" % (name, thr))
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "final_confusion.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(0, 1, 80)
    ax.hist(s[y == 0], bins=bins, density=True, alpha=0.6, label="fake")
    ax.hist(s[y == 1], bins=bins, density=True, alpha=0.6, label="real")
    ax.axvline(thr, color="k", ls="--", lw=1, label="thr=%.4g" % thr)
    ax.set_yscale("log"); ax.set_xlabel("DNN score"); ax.legend(); ax.set_title("%s: scores" % name)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "final_scores.png"), dpi=110); plt.close(fig)


def _t2np(t):
    """torch tensor -> float32 ndarray via tolist(): the CMSSW torch build lacks NumPy interop."""
    return np.asarray(t.cpu().tolist(), dtype=np.float32)


def fmt_array(name, arr, ctype="float"):
    flat = np.asarray(arr, dtype=np.float64).ravel()
    body = ", ".join("%.8ef" % v for v in flat)
    return "  HOST_DEVICE_CONSTANT %s %s[%d] = {%s};\n" % (ctype, name, flat.size, body)


def export_header(out_path, model_state, mean, std, thr, nfeat, bank="displaced", note=""):
    # bank = "displaced" | "prompt" : per-iteration weight bank. Each bank is its own
    # auto-generated header (namespace caTripletDNN_<bank>) so prompt and displaced bakes
    # never clobber each other; the dispatcher CATripletDNNWeights.h includes both.
    assert bank in ("displaced", "prompt"), bank
    guard = "RecoTracker_PixelSeeding_plugins_alpaka_CATripletDNNWeights_%s_h" % bank
    ns = "caTripletDNN_%s" % bank
    w1 = _t2np(model_state["net.0.weight"]).T  # (nfeat, 64)
    b1 = _t2np(model_state["net.0.bias"])
    w2 = _t2np(model_state["net.2.weight"]).T
    b2 = _t2np(model_state["net.2.bias"])
    w3 = _t2np(model_state["net.4.weight"]).ravel()
    b3 = _t2np(model_state["net.4.bias"])
    h1, h2 = w1.shape[1], w2.shape[1]
    with open(out_path, "w") as f:
        f.write("#ifndef %s\n" % guard)
        f.write("#define %s\n\n" % guard)
        f.write("// AUTO-GENERATED by test/models/train_triplet_dnn.py --bank %s -- do not edit by hand.\n" % bank)
        f.write("// %d features (%d raw + %d derived), torch-trained (weighted/focal loss).\n"
                % (nfeat, min(nfeat, len(BASE_FEATURES)), max(0, nfeat - len(BASE_FEATURES))))
        for line in note.splitlines():
            f.write("// %s\n" % line)
        f.write("\n#include <alpaka/alpaka.hpp>\n\n")
        f.write("#include \"FWCore/Utilities/interface/HostDeviceConstant.h\"\n\n")
        f.write("namespace ALPAKA_ACCELERATOR_NAMESPACE::%s {\n\n" % ns)
        f.write("  constexpr int kNFeat = %d;\n" % nfeat)
        f.write("  constexpr int kNH1 = %d;\n" % h1)
        f.write("  constexpr int kNH2 = %d;\n" % h2)
        f.write("  constexpr float kDefaultThreshold = %.6ef;\n\n" % thr)
        f.write(fmt_array("kMean", mean))
        f.write(fmt_array("kScale", std))
        f.write(fmt_array("kW1", w1))
        f.write(fmt_array("kB1", b1))
        f.write(fmt_array("kW2", w2))
        f.write(fmt_array("kB2", b2))
        f.write(fmt_array("kW3", w3))
        f.write(fmt_array("kB3", b3))
        f.write("\n}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::%s\n\n#endif\n" % ns)


_SCORE = {}


def _score_block(rng):
    i, j = rng
    c = _SCORE
    x = ((c["X"][i:j, :c["nf"]] - c["mean"]) / c["std"]).astype(np.float32)
    h = np.maximum(x @ c["w1"] + c["b1"], 0)
    h = np.maximum(h @ c["w2"] + c["b2"], 0)
    z = (h @ c["w3"] + c["b3"]).astype(np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))).astype(np.float32)


def score_dataset(model_state, mean, std, Xraw, workers=None, block=1 << 22):
    """Parallel float32 forward, identical arithmetic to numpy_forward (which CATripletDNN.h
    mirrors). Blocks are reduced in strict index order, so the output is bit-identical for any
    --workers. Uses a fork pool: the feature matrix stays shared copy-on-write, so the only
    per-worker transient is one block's hidden activations (~0.5 GB per 4 M rows).

    Xraw may be WIDER than the model: only its leading net.0.weight.shape[1] columns are read.
    That is the append-only feature ABI (CATripletDNNWeights.h) on the offline side, and it is what
    lets a 29-feature model (e.g. the deployed bank) be scored on a 31-feature dataset."""
    workers = default_workers() if workers is None else workers
    _SCORE.clear()
    _SCORE.update({"X": Xraw, "nf": int(model_state["net.0.weight"].shape[1]),
                   "mean": mean.astype(np.float32), "std": std.astype(np.float32),
                   "w1": _t2np(model_state["net.0.weight"]).T, "b1": _t2np(model_state["net.0.bias"]),
                   "w2": _t2np(model_state["net.2.weight"]).T, "b2": _t2np(model_state["net.2.bias"]),
                   "w3": _t2np(model_state["net.4.weight"]).ravel(),
                   "b3": _t2np(model_state["net.4.bias"])[0]})
    rngs = [(i, min(i + block, len(Xraw))) for i in range(0, len(Xraw), block)]
    if workers <= 1 or len(rngs) == 1:
        parts = [_score_block(r) for r in rngs]
    else:
        with _pool(workers) as pool:
            parts = list(pool.imap(_score_block, rngs, chunksize=1))
    _SCORE.clear()
    return np.concatenate(parts)


def numpy_forward(model_state, mean, std, Xraw, bs=1 << 22):
    """float32 forward exactly as CATripletDNN.h evaluates it (scaler + relu + relu + sigmoid)."""
    w1 = _t2np(model_state["net.0.weight"]).T
    b1 = _t2np(model_state["net.0.bias"])
    w2 = _t2np(model_state["net.2.weight"]).T
    b2 = _t2np(model_state["net.2.bias"])
    w3 = _t2np(model_state["net.4.weight"]).ravel()
    b3 = _t2np(model_state["net.4.bias"])
    mean32, std32 = mean.astype(np.float32), std.astype(np.float32)
    nf = w1.shape[0]  # leading columns only (append-only feature ABI); see score_dataset
    out = np.empty(len(Xraw), dtype=np.float32)
    for i in range(0, len(Xraw), bs):
        x = ((Xraw[i:i + bs, :nf] - mean32) / std32).astype(np.float32)
        h = np.maximum(x @ w1 + b1, 0)
        h = np.maximum(h @ w2 + b2, 0)
        z = (h @ w3 + b3[0]).astype(np.float64)
        out[i:i + bs] = (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "finalize", "eval"])
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--variant", default="derived_wbce")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bank", choices=["displaced", "prompt"], default="displaced",
                    help="per-iteration weight bank: selects the nano table (TrkDispTriplet / "
                         "TrkPromptTriplet) AND the baked header (default: displaced)")
    ap.add_argument("--out", default=None,
                    help="output header path (default: ../plugins/alpaka/CATripletDNNWeights_<bank>.h)")
    ap.add_argument("--recall", type=float, default=0.99)
    ap.add_argument("--label-mode", choices=["3of3", "2of3"], default="3of3",
                    help="truth-label rule: '3of3' (all three hits share a TP, default) or '2of3'")
    ap.add_argument("--max-events", type=int, default=None,
                    help="use only N events starting at --skip-events (event-level, bounds memory)")
    ap.add_argument("--skip-events", type=int, default=0,
                    help="first event to read (with --max-events gives event-DISJOINT train/test sets)")
    ap.add_argument("--chunk-events", type=int, default=8,
                    help="events per uproot read; the ingestion transient scales with this")
    ap.add_argument("--mem-guard-gb", type=float, default=260.0,
                    help="abort if the cgroup ANON footprint exceeds this (cgroup cap here is 360 G; "
                         "memory.current also carries ~160 G of reclaimable NFS page cache and is logged)")
    ap.add_argument("--mem-log", default=None)
    # objective / threshold rule
    ap.add_argument("--weight-cap", type=float, default=30.0,
                    help="upper clip of the per-sample real-triplet weight PRODUCT (variance control; "
                         "30 = the measured per-group threshold deficit of the deployed model)")
    ap.add_argument("--weight-floor", type=float, default=0.5,
                    help="lower clip of the per-sample real-triplet weight product (1.0 = never "
                         "downweight any group, i.e. tail-only upweighting)")
    ap.add_argument("--weight-normalize", choices=["total", "bulk"], default="total",
                    help="'total' preserves the summed real weight (legacy; it downweights the bulk "
                         "by whatever the tails gain, which costs more per-TP survival than the "
                         "tails gain). 'bulk' anchors the bulk at weight 1 so the bulk keeps "
                         "today's real-vs-fake calibration exactly.")
    ap.add_argument("--dro-freeze-epoch", type=int, default=10,
                    help="stop updating the DRO group weights after this epoch, so the objective "
                         "becomes stationary and early stopping is well defined")
    ap.add_argument("--dro-eta", type=float, default=12.0,
                    help="exponentiated-gradient step on the per-group recall deficit (DRO arm)")
    ap.add_argument("--select-metric", choices=["rej99", "worstgroup"], default="rej99",
                    help="epoch/variant selection metric: legacy global rej@99 or fake rejection "
                         "at the worst-group threshold")
    ap.add_argument("--threshold-rule",
                    choices=["worstgroup-survival", "worstgroup-recall", "global"],
                    default="worstgroup-survival",
                    help="DEFAULT 'worstgroup-survival': the largest threshold at which EVERY "
                         "populated (|tip|,pT) group's per-TP SURVIVAL clears its target. "
                         "'worstgroup-recall' uses per-triplet recall instead (does not account for "
                         "the recall^k compounding). 'global' is the LEGACY single global "
                         "target-recall point -- it does not guard sparsely populated groups and is "
                         "kept only for comparison.")
    ap.add_argument("--survival-floor", type=float, default=0.90,
                    help="per-group per-TP survival target for the default threshold rule")
    ap.add_argument("--ref-survival-json", default=None,
                    help="JSON with per-group reference survivals {'tip':[...], 'pt':[...]}; when "
                         "given, each group must match or beat ITS reference instead of the flat "
                         "floor (relative acceptance criteria)")
    ap.add_argument("--workers", type=int, default=None,
                    help="process-pool size for the CPU-bound chunked passes (ingestion, scoring). "
                         "Default is cap-sized: min(ncpu-2, MEM_BUDGET_GB/PER_WORKER_GB, 24). "
                         "Results are reduced in strict chunk order and are identical for any value.")
    ap.add_argument("--bake-thr", type=float, default=None,
                    help="finalize: bake THIS threshold verbatim (used when the working point was "
                         "pinned by an external rule, e.g. the worst-group SURVIVAL rule, whose "
                         "reference survivals come from another model)")
    ap.add_argument("--group-min-count", type=int, default=2000,
                    help="minimum population for a (|tip| or pT) group to bind the worst-group rule "
                         "(TPs for the survival rule, real triplets for the recall rule)")
    ap.add_argument("--eval-skip-events", type=int, default=None,
                    help="finalize/eval: score an event-DISJOINT held-out range starting here")
    ap.add_argument("--eval-max-events", type=int, default=None)
    ap.add_argument("--force-variant", default=None,
                    help="finalize: bake THIS variant instead of the val-metric winner (used when a "
                         "large event-disjoint holdout ranks the arms differently from val)")
    ap.add_argument("--model", default=None, help="eval mode: path to a model_<variant>.pt")
    ap.add_argument("--eval-thr", type=float, default=None, help="eval mode: threshold to report at")
    ap.add_argument("--header", default=None, help="eval mode: score a BAKED header instead of a .pt")
    ap.add_argument("--json-out", default=None)
    # schedule
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--nfeat", type=int, default=LEGACY_NFEAT,
                    help="width of the feature vector (APPEND-ONLY ABI: the leading n of "
                         "BASE_FEATURES+DERIVED+DERIVED_EXT). Default %d = the legacy vector, so "
                         "every pre-existing caller is unchanged; %d adds logTip/logPt. In eval "
                         "mode with --header it defaults to the header's own kNFeat."
                         % (LEGACY_NFEAT, len(ALL_FEATS)))
    ap.add_argument("--drop-features", default="",
                    help="comma-separated feature NAMES to remove from the active vector, applied "
                         "after --nfeat (default: none, so every pre-existing caller is unchanged). "
                         "Used by ablation studies. A dropped feature needs NO kernel change to "
                         "deploy: zero its kW1 column in the baked header and fold the constant "
                         "into kB1 -- layer 1 is affine in the standardized inputs.")
    ap.add_argument("--gpu-resident-gb", type=float, default=40.0)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../plugins/alpaka/CATripletDNNWeights_%s.h" % args.bank)

    # Feature-vector width, chosen before ingestion so the matrix is built at that width, with no
    # post-hoc slice and no second copy, and the fork workers inherit it. With eval --header it defaults
    # to the header's own kNFeat.
    if args.mode == "eval" and args.header and "--nfeat" not in sys.argv:
        import re as _re
        args.nfeat = int(_re.search(r"kNFeat\s*=\s*(\d+)", open(args.header).read()).group(1))
    set_nfeat(args.nfeat)
    if args.drop_features:
        drop_features([x.strip() for x in args.drop_features.split(",") if x.strip()])
    print("feature vector: %d columns (%s ... %s)" % (len(FEATS), FEATS[0], FEATS[-1]), flush=True)

    start_mem_guard(args.mem_guard_gb, log=args.mem_log)
    t0 = time.time()
    print("loading dataset... (bank=%s, label_mode=%s, events [%s,+%s), chunk=%d)"
          % (args.bank, args.label_mode, args.skip_events, args.max_events, args.chunk_events),
          flush=True)
    D = load_arrays(args.specs, bank=args.bank, label_mode=args.label_mode,
                    max_events=args.max_events, skip_events=args.skip_events,
                    chunk_events=args.chunk_events, workers=args.workers)
    print("  loaded %d triplets in %.0f s (X = %.1f GiB, rss %.0f GiB)"
          % (len(D["y"]), time.time() - t0, D["X"].nbytes / 2**30, _self_rss_gb()), flush=True)

    if args.mode == "train":
        train_variant(args.variant, D, args.device, args)
        return

    if args.mode == "eval":
        if args.header:
            import re
            txt = open(args.header).read()
            W = {}
            for nm in ("kMean", "kScale", "kW1", "kB1", "kW2", "kB2", "kW3", "kB3"):
                m = re.search(r"%s\[\d+\]\s*=\s*\{(.*?)\};" % nm, txt, re.S)
                W[nm] = np.array([float(v) for v in re.findall(r"[-+0-9.eE]+(?=f)", m.group(1))],
                                 dtype=np.float32)
            nf = int(re.search(r"kNFeat\s*=\s*(\d+)", txt).group(1))
            thr_hdr = float(re.search(r"kDefaultThreshold\s*=\s*([-+0-9.eE]+)f", txt).group(1))
            state = {"net.0.weight": t_of(np.ascontiguousarray(W["kW1"].reshape(nf, -1).T)).clone(),
                     "net.0.bias": t_of(W["kB1"]).clone(),
                     "net.2.weight": t_of(np.ascontiguousarray(W["kW2"].reshape(W["kB1"].size, -1).T)).clone(),
                     "net.2.bias": t_of(W["kB2"]).clone(),
                     "net.4.weight": t_of(np.ascontiguousarray(W["kW3"].reshape(1, -1))).clone(),
                     "net.4.bias": t_of(W["kB3"]).clone()}
            mean, std = W["kMean"], W["kScale"]
            name = os.path.basename(args.header)
        else:
            ck = torch.load(args.model, weights_only=False)
            state, mean, std = ck["state"], ck["mean"], ck["std"]
            thr_hdr = ck["res"].get("val_worstgroup_thr", 0.0)
            name = ck["res"]["variant"]
        s = score_dataset(state, mean, std, D["X"], workers=args.workers)
        gt, gp = group_ids(D)
        thr_wg, worst = worst_group_threshold(D["y"], s, gt, gp, args.recall, args.group_min_count)
        thr_gl = thr_at_recall(D["y"], s, args.recall)
        print("[%s] threshold rules on this set: global-recall%.3g -> %.6g | worst-group -> %.6g (%s)"
              % (name, args.recall, thr_gl, thr_wg, worst), flush=True)
        out = {"name": name, "thr_global": thr_gl, "thr_worstgroup": thr_wg, "worst_group": worst}
        for lbl, t in (("header", thr_hdr), ("global", thr_gl), ("worstgroup", thr_wg)):
            if args.eval_thr is not None and lbl != "header":
                continue
            out[lbl] = evaluate(D, s, t if args.eval_thr is None or lbl != "header" else args.eval_thr,
                                tag="%s/%s" % (name, lbl))
        if args.eval_thr is not None:
            out["custom"] = evaluate(D, s, args.eval_thr, tag="%s/custom" % name)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(out, f)
        return

    # Finalize.
    results = {}
    for nm in VARIANTS:
        p = os.path.join(PLOTDIR, "result_%s.json" % nm)
        if os.path.exists(p):
            results[nm] = json.load(open(p))
    assert results, "no trained variants found"
    key = "val_metric" if args.select_metric == "worstgroup" else "val_rej99"
    if args.force_variant:
        # The bake-off scores arms on a large event-disjoint holdout; when that ranking differs from the
        # validation ranking the holdout wins.
        assert args.force_variant in results, "no result for %s" % args.force_variant
        results = {args.force_variant: results[args.force_variant]}
    for n, r in sorted(results.items(), key=lambda kv: -kv[1].get(key, -1)):
        print("  %15s: val_rej99=%.4f val_worstgroup_rej=%.4f (wmode=%s)"
              % (n, r.get("val_rej99", float("nan")), r.get("val_worstgroup_rej", float("nan")),
                 r.get("weight_mode", "none")))
    best_simple = max((r for r in results.values() if not r["use_onehot"]), key=lambda r: r.get(key, -1))
    best_any = max(results.values(), key=lambda r: r.get(key, -1))
    winner = best_any if (best_any["use_onehot"] and
                          best_any.get(key, -1) - best_simple.get(key, -1) > 0.015) else best_simple
    name = winner["variant"]
    print("WINNER: %s (deployable preference: onehot only if >1.5pts better)" % name, flush=True)
    assert not winner["use_onehot"], \
        "one-hot variant won by >1.5 pts; kernel wiring for one-hot (W1 row adds) required -- not automated here"

    ck = torch.load(os.path.join(PLOTDIR, "model_%s.pt" % name), weights_only=False)
    state, mean, std = ck["state"], ck["mean"], ck["std"]
    s = score_dataset(state, mean, std, D["X"], workers=args.workers)
    gt, gp = group_ids(D)
    if args.bake_thr is not None:
        thr = float(args.bake_thr)
        print("threshold = EXTERNALLY PINNED %.6g (chosen in situ, outside this script's threshold rules)" % thr,
              flush=True)
        # An externally pinned working point is an in-situ decision, on axes an offline scorer cannot
        # see. What the offline rule would have chosen is printed for comparison; `thr` is unchanged.
        try:
            _TPh = tp_min_scores(D, s)
            _t2, _w2 = worst_group_survival_threshold(
                _TPh, survival_ref(_TPh, floor=args.survival_floor), args.group_min_count)
            print("  [compare] the offline worst-group SURVIVAL rule would have pinned thr=%.6g "
                  "(binding: %s); pinned/offline = %.3gx" % (_t2, _w2, thr / max(_t2, 1e-12)),
                  flush=True)
        except Exception as _e:  # never let a diagnostic block a bake
            print("  [compare] offline-rule comparison unavailable (%s)" % _e, flush=True)
    elif args.threshold_rule == "worstgroup-survival":
        TPh = tp_min_scores(D, s)
        ref = survival_ref(TPh, floor=args.survival_floor,
                           ref_json=json.load(open(args.ref_survival_json))
                           if args.ref_survival_json else None)
        thr, worst = worst_group_survival_threshold(TPh, ref, args.group_min_count)
        print("threshold rule = WORST-GROUP per-TP SURVIVAL target (%s) in every (|tip|,pT) group "
              "with >=%d TPs -> thr=%.6g (binding: %s)"
              % ("reference JSON" if args.ref_survival_json else "floor %.4g" % args.survival_floor,
                 args.group_min_count, thr, worst), flush=True)
    elif args.threshold_rule == "worstgroup-recall":
        thr, worst = worst_group_threshold(D["y"], s, gt, gp, args.recall, args.group_min_count)
        print("threshold rule = WORST-GROUP per-triplet recall>=%.4g in every (|tip|,pT) group with "
              ">=%d real triplets -> thr=%.6g (binding group: %s)"
              % (args.recall, args.group_min_count, thr, worst), flush=True)
    else:
        thr = thr_at_recall(D["y"], s, args.recall)
        print("threshold rule = GLOBAL recall %.4g (LEGACY -- does not guard the worst-populated "
              "group) -> thr=%.6g" % (args.recall, thr), flush=True)

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = MLP(len(FEATS)).to(dev)
    model.load_state_dict(state)
    idx = np.random.RandomState(3).choice(len(D["y"]), 2000, replace=False)
    Xchk = D["X"][idx]
    s_np = numpy_forward(state, mean, std, Xchk)
    s_t = scores_of(model, ((Xchk - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32),
                    dev)
    dmax = float(np.abs(s_np - s_t).max())
    print("self-check numpy-vs-torch max|diff| = %.3e" % dmax, flush=True)
    assert dmax < 1e-4

    note = ("weight_mode=%s cap=%.3g select=%s threshold_rule=%s recall=%.4g group_min_count=%d"
            % (winner.get("weight_mode", "none"), winner.get("weight_cap", 0.0),
               args.select_metric, args.threshold_rule, args.recall, args.group_min_count))
    export_header(args.out, state, mean, std, thr, len(FEATS), bank=args.bank, note=note)
    print("exported %s (bank=%s, thr=%.6g)" % (args.out, args.bank, thr), flush=True)

    res = evaluate(D, s, thr, tag="%s/FINAL(test)" % name)
    if args.threshold_rule == "global":
        # Show what the default sub-population rule would have picked, so the global point is never
        # adopted without seeing its per-group cost.
        TPh = tp_min_scores(D, s)
        t2, w2 = worst_group_survival_threshold(
            TPh, survival_ref(TPh, floor=args.survival_floor), args.group_min_count)
        print("\n[compare] worst-group SURVIVAL rule would pin thr=%.6g (binding: %s)" % (t2, w2),
              flush=True)
        evaluate(D, s, t2, tag="%s/worstgroup-survival" % name)
    sv_out = survival_tables(tp_min_scores(D, s), thr)
    final_plots(name, D["y"], s, np.asarray(D["sample_tags"], dtype=object)[D["sample"]], thr)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"winner": name, "thr": thr, "eval": res, "res": winner,
                       "survival_by_group": {k: np.asarray(v[0]).tolist()
                                             for k, v in sv_out.items() if k != "ALL"}}, f)


if __name__ == "__main__":
    main()
