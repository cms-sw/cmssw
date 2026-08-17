"""Loader for the trackNano training tables (trackNano_config.py output).

Per event, joins Trk<X>Full (ALL SoA tracks, exact 17-feature deployment ABI of
PixelTrackTorchHighPuritySelector) with Trk<X>Truth by (pt, phi) VALUE match (the truth
table is pt-sorted, the feature table in SoA order; values agree to 1e-7). Returns a
DataFrame of the JOINED population (the tracks the HP/tight selection decides on).

Prefixes: TrkPrompt, TrkDisp, and -- NANO_MERGED=1 -- TrkMerged (both iterations' output
together, the population a final HP forest decides on). Cache verbs: `cache-merged`
(35 cols, deploy order, y_mtv/y_eff) and `cache-arm` (per-arm, 31 deployed feats,
same npz schema as cache-merged).

Every cache records which array `y` is (y_is + label_def). A trainer MUST read y_is and
refuse a cache whose target is wrong -- a cache where y silently meant the narrow
`matched` trains a model that rejects real tracks.
"""
import numpy as np
import pandas as pd
# ROOT is imported lazily inside load_nano() so the feature constants stay importable without it.

# Kernel ABI order.
FEATS17 = ["chi2", "dzError", "dxyError", "eta", "nHits", "phi", "phiError", "pt",
           "qOverPtError", "dzBS", "dxyBS", "nLayers", "cotThetaError",
           "covCotThetaDz", "covDxyQOverPt", "covPhiDxy", "covPhiQOverPt"]
TRUTH_COLS = ["matched", "duplicate", "tpPt", "tpEta"]
TRUTH_MTV = ["matchedAny", "duplicateAny"]
# `matched` is the narrow, efficiency-selected match and is the recall axis; `matchedAny` matches
# against all TrackingParticles and is the training target. A row matched only under the inclusive
# selection is a real track failing an efficiency cut. matchedAny >= matched row-wise.

# Hit and stub features from caTrackFeatures::fill, in deployed order
# (PixelTrackFeaturesSoA columns 18-27).
CA_ADD = ["fitChi2", "psFrac", "r0", "nPS", "spanZ", "nStubs", "logChi2Stub", "kErr", "dcaEst", "nBarrel"]
FEATS_UNION = FEATS17 + CA_ADD
# Study-only columns, read via load_nano(..., extra_ca=CA_EXTRA).
CA_EXTRA = ["meanStubKappa", "leverArm", "rMax", "rzChi2", "meanClusterY", "nTilted", "tiltedFrac"]

# Deployed 31-feature ABI (PixelTrackFeaturesSoA columns 1-31). The four hit-geometry extras are
# in ABI order here, not in CA_EXTRA order.
CA_GEOM4 = ["rzChi2", "meanStubKappa", "leverArm", "rMax"]
FEATS31 = FEATS_UNION + CA_GEOM4
# Merged-collection provenance, appended after the 31 so the deploy block stays a prefix.
CA_PROV = ["iteration", "ndof", "nOTExtra", "nAttached"]
# Pixel-cluster block, appended after the provenance block so X[:, :31] and X[:, :35] both stay
# prefixes. It carries its own -1 sentinel and is not part of the finite gate.
CLUSTER8 = ["nPixHits", "minCharge", "meanCharge", "minChargeNorm",
            "maxSizeY", "meanSizeY", "maxSizeX", "nLowCharge"]


# Stored in every cache so no consumer has to guess which array the target is.
LABEL_DEFS = {
    "y_mtv": ("y = y_mtv = matchedAny: matched to ANY TrackingParticle. MTV's own definition of a "
              "real track (a track is fake iff it matches no TP at all), and the DQM fake-rate "
              "axis. Recall belongs on y_eff, never on y."),
    "y_eff": ("y = y_eff = matched: matched to a TP passing the arm's EFFICIENCY selection. This is "
              "the NARROW label -- real tracks failing an efficiency cut (low pT, secondaries, "
              "large |tip|, out-of-time PU) count as negatives, so a model trained on it learns to "
              "reject them. Kept only for legacy comparisons."),
}


def _arr(tree, name):
    return np.asarray(getattr(tree, name), dtype=np.float64)


def has_branches(path, prefix, table, cols):
    """True iff every <prefix><table>_<c> branch exists in `path`. Used to decide whether a per-arm
    nano carries the merged-provenance block (emitMergedProvenance is False on the prompt/displaced
    CA tables by default), instead of failing with an AttributeError deep inside the event loop."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(path)
    t = f.Get("Events")
    ok = all(t.GetBranch(f"{prefix}{table}_{c}") for c in cols)
    f.Close()
    return bool(ok)


def load_nano(path, prefix, max_events=None, quiet=False, with_ca=False, extra_ca=None,
              extra_truth=None):
    """path: trackNano root file; prefix: 'TrkPrompt', 'TrkDisp' or 'TrkMerged'.
    with_ca=True: also attach the 10 CA_ADD hit/stub features by INDEX
    (Trk<X>CA row j == Trk<X>Full row j; phi agreement asserted), for the 27-feature
    final selector.
    extra_ca: optional list of EXTRA Trk<X>CA columns to also read (study/provenance features).
    extra_truth: optional list of EXTRA Trk<X>Truth columns to also read (e.g. TRUTH_MTV)."""
    if with_ca and prefix not in ("TrkDisp", "TrkPrompt", "TrkMerged"):
        raise ValueError("with_ca is only available for the TrkDisp / TrkPrompt / TrkMerged tables")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    extra = (list(CA_ADD) + list(extra_ca or [])) if with_ca else []
    truth_cols = list(TRUTH_COLS) + list(extra_truth or [])
    f = ROOT.TFile.Open(path)
    t = f.Get("Events")
    missing = [c for c in (extra_truth or []) if not t.GetBranch(f"{prefix}Truth_{c}")]
    if missing:
        f.Close()
        raise RuntimeError(
            f"{path}: {prefix}Truth has no branch(es) {missing}. The MTV-true label needs a nano "
            f"dumped with the second, all-TrackingParticle association: NANO_MERGED=1 for the "
            f"merged arm, NANO_MTV_LABEL=1 for the prompt/displaced arms.")
    n = t.GetEntries()
    if max_events:
        n = min(n, max_events)
    rows = []
    n_truth_tot = n_joined = n_phimismatch = 0
    for iev in range(n):
        t.GetEntry(iev)
        full = {c: _arr(t, f"{prefix}Full_{c}") for c in FEATS17}
        ca = {c: _arr(t, f"{prefix}CA_{c}") for c in extra}
        if with_ca:
            ca_phi = _arr(t, f"{prefix}CA_phi")
            if len(ca_phi) != len(full["phi"]):
                raise RuntimeError(f"event {iev}: {prefix}CA rows != {prefix}Full rows")
            n_phimismatch += int((np.abs(ca_phi - full["phi"]) > 1e-4).sum())
        tr_pt = _arr(t, f"{prefix}Truth_pt")
        tr_phi = _arr(t, f"{prefix}Truth_phi")
        truth = {c: _arr(t, f"{prefix}Truth_{c}") for c in truth_cols}
        n_truth_tot += len(tr_pt)
        if len(tr_pt) == 0 or len(full["pt"]) == 0:
            continue
        # Value join keyed on the rounded (pt, phi) pair.
        key_full = {}
        for j, (p, ph) in enumerate(zip(full["pt"], full["phi"])):
            key_full[(round(float(p), 5), round(float(ph), 5))] = j
        for i in range(len(tr_pt)):
            j = key_full.get((round(float(tr_pt[i]), 5), round(float(tr_phi[i]), 5)))
            if j is None:
                continue
            n_joined += 1
            row = ([iev] + [full[c][j] for c in FEATS17] + [ca[c][j] for c in extra]
                   + [truth[c][i] for c in truth_cols])
            rows.append(row)
    f.Close()
    df = pd.DataFrame(rows, columns=["evt"] + FEATS17 + extra + truth_cols)
    df["label"] = df["matched"].astype(np.float32)
    if not quiet:
        jr = n_joined / max(1, n_truth_tot)
        ca_msg = f" phi-mismatch(idx)={n_phimismatch}" if with_ca else ""
        mtv_msg = f" realAny(MTV)={df.matchedAny.mean():.4f}" if "matchedAny" in df.columns else ""
        print(f"  {path}:{prefix}: events={n} truth rows={n_truth_tot} joined={n_joined} "
              f"({jr:.4f}) real={df.label.mean():.4f}{mtv_msg} dup={df.duplicate.mean():.4f}{ca_msg}",
              flush=True)
    return df


def load_cache(path, feats=None, quiet=False):
    """Load a feature cache written by dump_*_cache() back into a DataFrame.

    Lets every prompt-HP model class train on the SAME rows and event split: the forest
    reads the cache natively, and this lets the MLP too (loading from ROOT per-arm applies
    different NaN policies, silently changing the population).

    feats: optional feature names to select/reorder. Returns the selected features + evt
    + label (+ dxyBS if cached). Needs NO ROOT (works in the XGB venv)."""
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["feats"]]
    if feats is not None:
        missing = [f for f in feats if f not in cols]
        if missing:
            raise ValueError(f"{path}: cache has {len(cols)} features {cols}; missing {missing}. "
                             f"Rebuild it with the right nano_loader.py cache* verb "
                             f"(cache-prompt27 gives the deployed 27-feature prompt block).")
        idx = [cols.index(f) for f in feats]
        X, cols = z["X"][:, idx], list(feats)
    else:
        X = z["X"]
    df = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=cols)
    df["evt"] = np.asarray(z["evt"], dtype=np.int64)
    df["label"] = np.asarray(z["y"], dtype=np.float32)
    if "dxyBS" in z.files and "dxyBS" not in df.columns:
        df["dxyBS"] = np.asarray(z["dxyBS"], dtype=np.float32)
    if not quiet:
        print(f"  {path}: {len(df)} rows, {len(cols)} feats, real={df.label.mean():.4f}", flush=True)
    return df


def dump_union_cache(files, out, min_dxy=0.5, extra_ca=None):
    """Dump the displaced union-27 cache (17 fit + 10 hit/stub CA features) to a .npz
    the no-ROOT venv reads. Run under cmsenv (needs ROOT). Restricts to |dxyBS|>min_dxy,
    drops non-finite rows. extra_ca: optional study columns appended after the 27."""
    cols = list(FEATS_UNION) + list(extra_ca or [])
    df = pd.concat([load_nano(p, "TrkDisp", with_ca=True, extra_ca=extra_ca) for p in files], ignore_index=True)
    df = df[np.abs(df["dxyBS"].values) > min_dxy].reset_index(drop=True)
    fin = np.isfinite(df[FEATS_UNION].values).all(axis=1)  # gate on the deployed features only
    df = df[fin].reset_index(drop=True)
    np.savez_compressed(out,
                        X=df[cols].values.astype(np.float32),
                        y=df["label"].values.astype(np.float32),
                        dxyBS=df["dxyBS"].values.astype(np.float32),
                        evt=df["evt"].values.astype(np.int64),
                        feats=np.array(cols),
                        y_is=np.array("y_eff"), label_def=np.array(LABEL_DEFS["y_eff"]))
    print(f"cached {len(df)} rows real={df.label.mean():.4f} feats={len(cols)} "
          f"y=y_eff(NARROW label) -> {out}", flush=True)


def dump_prompt_cache(files, out, feat_set="fit17"):
    """Dump the PROMPT HP training cache to a .npz the no-ROOT venv / build_tree_model.py reads.
    Run under cmsenv (needs ROOT). Drops non-finite rows on the selected feature block;
    the evt column feeds the same evt%10 split as train_prompt_hp_nano.py so bake-offs are fair.

    feat_set: "fit17" (default) = the 17 fit/cov features; "union" = the 27-feature prompt HP
    block: 17 fit/cov + 10 hit/stub CA (PixelTrackFeaturesSoA columns 1-27). Requires the
    TrkPromptCA table in the nano."""
    feats = {"fit17": FEATS17, "union": FEATS_UNION}[feat_set]
    with_ca = feat_set == "union"
    df = pd.concat([load_nano(p, "TrkPrompt", with_ca=with_ca) for p in files], ignore_index=True)
    df = df[np.isfinite(df[feats].values).all(axis=1)].reset_index(drop=True)
    np.savez_compressed(out,
                        X=df[feats].values.astype(np.float32),
                        y=df["label"].values.astype(np.float32),
                        dxyBS=df["dxyBS"].values.astype(np.float32),
                        evt=df["evt"].values.astype(np.int64),
                        feats=np.array(feats),
                        y_is=np.array("y_eff"), label_def=np.array(LABEL_DEFS["y_eff"]))
    print(f"cached {len(df)} prompt rows real={df.label.mean():.4f} feat_set={feat_set} "
          f"feats={len(feats)} y=y_eff(NARROW label) -> {out}", flush=True)


def dump_merged_cache(files, out):
    """Dump the MERGED-collection cache (TrkMerged*) in DEPLOY order for a final HP forest.

    Layout: 35 cols = 31 deployed features (FEATS31, deploy order, no reorder needed) +
    4 provenance (CA_PROV: iteration, ndof, nOTExtra, nAttached). Deploy block is a prefix.

    THREE label arrays: y_mtv = matchedAny (the TRAINING TARGET; `y` is its alias),
    y_eff = matched (the RECALL axis), duplicate (under the efficiency selection). Rows with
    y_mtv==1 and y_eff==0 are real tracks failing an efficiency cut (low pT, secondary, large
    |tip|, out-of-time PU) -- the population a narrow-label training teaches the model to reject.

    NaN policy: rzChi2 NaN -> -1 (the kernel's sentinel for an undefined r-z fit) rather than
    costing the row, so training sees the inference value. The other 30 columns are finite-gated."""
    cl_flags = [has_branches(p, "TrkMerged", "CA", CLUSTER8) for p in files]
    with_cl = all(cl_flags)
    if any(cl_flags) and not with_cl:
        missing = [p for p, ok in zip(files, cl_flags) if not ok]
        raise SystemExit(
            "some inputs carry the pixel-cluster block and some do not, so the columns would not "
            "line up: %s lack Trk MergedCA_%s. Re-dump the whole set with NANO_CLUSTER=1, or none "
            "of it." % (missing, "/".join(CLUSTER8)))
    clus = list(CLUSTER8) if with_cl else []
    cols = list(FEATS31) + list(CA_PROV) + clus
    df = pd.concat([load_nano(p, "TrkMerged", with_ca=True, extra_ca=CA_GEOM4 + CA_PROV + clus,
                              extra_truth=TRUTH_MTV) for p in files], ignore_index=True)
    rz_nan = ~np.isfinite(df["rzChi2"].values)
    df["rzChi2"] = np.where(rz_nan, -1.0, df["rzChi2"].values)
    gate = [c for c in FEATS31 if c != "rzChi2"]
    keep = np.isfinite(df[gate].values).all(axis=1)
    n_rzfix = int((rz_nan & keep).sum())  # rows rescued by the map; the rest fail another column
    df = df[keep].reset_index(drop=True)
    y_mtv = df["matchedAny"].values.astype(np.float32)
    y_eff = df["matched"].values.astype(np.float32)
    bad = int((y_eff > y_mtv).sum())
    if bad:
        print(f"  WARNING: {bad} rows have matched=1 with matchedAny=0 -- the two associations "
              f"disagree, which should be impossible; check the tpSelectorAnyTP configuration.",
              flush=True)
    np.savez_compressed(out,
                        X=df[cols].values.astype(np.float32),
                        y=y_mtv,  # `y` is the training target
                        y_mtv=y_mtv,
                        y_eff=y_eff,
                        duplicate=df["duplicate"].values.astype(np.float32),
                        dxyBS=df["dxyBS"].values.astype(np.float32),
                        evt=df["evt"].values.astype(np.int64),
                        feats=np.array(cols),
                        y_is=np.array("y_mtv"), label_def=np.array(LABEL_DEFS["y_mtv"]),
                        prefix=np.array("TrkMerged"), min_dxy=np.array(-1.0))
    cl_msg = ""
    if with_cl:
        nosent = int((df[CLUSTER8[0]].values < 0).sum())
        cl_msg = (f" + 8 pixel-cluster, sentinel(-1) on {nosent} rows "
                  f"({nosent / max(1, len(df)):.4f})")
    print(f"cached {len(df)} merged rows  y_mtv={y_mtv.mean():.4f} y_eff={y_eff.mean():.4f} "
          f"(B = real but not an efficiency TP: {(y_mtv - y_eff).mean():.4f})  feats={len(cols)} "
          f"(31 deployed + 4 provenance{cl_msg})  rzChi2 NaN->-1 on {n_rzfix} rows -> {out}",
          flush=True)


def dump_arm_cache(files, out, prefix="TrkPrompt", min_dxy=None, label="mtv"):
    """Dump a PER-ARM cache (TrkPrompt / TrkDisp) in the SAME npz schema as dump_merged_cache,
    so the prompt and displaced HP forests train on the same code path and metric axes as the merged one.

    Layout: 31 deployed features (FEATS31) as a PREFIX. The 4 merged-provenance columns are
    appended only if the nano carries them (emitMergedProvenance is False on per-arm dumps, so
    normally the cache is exactly 31 wide). Never fabricated: a zero column would enter a
    35-feature model as a constant.

    Labels (same meaning as the merged cache): y_mtv = matchedAny (target; y is its alias),
    y_eff = matched (recall axis). label="legacy" makes y an alias of y_eff instead.

    min_dxy: keep only |dxyBS| > min_dxy (displaced forest: 0.5; None for the prompt arm).
    NaN policy: rzChi2 NaN -> -1 (kernel sentinel), other 30 columns finite-gated."""
    if prefix not in ("TrkPrompt", "TrkDisp"):
        raise ValueError("cache-arm is for the per-arm tables TrkPrompt / TrkDisp; the merged row "
                         "space has its own verb, cache-merged (different TP selection and always "
                         "both labels)")
    if label not in ("mtv", "legacy"):
        raise ValueError("label must be 'mtv' or 'legacy'")
    files = list(files)
    # The provenance block must be present in every file or in none: concatenating both widths
    # would silently drop it.
    prov_flags = [has_branches(p, prefix, "CA", CA_PROV) for p in files]
    if any(prov_flags) and not all(prov_flags):
        raise RuntimeError(
            "some nanos carry the %sCA provenance block and some do not: %s. Re-dump the odd ones "
            "out, or drop them." % (prefix, dict(zip(files, prov_flags))))
    with_prov = all(prov_flags) and bool(files)
    prov = list(CA_PROV) if with_prov else []
    cols = list(FEATS31) + prov
    df = pd.concat([load_nano(p, prefix, with_ca=True, extra_ca=CA_GEOM4 + prov,
                              extra_truth=TRUTH_MTV) for p in files], ignore_index=True)
    n_all = len(df)
    if min_dxy is not None:
        df = df[np.abs(df["dxyBS"].values) > min_dxy].reset_index(drop=True)
    n_dxy = len(df)
    rz_nan = ~np.isfinite(df["rzChi2"].values)
    df["rzChi2"] = np.where(rz_nan, -1.0, df["rzChi2"].values)
    gate = [c for c in FEATS31 if c != "rzChi2"]
    keep = np.isfinite(df[gate].values).all(axis=1)
    n_rzfix = int((rz_nan & keep).sum())
    df = df[keep].reset_index(drop=True)
    y_mtv = df["matchedAny"].values.astype(np.float32)
    y_eff = df["matched"].values.astype(np.float32)
    bad = int((y_eff > y_mtv).sum())
    if bad:
        print(f"  WARNING: {bad} rows have matched=1 with matchedAny=0 -- the two associations "
              f"disagree, which should be impossible; check tpSelectorAnyTP.", flush=True)
    y_is = "y_mtv" if label == "mtv" else "y_eff"
    y = y_mtv if label == "mtv" else y_eff
    np.savez_compressed(out,
                        X=df[cols].values.astype(np.float32),
                        y=y,
                        y_mtv=y_mtv,
                        y_eff=y_eff,
                        duplicate=df["duplicate"].values.astype(np.float32),
                        dxyBS=df["dxyBS"].values.astype(np.float32),
                        evt=df["evt"].values.astype(np.int64),
                        feats=np.array(cols),
                        y_is=np.array(y_is), label_def=np.array(LABEL_DEFS[y_is]),
                        prefix=np.array(prefix),
                        min_dxy=np.array(-1.0 if min_dxy is None else float(min_dxy)))
    print(f"cached {len(df)} {prefix} rows  y_mtv={y_mtv.mean():.4f} y_eff={y_eff.mean():.4f} "
          f"(B = real but not an efficiency TP: {(y_mtv - y_eff).mean():.4f})  y={y_is}  "
          f"feats={len(cols)} ({'31 deployed + 4 provenance' if with_prov else '31 deployed'})  "
          f"rows {n_all} -> {n_dxy} after |dxyBS|>{min_dxy} -> {len(df)} finite  "
          f"rzChi2 NaN->-1 on {n_rzfix} rows -> {out}", flush=True)


if __name__ == "__main__":
    # python nano_loader.py cache <out.npz> <trackNano_a.root> [more.root ...]
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "cache":
        dump_union_cache(sys.argv[3:], sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "cache-ext":  # 27 + study columns
        dump_union_cache(sys.argv[3:], sys.argv[2], extra_ca=CA_EXTRA)
    elif len(sys.argv) >= 4 and sys.argv[1] == "cache-prompt":  # 17-feat prompt HP tree bake-off cache
        dump_prompt_cache(sys.argv[3:], sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "cache-prompt27":  # prompt HP block (17 fit + 10 hit/stub)
        dump_prompt_cache(sys.argv[3:], sys.argv[2], feat_set="union")
    elif len(sys.argv) >= 4 and sys.argv[1] == "cache-merged":  # MERGED collection, deploy order
        # 35 columns: the 31 deployed features plus iteration, ndof, nOTExtra, nAttached. Needs a
        # nano carrying the TrkMerged tables, the provenance block and both associations.
        dump_merged_cache(sys.argv[3:], sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "cache-arm":  # PER-ARM row space, deploy order
        # python nano_loader.py cache-arm <out.npz> <files...> --prefix TrkPrompt|TrkDisp
        #                                 [--min-dxy 0.5] [--label mtv|legacy]
        # 31 deployed features, plus the 4 provenance columns when the nano carries them.
        argv, kw = [], {"prefix": "TrkPrompt", "min_dxy": None, "label": "mtv"}
        it = iter(sys.argv[2:])
        for a in it:
            if a == "--prefix":
                kw["prefix"] = next(it)
            elif a == "--min-dxy":
                kw["min_dxy"] = float(next(it))
            elif a == "--label":
                kw["label"] = next(it)
            elif a.startswith("--"):
                raise SystemExit("cache-arm: unknown option %r" % a)
            else:
                argv.append(a)
        if len(argv) < 2:
            raise SystemExit("cache-arm: give <out.npz> and at least one trackNano file")
        dump_arm_cache(argv[1:], argv[0], **kw)
    else:
        print("usage: python nano_loader.py "
              "cache|cache-ext|cache-prompt|cache-prompt27|cache-merged <out.npz> <files...>\n"
              "       python nano_loader.py cache-arm <out.npz> <files...> "
              "--prefix TrkPrompt|TrkDisp [--min-dxy 0.5] [--label mtv|legacy]")
