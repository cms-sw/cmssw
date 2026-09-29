#!/usr/bin/env python3
"""PROMPT final high-purity selector comparison: train every candidate model class on ONE
shared dataset, score them all at ONE working-point rule, and stage the winner. Run by
`retrain_prompt.sh hp` on every retrain, so a change of model family is a measured decision.

ARMS (--arms, default all four):
  mlp17     17 feats, Torch MLP 64x64x64x64   -- the family PixelTrackTorchHighPuritySelector loads
  mlp27     27 feats (17 fit/cov + 10 hit/stub) -- the MLP on the SAME inputs as the forest
  forest17  17 feats, gradient-boosted forest
  forest27  27 feats, gradient-boosted forest   -- the family PixelTrackForestHighPuritySelector loads

WHY IT IS FAIR: ONE cache (identical rows; a cache is required, not optional -- loading from
ROOT per-arm applies different NaN policies); ONE event-level split (evt%10); ONE weighting
recipe; ONE working-point rule for the table and the winner; each arm calls its deploy path's
own helpers.

WORKING-POINT RULE: fake rejection at matched per-track recall (default 0.99). Legitimate here
(one decision per track => per-TP survival == recall; NOT true of the per-triplet gate -- see
retrain_triplet_gate.sh). The winner's printed threshold is a STARTING POINT: the DEPLOYED
threshold is an in-situ (cmsRun/MTV) pick.

USAGE (cmsenv; driven by retrain_prompt.sh):
  python3 prompt_hp_bakeoff.py --cache prompt27_cache.npz --outdir $MODELS_WORKDIR \
      [--arms mlp17,mlp27,forest17,forest27] [--select-recall 0.99] [--device cuda:0] \
      [--max-rows N] [--epochs N] [--depth 8 --ntrees 800]

OUTPUTS (in --outdir): HP_BAKEOFF.json/.txt, WINNER.json (read by retrain_prompt.sh),
model_<arm>.pt (MLP arms, the format train_prompt_hp_nano.py finalize expects),
booster_<arm>.json (forest arms, what export_compact_tree.py reads).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tree_model import fit_forest  # noqa: E402
from nano_loader import FEATS17, FEATS_UNION, load_cache  # noqa: E402
from nano_train_utils import MLPn, scores  # noqa: E402
from train_prompt_hp_nano import HIDDEN, _wfocal, split_by_event, xy  # noqa: E402

FEATSETS = {"fit17": FEATS17, "union": FEATS_UNION}
ARMS = {  # arm -> (model class, feature set)
    "mlp17": ("mlp", "fit17"),
    "mlp27": ("mlp", "union"),
    "forest17": ("forest", "fit17"),
    "forest27": ("forest", "union"),
}


def thr_at_recall(y, s, rc):
    r = np.sort(s[y == 1])
    return float(r[int((1.0 - rc) * len(r))])


def rej_at(y, s, rc):
    return float(1.0 - (s[y == 0] >= thr_at_recall(y, s, rc)).mean())


def metrics(y, s, aeta=None, recalls=(0.999, 0.995, 0.99, 0.98, 0.95)):
    out = {"auc": float(roc_auc_score(y, s))}
    for rc in recalls:
        out["rej@%.3f" % rc] = rej_at(y, s, rc)
        out["thr@%.3f" % rc] = thr_at_recall(y, s, rc)
    if aeta is not None:  # per-|eta| rejection at the 99 % point
        band = {}
        for lo, hi in ((0, 0.8), (0.8, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0)):
            m = (aeta >= lo) & (aeta < hi)
            band["%.1f-%.1f" % (lo, hi)] = rej_at(y[m], s[m], 0.99) if m.sum() > 200 else None
        out["rej@0.99_by_eta"] = band
    return out


def train_mlp(tr, va, te, feats, tag, outdir, device, epochs, patience=20, hidden=HIDDEN):
    """The prompt-MLP recipe (train_prompt_hp_nano.train), on the shared split."""
    Xtr, ytr, mean, std = xy(tr, feats)
    Xva, yva, _, _ = xy(va, feats, mean, std)
    Xte, yte, _, _ = xy(te, feats, mean, std)
    pw = (1.0 - tr.label.mean()) / max(1e-3, tr.label.mean())
    wtr = np.where(ytr == 1, pw, 1.0).astype(np.float32)
    torch.manual_seed(7)
    dev = torch.device(device)
    net = MLPn(len(feats), hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    Xt = torch.tensor(Xtr.tolist(), device=dev)
    yt = torch.tensor(ytr.tolist(), device=dev)
    wt = torch.tensor(wtr.tolist(), device=dev)
    best = (-1.0, None, -1)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i0 in range(0, len(Xt), 65536):
            idx = perm[i0:i0 + 65536]
            loss = _wfocal(net(Xt[idx]), yt[idx], 0.0, wt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        r = rej_at(yva, scores(net, Xva, dev), 0.99)
        sch.step(r)
        if r > best[0]:
            best = (r, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
        if ep - best[2] >= patience:
            break
    net.load_state_dict(best[1])
    # Blob format matches train_prompt_hp_nano.py, so `finalize --name <arm>` exports the winner
    # to TorchScript without re-training it.
    torch.save({"state": best[1], "mean": mean.tolist(), "std": std.tolist(), "feats": list(feats),
                "feat_set": "union" if len(feats) == len(FEATS_UNION) else "fit17",
                "hidden": list(hidden)}, os.path.join(outdir, "model_%s.pt" % tag))
    return scores(net, Xte, dev), yte, {"best_epoch": best[2]}, "model_%s.pt" % tag


def train_forest(tr, va, te, feats, tag, outdir, depth, ntrees, threads):
    Xtr = tr[feats].values.astype(np.float32)
    ytr = tr.label.values.astype(np.float32)
    Xva = va[feats].values.astype(np.float32)
    yva = va.label.values.astype(np.float32)
    Xte = te[feats].values.astype(np.float32)
    yte = te.label.values.astype(np.float32)
    pw = (1.0 - ytr.mean()) / max(1e-3, ytr.mean())
    wtr = np.where(ytr == 1, pw, 1.0).astype(np.float32)
    clf, nt = fit_forest(Xtr, ytr, wtr, Xva, yva, depth=depth, ntrees=ntrees, threads=threads,
                         eval_metric="auc", early_stopping_rounds=40)
    out = os.path.join(outdir, "booster_%s.json" % tag)
    clf.get_booster().save_model(out)  # booster, not the sklearn wrapper
    return clf.predict_proba(Xte)[:, 1], yte, {"n_trees_scored": nt, "n_trees_saved": ntrees}, \
        "booster_%s.json" % tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="shared feature cache (nano_loader.py cache-prompt27)")
    ap.add_argument("--outdir", default=os.environ.get("MODELS_WORKDIR", "."))
    ap.add_argument("--arms", default="mlp17,mlp27,forest17,forest27")
    ap.add_argument("--select-recall", type=float, default=0.99,
                    help="the ONE rule: winner = highest fake rejection at this per-track recall")
    ap.add_argument("--device", default="cuda:0", help="torch device for the MLP arms")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--ntrees", type=int, default=800)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="subsample rows (SMOKE TESTS ONLY -- a subsampled bake-off is not a "
                         "valid basis for changing the deployed model class)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        ap.error("unknown arm(s) %s; choose from %s" % (bad, sorted(ARMS)))

    # One load and one split, shared by every arm.
    need_union = any(ARMS[x][1] == "union" for x in arms)
    df = load_cache(a.cache, FEATS_UNION if need_union else FEATS17)
    if a.max_rows and a.max_rows < len(df):
        df = df.iloc[:a.max_rows].reset_index(drop=True)
        print("WARNING: --max-rows %d -> SMOKE MODE, results are not deployment-grade"
              % a.max_rows, flush=True)
    tr, va, te = split_by_event(df)
    aeta = np.abs(te["eta"].values) if "eta" in df.columns else None
    print("shared split: train=%d val=%d test=%d  real=%.4f  arms=%s"
          % (len(tr), len(va), len(te), df.label.mean(), ",".join(arms)), flush=True)
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("a split is empty -- the cache spans too few events for evt%10 splitting")

    res = {}
    for arm in arms:
        klass, fs = ARMS[arm]
        feats = FEATSETS[fs]
        print("=== %s (%s, %d features)" % (arm, klass, len(feats)), flush=True)
        if klass == "mlp":
            s, y, meta, art = train_mlp(tr, va, te, feats, arm, a.outdir, a.device, a.epochs)
        else:
            s, y, meta, art = train_forest(tr, va, te, feats, arm, a.outdir, a.depth, a.ntrees,
                                           a.threads)
        r = metrics(y, s, aeta)
        r.update(meta)
        r.update({"class": klass, "feat_set": fs, "n_feats": len(feats), "artifact": art})
        res[arm] = r
        print("    AUC=%.5f  rej@%.3f=%.4f  thr=%.5f"
              % (r["auc"], a.select_recall, r["rej@%.3f" % a.select_recall],
                 r["thr@%.3f" % a.select_recall]), flush=True)

    key = "rej@%.3f" % a.select_recall
    winner = max(res, key=lambda k: res[k][key])
    ranked = sorted(res, key=lambda k: -res[k][key])
    margin = (res[ranked[0]][key] - res[ranked[1]][key]) if len(ranked) > 1 else None

    lines = []
    lines.append("\nPROMPT-HP BAKE-OFF -- one dataset, one split, one rule "
                 "(fake rejection at %.1f%% per-track recall)" % (a.select_recall * 100))
    lines.append("%-10s %-7s %6s %9s %9s %9s %9s %9s"
                 % ("arm", "class", "nfeat", "AUC", "rej@99.9", "rej@99.5", "rej@99", "rej@98"))
    for k in ranked:
        v = res[k]
        lines.append("%-10s %-7s %6d %9.5f %9.4f %9.4f %9.4f %9.4f%s"
                     % (k, v["class"], v["n_feats"], v["auc"], v["rej@0.999"], v["rej@0.995"],
                        v["rej@0.990"], v["rej@0.980"], "   <- WINNER" if k == winner else ""))
    lines.append("")
    for k in ranked:
        b = res[k].get("rej@0.99_by_eta") or {}
        if b:
            lines.append("  %-10s rej@99 by |eta|: %s" % (
                k, "  ".join("[%s]=%s" % (n, "na" if v is None else "%.3f" % v)
                             for n, v in b.items())))
    lines.append("\nWINNER: %s (%s, %d feats) -- %s = %.4f%s"
                 % (winner, res[winner]["class"], res[winner]["n_feats"], key, res[winner][key],
                    "" if margin is None else ", margin over %s = %+.4f" % (ranked[1], margin)))
    if margin is not None and margin < 0.005:
        lines.append("  NOTE margin < 0.005 -- treat the top two as tied and prefer the incumbent "
                     "class unless the challenger also wins on compute/memory.")
    lines.append("  offline threshold at the rule: %.5f  <-- STARTING POINT."
                 % res[winner]["thr@%.3f" % a.select_recall])
    lines.append("  The DEPLOYED threshold is an IN-SITU (cmsRun/MTV) pick and can differ "
                 "materially from the offline rule. Re-scan in situ.")
    txt = "\n".join(lines)
    print(txt, flush=True)

    json.dump(res, open(os.path.join(a.outdir, "HP_BAKEOFF.json"), "w"), indent=1)
    open(os.path.join(a.outdir, "HP_BAKEOFF.txt"), "w").write(txt + "\n")
    json.dump({"winner": winner, "class": res[winner]["class"], "feat_set": res[winner]["feat_set"],
               "n_feats": res[winner]["n_feats"], "artifact": res[winner]["artifact"],
               "select_rule": key, "select_value": res[winner][key],
               "offline_threshold": res[winner]["thr@%.3f" % a.select_recall],
               "margin_over_runner_up": margin, "ranked": ranked, "smoke": bool(a.max_rows)},
              open(os.path.join(a.outdir, "WINNER.json"), "w"), indent=1)
    print("wrote HP_BAKEOFF.json / .txt / WINNER.json -> %s" % a.outdir, flush=True)


if __name__ == "__main__":
    main()
