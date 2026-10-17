"""Plain gradient-boosted forest training on a feature cache, for the prompt model-family
comparison (prompt_hp_bakeoff.py imports fit_forest for its forest arms).

Reads a feature cache .npz (nano_loader.py cache*), trains an XGB booster, and saves the
booster json that export_compact_tree.py turns into the compact .bin the selector reads.
The deployed selectors are trained by train_merged_forest.py, not by this script.

WORKING POINT: the printed recall-0.99 threshold is a STARTING POINT, not the deployed
value. The deployed threshold comes from an in-situ (cmsRun/MTV) scan; re-scan before
pinning it.
"""
import argparse
import numpy as np
import torch
import xgboost as xgb


def thr_at_recall(y, s, rc):
    r = np.sort(s[y == 1]); return r[int((1.0 - rc) * len(r))]


def rej(y, s, rc=0.99):
    return float(1.0 - (s[y == 0] >= thr_at_recall(y, s, rc)).mean())


def fit_forest(Xtr, ytr, wtr, Xva, yva, depth=8, ntrees=800, lr=0.05, min_child_weight=5.0,
               reg_lambda=2.0, threads=16, eval_metric="logloss", early_stopping_rounds=50):
    """Fit the gradient-boosted forest. Factored out so prompt_hp_bakeoff.py's forest arms use
    the same estimator the deploy path builds. Returns (clf, n_trees_scored) where
    n_trees_scored = best_iteration+1 (see the tree-count caveat in main())."""
    clf = xgb.XGBClassifier(tree_method="hist", max_depth=depth, n_estimators=ntrees,
                            learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
                            n_jobs=threads, eval_metric=eval_metric,
                            early_stopping_rounds=early_stopping_rounds)
    clf.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    return clf, int(clf.best_iteration) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="feature cache npz (nano_loader.py cache* ...)")
    ap.add_argument("--out", default=None,
                    help="optional raw Hummingbird TorchScript output. Hummingbird is NOT in "
                         "cmsenv (XGB venv only); the DEPLOYED artifact is the compact .bin from "
                         "export_compact_tree.py, so omit --out unless you want the TorchScript.")
    ap.add_argument("--xgb-json", default=None, help="optional: also save the XGB booster json")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--ntrees", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-child-weight", type=float, default=5.0)
    ap.add_argument("--reg-lambda", type=float, default=2.0)
    ap.add_argument("--disp-weight", type=float, default=5.0, help="displacement loss weight beta")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--feats", default=None,
                    help="comma-separated feature names to select/reorder from the cache (default: all)")
    ap.add_argument("--eta-weight", type=float, default=1.0,
                    help="extra training-weight multiplier for |eta| in [eta-lo,eta-hi) (1.0 = off). "
                         "Up-weights the |eta|~1.3-1.6 tilted-module fake-spike band so the tree spends "
                         "more capacity separating it.")
    ap.add_argument("--eta-lo", type=float, default=1.3)
    ap.add_argument("--eta-hi", type=float, default=1.6)
# Early-stopping defaults; the forest arms use auc / 40.
    ap.add_argument("--eval-metric", default="logloss",
                    help="XGB eval_metric for early stopping (logloss or auc)")
    ap.add_argument("--early-stopping-rounds", type=int, default=50,
                    help="XGB early_stopping_rounds")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    z = np.load(args.cache, allow_pickle=True)
    X, y, dxy, evt = z["X"].astype(np.float32), z["y"], z["dxyBS"], z["evt"]
    aeta = np.abs(X[:, list(z["feats"]).index("eta")]).astype(np.float32)  # for the optional eta-weight
    if args.feats:  # select or reorder columns by name
        want = args.feats.split(",")
        cf = list(z["feats"])
        X = X[:, [cf.index(f) for f in want]]
        # rzChi2 is NaN where r-z is undefined; -1 is the in-kernel sentinel
        X = np.nan_to_num(X, nan=-1.0, posinf=1e9, neginf=-1e9).astype(np.float32)
        print(f"selected {len(want)} feats: {want}", flush=True)
    m = evt % 10
    mtr, mva, mte = (m >= 4), (m == 3), (m < 3)            # event-split train/val/test
    Xtr, ytr, Xva, yva, Xte, yte = X[mtr], y[mtr], X[mva], y[mva], X[mte], y[mte]
    # class balance times displacement weight
    pw = (1.0 - ytr.mean()) / max(1e-3, ytr.mean())
    wtr = np.where(ytr == 1, pw, 1.0).astype(np.float32) * \
        (1.0 + args.disp_weight * np.minimum(np.abs(dxy[mtr]), 60.0) / 20.0).astype(np.float32)
    if args.eta_weight != 1.0:  # up-weight the |eta|~1.3-1.6 tilted-module fake-spike band
        band = (aeta[mtr] >= args.eta_lo) & (aeta[mtr] < args.eta_hi)
        wtr = wtr * np.where(band, args.eta_weight, 1.0).astype(np.float32)
        print(f"eta-weight x{args.eta_weight} on |eta| in [{args.eta_lo},{args.eta_hi}): "
              f"{band.mean()*100:.1f}% of train rows up-weighted", flush=True)

    clf, nt = fit_forest(Xtr, ytr, wtr, Xva, yva, depth=args.depth, ntrees=args.ntrees,
                         lr=args.lr, min_child_weight=args.min_child_weight,
                         reg_lambda=args.reg_lambda, threads=args.threads,
                         eval_metric=args.eval_metric,
                         early_stopping_rounds=args.early_stopping_rounds)
    ste = clf.predict_proba(Xte)[:, 1].astype(np.float32)
    print(f"XGB d{args.depth} nt={nt}: test rej@99={rej(yte, ste):.4f} "
          f"rej@99.5={rej(yte, ste, 0.995):.4f} rej@98={rej(yte, ste, 0.98):.4f}", flush=True)
# Working point at the recall-0.99 rule.
    print(f"  recall 99.0%: scoreThreshold={float(thr_at_recall(yte, ste, 0.99)):.4f}  "
          f"fakeRej={rej(yte, ste, 0.99):.4f}  <- tree deploy WP", flush=True)
    # per-|eta|-band fake rejection at 99 % recall
    ae = aeta[mte]
    bands = [(0, 0.8), (0.8, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0)]
    msg = "  rej@99 by |eta|:"
    for lo, hi in bands:
        bm = (ae >= lo) & (ae < hi)
        msg += f" [{lo}-{hi}]={rej(yte[bm], ste[bm]):.3f}" if bm.sum() > 200 else f" [{lo}-{hi}]=na"
    print(msg, flush=True)
    if args.xgb_json:
        # Save the booster, not the sklearn wrapper: export_compact_tree.py loads it with
        # xgb.Booster().load_model(), and XGBClassifier.save_model() raises on scikit-learn 1.8.
        clf.get_booster().save_model(args.xgb_json)
        # predict_proba() honours iteration_range up to best_iteration, while save_model() writes
        # all n_estimators, so the deployed forest is longer than the printed metrics describe.
        print(f"saved booster -> {args.xgb_json} (ALL {args.ntrees} trees; the metrics above are "
              f"for the first {nt} = best_iteration+1)", flush=True)

    if args.out:
        # Lazy import: hummingbird lives only in the XGB venv, and a module-scope import would
        # make the script unimportable under cmsenv.
        from hummingbird.ml import convert
        hb = convert(clf, "torch")
        hb.model.eval()
        sample = torch.tensor(Xte[:64].tolist())
        torch.jit.trace(hb.model, sample).save(args.out)
        print(f"saved raw Hummingbird TorchScript -> {args.out} (nt={nt})", flush=True)


if __name__ == "__main__":
    main()
