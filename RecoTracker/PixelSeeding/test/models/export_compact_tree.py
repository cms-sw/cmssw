#!/usr/bin/env python3
"""Export an XGBoost booster to the compact gradient-boosted-tree binary the final-HP
forest selector (PixelTrackForestHighPuritySelector) reads. The selector loads ONE
read-only copy per DEVICE and traverses a flat node array (not a TorchScript .pt).

.bin format (little-endian, exactly what DispTreeCache::load reads):
  HEADER (12 bytes): int32 nNodes, int32 nTrees, float baseLogit = logit(base_score)
  ARRAYS (contiguous, no padding):
    int8  feat[nNodes]    split-feature index, or -1 for a leaf
    float val[nNodes]     split threshold (internal) or leaf value (leaf) -- fp32 BYTES
    int32 left[nNodes]    global index of the "yes/missing-left" child, or -1 for a leaf
    int32 right[nNodes]   global index of the "no" child, or -1 for a leaf
    int32 roots[nTrees]   global index of each tree's root node
  expected size = 12 + nNodes*(1+4+4+4) + nTrees*4

.npz format (intermediate, for analysis):
  feat (int8), val (float16 storage-only), left (int32), right (int32),
  roots (int32), base_logit (float32), nt (int64). With the default fp16 quantisation the
  npz stores `val` as float16 and the .bin upcasts it back to fp32; --no-fp16 keeps fp32.

Node layout (BFS / level-order with sequential child allocation):
  Per tree, nodes emitted BFS from the root. Each internal node at slot i has children
  at the next two free slots (left=next, right=next+1). Leaves get feat=-1,
  left=right=-1 (contiguous, gap-free; XGBoost nodeids are not contiguous in pruned trees).
  The "yes/missing" child becomes `left`, the "no" child becomes `right`.

PRECISION. By default the value array is rounded to fp16 (and stored as fp32); --no-fp16
keeps it in fp32. The flag changes only the value precision, never the file size. The
deployed prompt and merged files are fp32 exports, the deployed displaced file an fp16 one;
train_merged_forest.py passes the flag per collection.

TREE COUNT: save_model() writes ALL n_estimators trees; predict_proba() scores only
  best_iteration+1, so the deployed forest can be a few trees longer than the printed
  metrics describe. An exact metric re-derivation must score at iteration_range=(0,
  best_iteration+1); an exact score re-derivation of the .bin uses all of them.
"""
import argparse
import json
import math
import struct

import numpy as np
import xgboost as xgb


def base_logit_from_booster(bst):
    """Return logit(base_score). For binary:logistic, XGBoost stores base_score as a probability and
    predicts sigmoid(margin + logit(base_score)); the kernel adds baseLogit in margin space."""
    cfg = json.loads(bst.save_config())
    bs = float(cfg["learner"]["learner_model_param"]["base_score"])
    # guard the boundaries
    bs = min(max(bs, 1e-7), 1.0 - 1e-7)
    return math.log(bs / (1.0 - bs))


def flatten_tree(node, feat, val, left, right):
    """Append one tree's nodes (BFS, sequential child allocation) to the running lists.
    Returns the global index of this tree's root."""
    root_global = len(feat)
    # queue holds (node_json, slot_already_reserved). Reserve the root slot first.
    feat.append(0)
    val.append(0.0)
    left.append(-1)
    right.append(-1)
    queue = [(node, root_global)]
    while queue:
        n, slot = queue.pop(0)
        if "leaf" in n:
            feat[slot] = -1
            val[slot] = float(n["leaf"])
            left[slot] = -1
            right[slot] = -1
            continue
        # internal node
        feat[slot] = int(n["split"][1:])  # "f12" -> 12
        val[slot] = float(n["split_condition"])
        children = {c["nodeid"]: c for c in n["children"]}
        yes_c = children[n["yes"]]
        no_c = children[n["no"]]
        # reserve the two child slots contiguously
        l_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        r_idx = len(feat)
        feat.append(0); val.append(0.0); left.append(-1); right.append(-1)
        left[slot] = l_idx
        right[slot] = r_idx
        queue.append((yes_c, l_idx))
        queue.append((no_c, r_idx))
    return root_global


def build_compact(bst):
    dump = bst.get_dump(dump_format="json")
    nt = len(dump)
    feat, val, left, right, roots = [], [], [], [], []
    for t in range(nt):
        r = flatten_tree(json.loads(dump[t]), feat, val, left, right)
        roots.append(r)
    feat = np.asarray(feat, dtype=np.int8)
    val = np.asarray(val, dtype=np.float32)
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    roots = np.asarray(roots, dtype=np.int32)
    base_logit = np.float32(base_logit_from_booster(bst))
    return feat, val, left, right, roots, base_logit, nt


def write_bin(path, feat, val, left, right, roots, base_logit, nt):
    nN = int(feat.shape[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<iif", nN, int(nt), float(base_logit)))
        f.write(feat.astype(np.int8).tobytes())
        f.write(val.astype(np.float32).tobytes())
        f.write(left.astype(np.int32).tobytes())
        f.write(right.astype(np.int32).tobytes())
        f.write(roots.astype(np.int32).tobytes())
    return 12 + nN * (1 + 4 + 4 + 4) + int(nt) * 4


def write_npz(path, feat, val, left, right, roots, base_logit, nt, fp16):
    np.savez(
        path,
        feat=feat.astype(np.int8),
        val=(val.astype(np.float16) if fp16 else val.astype(np.float32)),
        left=left.astype(np.int32),
        right=right.astype(np.int32),
        roots=roots.astype(np.int32),
        base_logit=np.float32(base_logit),
        nt=np.int64(nt),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", required=True, help="XGBoost booster (.json / .ubj / .bin)")
    ap.add_argument("--out-bin", required=True, help="output compact .bin path")
    ap.add_argument("--out-npz", default=None, help="optional output compact .npz path")
    ap.add_argument("--no-fp16", action="store_true",
                    help="keep val in full fp32 in BOTH outputs (default: quantize val to fp16)")
    args = ap.parse_args()

    bst = xgb.Booster()
    bst.load_model(args.booster)
    feat, val, left, right, roots, base_logit, nt = build_compact(bst)

    fp16 = not args.no_fp16
    # fp16 quantisation: store the values upcast from fp16 back to fp32, unless --no-fp16.
    val_for_output = val.astype(np.float16).astype(np.float32) if fp16 else val

    size = write_bin(args.out_bin, feat, val_for_output, left, right, roots, base_logit, nt)
    print(f"wrote {args.out_bin}: nNodes={feat.shape[0]} nTrees={nt} baseLogit={float(base_logit):.8f} "
          f"size={size} bytes (val precision={'fp16->fp32' if fp16 else 'fp32'})")

    if args.out_npz:
        write_npz(args.out_npz, feat, val, left, right, roots, base_logit, nt, fp16)
        print(f"wrote {args.out_npz}: val dtype={'float16' if fp16 else 'float32'}")


if __name__ == "__main__":
    main()
