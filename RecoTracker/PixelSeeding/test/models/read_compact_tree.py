"""Read a compact forest .bin (the file PixelTrackForestHighPuritySelector loads) and score rows
with it, so a trained model can be checked against the file that ships, and a deployed file can be
evaluated on a new dataset (make_profile.py).

Format (export_compact_tree.py write_bin):
    int32 nNodes; int32 nTrees; float32 baseLogit;
    int8 splitFeature[nNodes]  (-1 => leaf)
    float32 value[nNodes]      (split threshold, or leaf value at a leaf)
    int32 left[nNodes]; int32 right[nNodes]; int32 roots[nTrees]

The traversal is the one the selector kernel performs: go LEFT when x[feature] < threshold, else
RIGHT (so a NaN feature compares false and goes RIGHT), sum the leaf values over all trees, add
baseLogit and apply the logistic function.

    python3 read_compact_tree.py <model.bin>     prints the header and the widest feature index
"""
import numpy as np
import struct


def load_compact(path):
    with open(path, "rb") as f:
        blob = f.read()
    nN, nT, base = struct.unpack_from("<iif", blob, 0)
    off = 12
    feat = np.frombuffer(blob, dtype=np.int8, count=nN, offset=off); off += nN
    val = np.frombuffer(blob, dtype=np.float32, count=nN, offset=off); off += 4 * nN
    left = np.frombuffer(blob, dtype=np.int32, count=nN, offset=off); off += 4 * nN
    right = np.frombuffer(blob, dtype=np.int32, count=nN, offset=off); off += 4 * nN
    roots = np.frombuffer(blob, dtype=np.int32, count=nT, offset=off); off += 4 * nT
    if off != len(blob):
        raise ValueError("%s: %d trailing bytes -- format mismatch" % (path, len(blob) - off))
    return {"nNodes": nN, "nTrees": nT, "baseLogit": float(base),
            "feat": feat, "val": val, "left": left, "right": right, "roots": roots}


def score_compact(m, X, chunk=200000):
    """Vectorised traversal. X: (n, nFeat) float. Returns P(signal) in [0,1]."""
    feat, val, left, right = m["feat"], m["val"], m["left"], m["right"]
    X = np.asarray(X, dtype=np.float32)
    out = np.empty(len(X), dtype=np.float64)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        acc = np.zeros(len(Xc), dtype=np.float64)
        for r in m["roots"]:
            node = np.full(len(Xc), r, dtype=np.int64)
            active = np.ones(len(Xc), dtype=bool)
            while active.any():
                nd = node[active]
                fi = feat[nd]
                isleaf = fi < 0
                idx = np.flatnonzero(active)
                if isleaf.any():
                    acc[idx[isleaf]] += val[nd[isleaf]]
                    active[idx[isleaf]] = False
                inner = ~isleaf
                if inner.any():
                    ii = idx[inner]
                    ndi = nd[inner]
                    xv = Xc[ii, fi[inner].astype(np.int64)]
                    goleft = xv < val[ndi]          # NaN -> False -> RIGHT, as in the kernel
                    node[ii] = np.where(goleft, left[ndi], right[ndi])
        out[lo:lo + chunk] = 1.0 / (1.0 + np.exp(-(acc + m["baseLogit"])))
    return out


if __name__ == "__main__":
    import sys
    m = load_compact(sys.argv[1])
    print("nNodes=%d nTrees=%d baseLogit=%.8f leaves=%d maxFeat=%d"
          % (m["nNodes"], m["nTrees"], m["baseLogit"], int((m["feat"] < 0).sum()),
             int(m["feat"].max())))
