#!/usr/bin/env python3
"""Sum the per-job accumulators written by blMaterialMapBuild and emit the compiled-in material table
src/BLMaterialMap<tag>.cc, or check an existing table against them.

  blMaterialMapEmit.py --bins DIR --provenance FILE [--tag D121] --out BLMaterialMapD121.cc
  blMaterialMapEmit.py --bins DIR --check src/BLMaterialMapD121.cc

The accumulators are summed in sorted file-name order (the job order); rho = num/den per cell where the
cell was sampled and 0 elsewhere; every value is printed as a float with six significant digits. The
table's provenance header is written from the file blMaterialMapRun.sh records for the run (geometry,
era, release, beam pipe and material XML, ray sample), so it always describes the table it heads.
--check compares the values, token by token, with the kRho array of an existing table and exits
non-zero on any difference.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

KNR, KNZ = 250, 560  # blMaterialMap::kNR, kNZ (0.5 cm radial x 1 cm z lattice)


def read_bins(paths):
    num = np.zeros((KNR, KNZ))
    den = np.zeros((KNR, KNZ))
    nray = 0.0
    for p in paths:
        with open(p, "rb") as f:
            a, b = np.fromfile(f, dtype=np.int32, count=2)
            if (a, b) != (KNR, KNZ):
                sys.exit("%s: lattice %dx%d, expected %dx%d" % (p, a, b, KNR, KNZ))
            num += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            den += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            nray += np.fromfile(f, dtype=np.float64, count=1)[0]
    rho = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).astype(np.float32)
    return rho.reshape(-1), int(nray)


def tokens(flat):
    return ["%.6gf" % v if v != 0 else "0.0f" for v in flat]


def table_tokens(path):
    txt = open(path).read()
    m = re.search(r"kRho\w*\[kSize\] = \{", txt)
    if not m:
        sys.exit("%s: no kRho[kSize] array found" % path)
    body = txt[m.end():txt.index("};", m.end())]
    return [t for t in body.replace(",", " ").split() if t]


def read_provenance(path):
    prov = {}
    for line in open(path):
        if ":" in line:
            key, value = line.split(":", 1)
            prov[key.strip()] = value.strip()
    return prov


def header(prov, nray):
    def get(key):
        return prov.get(key, "unknown")

    return """\
// Geant4 material radiation-length density rho(r,z) [X0/cm] (Tracker + BeamPipe), phi-averaged, on the
// 0.5 cm radial lattice (250 x 560). HOST-ONLY data table, kept out of device-compiled code: the
// BLMaterialMap ESProducer uploads it for the device fits and the unit tests read the array directly.
//
// PROVENANCE. Made by test/blMaterialMap/blMaterialMapRun.sh (README there), which records the lines below.
// GEOMETRY   %s
//            beam pipe %s
//            materials %s
// ERA        %s
// RELEASE    %s
// GENERATOR  Validation/Geometry MaterialBudgetAction (AllStepsToTree), SelectedVolumes = {Tracker, BEAM};
//            probes = 10 GeV nu_mu, DummyPhysics + DummyEMPhysics, magnetic field OFF,
//            StackingAction.TrackNeutrino = true, vertex NoSmear at (0,0,0);
//            %s; %d rays reached the selected volumes.
// ESTIMATOR  Every Geant4 step is split exactly into the (r,z) cells it crosses (breakpoints at the roots of
//            |P(t)|_T = r_b and z(t) = z_b); a G4 step never crosses a volume boundary, so rho is constant
//            along it. Each deposit is weighted by the local radius, because rays from the origin flat in eta
//            carry path measure dr dz / r, so rho[cell] = sum(dmb * rbar) / sum(dl * rbar) is the exact
//            phi-averaged area average of the local 1/X0 over the cell.
// LATTICE    0.5 cm radial x 1 cm z (device buffer 560 kB).
// NOTE       Air (rho = 3.3e-5 /cm) is present in the table rather than stored as an exact zero: it is real
//            material and contributes ~0.010 X/X0 over a 300 cm forward path.
""" % (get("geometry"), get("beam pipe"), get("materials"), get("era"), get("release"), get("rays"), nray)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True, help="directory of *.bin accumulators (or a glob)")
    ap.add_argument("--provenance", help="PROVENANCE.txt written by blMaterialMapRun.sh (needed with --out)")
    ap.add_argument("--tag", default="D121", help="geometry tag, only used to name the output file")
    ap.add_argument("--out", help="table to write")
    ap.add_argument("--check", help="existing table to compare with")
    a = ap.parse_args()
    if not (a.out or a.check):
        ap.error("give --out and/or --check")
    if a.out and not a.provenance:
        ap.error("--out needs --provenance")

    paths = sorted(glob.glob(os.path.join(a.bins, "*.bin")) if os.path.isdir(a.bins) else glob.glob(a.bins))
    if not paths:
        sys.exit("no accumulators under %s" % a.bins)
    flat, nray = read_bins(paths)
    toks = tokens(flat)
    print("%d accumulators, %d rays, %d cells, %d non-zero" % (len(paths), nray, flat.size, int((flat > 0).sum())))

    rc = 0
    if a.check:
        ref = table_tokens(a.check)
        nd = sum(1 for x, y in zip(toks, ref) if x != y) + abs(len(toks) - len(ref))
        print("%s: %d of %d values differ" % (a.check, nd, len(ref)))
        rc = 1 if nd else 0

    if a.out:
        with open(a.out, "w") as f:
            f.write(header(read_provenance(a.provenance), nray))
            f.write('#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"\n\n')
            f.write("namespace blMaterialMap {\n  namespace {\n    const float kRho[kSize] = {\n")
            for i in range(0, len(toks), 8):
                f.write("        " + " ".join(t + "," for t in toks[i:i + 8]) + "\n")
            f.write("    };\n  }  // namespace\n")
            f.write("  const float* blMaterialMapData() { return kRho; }\n")
            f.write("}  // namespace blMaterialMap\n")
        print("wrote %s" % a.out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
