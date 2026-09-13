#!/usr/bin/env python3
"""Sum the per-job accumulators written by blMaterialMapBuild and emit the compiled-in material table
src/BLMaterialMap<tag>.cc (the 1/X0 lattice kRho and the dE/dx lattice kDedx), or check an existing table
against them.

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
    numE = np.zeros((KNR, KNZ))
    numEI = np.zeros((KNR, KNZ))
    numEE = np.zeros((KNR, KNZ))
    nray = 0.0
    for p in paths:
        with open(p, "rb") as f:
            a, b = np.fromfile(f, dtype=np.int32, count=2)
            if (a, b) != (KNR, KNZ):
                sys.exit("%s: lattice %dx%d, expected %dx%d" % (p, a, b, KNR, KNZ))
            num += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            den += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            numE += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            numEI += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            numEE += np.fromfile(f, dtype=np.float64, count=a * b).reshape(a, b)
            nray += np.fromfile(f, dtype=np.float64, count=1)[0]
    safe = np.where(den > 0, den, 1.0)
    rho = np.where(den > 0, num / safe, 0.0).astype(np.float32)
    rhoE = np.where(den > 0, numE / safe, 0.0)
    safeE = np.where(numE > 0, numE, 1.0)
    lnI = np.where(numE > 0, numEI / safeE, 0.0)
    lnRhoE = np.where(numE > 0, numEE / safeE, 0.0)
    # per cell the triple (rho Z/A [mol/cm^3], <ln(I/eV)>, <ln(rho Z/A)>), xi-weighted log-means
    dedx = np.stack([rhoE, lnI, lnRhoE], axis=-1).astype(np.float32)
    return rho.reshape(-1), dedx.reshape(-1, 3), int(nray)


def tokens(flat):
    return ["%.6gf" % v if v != 0 else "0.0f" for v in flat]


def table_tokens(path):
    """The float literals of the table's data array, as (density tokens, dE/dx tokens).

    The array holds the whole density lattice, then the whole dE/dx lattice; a table older than the dE/dx
    columns holds the densities only and yields an empty dE/dx list.
    """
    txt = open(path).read()
    m = re.search(r"const float k\w+\[[^\]]*\] = \{", txt)
    if not m:
        return None, None
    body = re.sub(r"//[^\n]*", "", txt[m.end():txt.index("};", m.end())])
    toks = [t for t in body.replace(",", " ").split() if t]
    n = KNR * KNZ
    return toks[:n], toks[n:]


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
// LATTICE    0.5 cm radial x 1 cm z, four floats per cell (device buffer 2.24 MB: the density lattice, then
//            the dE/dx lattice).
// DEDX       Next to rho, each cell carries (rho Z/A [mol/cm^3], <ln(I/eV)>, <ln(rho Z/A)>): the electron
//            density per Avogadro (the Landau xi per cm of path) area-averaged like rho, and the xi-weighted
//            log-means of the mean excitation energy and of the electron density over the cell (the ln I and
//            plasma-energy terms of the Landau most-probable loss of a composite column). Materials are
//            identified per Geant4 step by their (density, X0) pair in the run's G4 material table
//            (BLMaterialTableDump).
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
    flat, dedx, nray = read_bins(paths)
    toks = tokens(flat)
    dtoks = [tokens(row) for row in dedx]
    print("%d accumulators, %d rays, %d cells, %d non-zero" % (len(paths), nray, flat.size, int((flat > 0).sum())))

    rc = 0
    if a.check:
        refRho, refDedx = table_tokens(a.check)
        if refRho is None:
            sys.exit("%s: no data array" % a.check)
        flatd = [t for row in dtoks for t in row]
        for name, mine, ref in (("density", toks, refRho), ("dE/dx", flatd, refDedx)):
            if not ref:
                print("%s: no %s values (a table older than the dE/dx columns)" % (a.check, name))
                continue
            nd = sum(1 for x, y in zip(mine, ref) if x != y) + abs(len(mine) - len(ref))
            print("%s: %s %d of %d values differ" % (a.check, name, nd, len(ref)))
            rc = 1 if nd else rc

    if a.out:
        with open(a.out, "w") as f:
            f.write(header(read_provenance(a.provenance), nray))
            f.write('#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"\n\n')
            f.write("namespace blMaterialMap {\n  namespace {\n")
            f.write("    // the whole density lattice [X0/cm], then the dE/dx triples of the same cells, both\n")
            f.write("    // kNZ-major: one array, so that one pointer reaches both (blMaterialMap::dedxOf).\n")
            f.write("    const float kMap[kBufferFloats] = {\n")
            for i in range(0, len(toks), 8):
                f.write("        " + " ".join(t + "," for t in toks[i:i + 8]) + "\n")
            for row in dtoks:
                f.write("        " + " ".join(t + "," for t in row) + "\n")
            f.write("    };\n  }  // namespace\n")
            f.write("  const float* blMaterialMapData() { return kMap; }\n")
            f.write("}  // namespace blMaterialMap\n")
        print("wrote %s" % a.out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
