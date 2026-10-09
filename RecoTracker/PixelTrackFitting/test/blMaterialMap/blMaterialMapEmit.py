#!/usr/bin/env python3
"""Emit a binary BL-fit material map BLMaterialMap_<T>_BP<bp>_v<n>.bin (format: BLMaterialMapFile.h)
from the per-job accumulators of blMaterialMapBuild, or check the accumulated values against an existing map.

  # accumulators -> .bin (+ an index line)
  blMaterialMapEmit.py --bins DIR --provenance FILE --tag T36 --beam-pipe "2030/v3" \
      --fingerprint 0x0123456789abcdef --positions POSITIONS.txt \
      --out BLMaterialMap_T36_BP2030v3_v1.bin [--index BLMaterialMap.index]
  # check (exit nonzero on any difference)
  blMaterialMapEmit.py --bins DIR --check TARGET.bin

Accumulators are summed in sorted file-name order; rho = num/den where the cell was sampled, 0
elsewhere. Every value written is np.float32(float("%.6g" % v)), zeros exactly 0.0.
--positions is the BLMaterialMapFingerprintDump dumpPositions file; its LEAF lines become the sensor
reference appendix (0.1 mm buckets, sorted by rawId) the ESProducer matches against the job's geometry.
The provenance is PROVENANCE.txt plus the ray count; a provenance "fingerprint:" line must agree with
--fingerprint. --check compares all 560000 values (rho lattice and dE/dx triples) and prints the first
differing cells.
"""
import argparse
import glob
import math
import os
import re
import struct
import sys

import numpy as np

KNR, KNZ = 250, 560  # blMaterialMap::kNR, kNZ (0.5 cm radial x 1 cm z lattice)
DR, DZ, ZMAX = 0.5, 1.0, 280.0  # blMaterialMap::kDR, kDZ, kZMAX
KSIZE = KNR * KNZ  # blMaterialMap::kSize; the map is rho[kSize] then the dE/dx triples of the same cells
KFLOATS = 4 * KSIZE  # the floats of blMaterialMap::Map; the body is KFLOATS float32 = 2240000 bytes

MAGIC = b"BLMM0001"
# the 84-byte little-endian header of BLMaterialMapFile.h
HEADER_FMT = "<8sIii fff 16s16sIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 84, HEADER_SIZE


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
    rho = np.where(den > 0, num / safe, 0.0)
    rhoE = np.where(den > 0, numE / safe, 0.0)
    safeE = np.where(numE > 0, numE, 1.0)
    lnI = np.where(numE > 0, numEI / safeE, 0.0)
    lnRhoE = np.where(numE > 0, numEE / safeE, 0.0)
    # per cell: rhoE [mol/cm^3], <ln(I/eV)>, <ln(rhoE)>
    dedx = np.stack([rhoE, lnI, lnRhoE], axis=-1)
    return rho.reshape(-1), dedx.reshape(-1, 3), int(nray)


def canonical(flat):
    """Elementwise (float("%.6g" % v), np.float32 of it): the value a "%.6gf" literal denotes, and the
    float32 written to the .bin and used in every comparison."""
    v64 = np.array([float("%.6g" % v) for v in np.asarray(flat, dtype=np.float64).reshape(-1)])
    return v64, v64.astype(np.float32)


def read_provenance(path):
    with open(path) as f:
        return f.read().strip("\n")


def fingerprint(text):
    """Argparse type of --fingerprint: the 64-bit geometry hash, with or without the 0x prefix."""
    try:
        fp = int(text, 16)  # base 16 accepts the "0x" prefix
    except ValueError:
        raise argparse.ArgumentTypeError("not a hexadecimal fingerprint: %r" % text)
    if not 0 <= fp <= 0xFFFFFFFFFFFFFFFF:
        raise argparse.ArgumentTypeError("fingerprint outside uint64: %r" % text)
    return fp


def uint32(text):
    try:
        v = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("not an integer: %r" % text)
    if not 0 <= v <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("outside uint32: %r" % text)
    return v


def write_bin(path, body32, tag, beam_pipe, map_version, fp, provenance, positions=None):
    """The 84-byte header, the provenance text, the 2240000 body bytes, then the optional sensor
    reference positions (rawId + x/y/z in 0.1 mm buckets, sorted by rawId)."""
    if tag.split() != [tag] or beam_pipe.split() != [beam_pipe]:
        sys.exit("tags must not contain whitespace (the index is space separated): %r, %r"
                 % (tag, beam_pipe))
    tag, beam_pipe = tag.encode("ascii"), beam_pipe.encode("ascii")
    if len(tag) >= 16 or len(beam_pipe) >= 16:
        sys.exit("tags must fit the 16-byte char fields with a NUL: %r, %r" % (tag, beam_pipe))
    prov = (provenance.rstrip("\n") + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, MAGIC, 1, KNR, KNZ, DR, DZ, ZMAX, tag, beam_pipe, map_version,
                            0, fp, len(prov)))
        f.write(prov)
        f.flush()  # tofile bypasses Python's buffer
        body32.astype("<f4").tofile(f)  # the rho lattice, then the dE/dx triples
        if positions is not None:
            positions.tofile(f)  # little-endian (rawId, x, y, z) int32 records, 16 bytes each


POS_DT = np.dtype([("rawId", "<u4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4")])


def read_positions(path):
    """The sensor reference positions from a BLMaterialMapFingerprintDump dumpPositions file: the LEAF
    lines, sorted by rawId and quantized to the 0.1 mm buckets of blMaterialMap::LeafPos."""
    # round half away from zero, as std::llround does
    q = lambda v: math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)
    pos = []
    for line in open(path):
        if line.startswith("LEAF "):
            _, rid, x, y, z = line.split()
            pos.append((int(rid), q(float(x) * 10), q(float(y) * 10), q(float(z) * 10)))
    if not pos:
        sys.exit("%s: no LEAF lines (run the dump with dumpPositions=True)" % path)
    pos.sort()
    print("%s: %d sensor reference positions" % (path, len(pos)))
    return np.array(pos, dtype=POS_DT)


def read_bin(path):
    """The body of a .bin, header and size checked; prints the header identity."""
    data = open(path, "rb").read()
    if len(data) < HEADER_SIZE:
        sys.exit("%s: %d bytes, shorter than the %d-byte header" % (path, len(data), HEADER_SIZE))
    magic, version, nr, nz, dr, dz, zmax, tag, bp, map_version, reserved, fp, plen = struct.unpack(
        HEADER_FMT, data[:HEADER_SIZE])
    if magic != MAGIC:
        sys.exit("%s: magic %r, expected %r" % (path, magic, MAGIC))
    if version != 1:
        sys.exit("%s: formatVersion %d, expected 1" % (path, version))
    if (nr, nz) != (KNR, KNZ) or (dr, dz, zmax) != (DR, DZ, ZMAX):
        sys.exit("%s: lattice %dx%d, %gx%g zmax %g, expected %dx%d, %gx%g zmax %g"
                 % (path, nr, nz, dr, dz, zmax, KNR, KNZ, DR, DZ, ZMAX))
    if reserved != 0:
        sys.exit("%s: reserved header word %d, expected 0" % (path, reserved))
    head = HEADER_SIZE + plen + KFLOATS * 4
    if len(data) < head or (len(data) - head) % 16:
        sys.exit("%s: %d bytes, expected 84 + %d provenance + %d body + 16*n positions"
                 % (path, len(data), plen, KFLOATS * 4))
    print("%s: geometry %s, beam pipe %s, map version %d, fingerprint 0x%016x, %d reference positions"
          % (path, tag.split(b"\0")[0].decode(), bp.split(b"\0")[0].decode(), map_version, fp,
             (len(data) - head) // 16))
    return np.frombuffer(data, dtype="<f4", count=KFLOATS, offset=head - KFLOATS * 4).astype(np.float32)


def cell(index):
    """What a flat body index is: a rho cell, or one component of a cell's dE/dx triple."""
    if index < KSIZE:
        return "rho cell (%d,%d)" % (index // KNZ, index % KNZ)
    j = index - KSIZE
    return "dE/dx cell (%d,%d) %s" % (j // 3 // KNZ, j // 3 % KNZ, ("rhoE", "lnI", "lnRhoE")[j % 3])


def update_index(path, fp, tag, beam_pipe, map_version, basename):
    """Insert or replace this map's line (key: fingerprint and file; one census may bind several files,
    T36-T38); '#' comments are kept above the entries; entries sorted by key; malformed lines are an
    error."""
    line = "0x%016x %s %s %d %s" % (fp, tag, beam_pipe, map_version, basename)
    comments, entries = [], {}
    if os.path.exists(path):
        for n, old in enumerate(open(path), 1):
            old = old.rstrip("\n")
            if not old.strip():
                continue
            if old.lstrip().startswith("#"):
                comments.append(old)  # keep the comments, above the entries
            else:
                fields = old.split()
                if len(fields) != 5 or not re.fullmatch(r"0x[0-9a-fA-F]{16}", fields[0]):
                    sys.exit("%s line %d: not '0x<16 hex> <tag> <beam-pipe> <version> <file>': %s"
                             % (path, n, old))
                entries[(int(fields[0], 16), fields[-1])] = old
    entries[(fp, basename)] = line
    with open(path, "w") as f:
        for c in comments:
            f.write(c + "\n")
        for k in sorted(entries):
            f.write(entries[k] + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True, help="directory of *.bin accumulators (or a glob), from the rays")
    ap.add_argument("--provenance", metavar="FILE",
                    help="PROVENANCE.txt written by blMaterialMapRun.sh, embedded verbatim (required with --out)")
    ap.add_argument("--tag", help="tracker geometry tag for the header, e.g. T36 (no spaces, <16 bytes)")
    ap.add_argument("--beam-pipe",
                    help='beam-pipe tag for the header, e.g. "2030/v3" (no spaces, <16 bytes)')
    ap.add_argument("--fingerprint", type=fingerprint,
                    help="64-bit geometry fingerprint (BLMaterialMapFingerprint), 0x-prefixed or bare hex")
    ap.add_argument("--map-version", type=uint32, default=1,
                    help="version of the map for this geometry and beam pipe (default 1)")
    ap.add_argument("--index", metavar="INDEXFILE",
                    help="with --out: add (or replace) the map's line in this BLMaterialMap.index")
    ap.add_argument("--positions", metavar="POSITIONS.txt",
                    help="with --out: the BLMaterialMapFingerprintDump dumpPositions file; its sensor "
                    "reference positions are appended to the written map")
    ap.add_argument("--out", metavar="OUT.bin", help="write the material map")
    ap.add_argument("--check", metavar="TARGET.bin", help="an existing map to compare the values with")
    a = ap.parse_args()
    if not (a.out or a.check):
        ap.error("give --out and/or --check")
    if a.out:
        for opt in ("tag", "beam_pipe", "fingerprint", "positions"):
            if getattr(a, opt) is None:
                ap.error("--out needs --%s" % opt.replace("_", "-"))
        if not a.provenance:
            ap.error("--out needs --provenance")
    if a.index and not a.out:
        ap.error("--index maintains the catalogue of written maps: give --out too")

    paths = sorted(glob.glob(os.path.join(a.bins, "*.bin")) if os.path.isdir(a.bins) else glob.glob(a.bins))
    if not paths:
        sys.exit("no accumulators under %s" % a.bins)
    rho, dedx, nray = read_bins(paths)
    flat = np.concatenate([rho, dedx.reshape(-1)])
    print("%d accumulators, %d rays, %d cells, %d non-zero" % (len(paths), nray, rho.size, int((rho > 0).sum())))
    v32 = canonical(flat)[1]

    rc = 0
    if a.check:
        # both arrays hold the same float32 format, so compare with ==
        mine, ref = v32, read_bin(a.check)
        n = min(mine.size, ref.size)
        nd = int((mine[:n] != ref[:n]).sum()) + abs(mine.size - ref.size)
        if nd:
            print("%s: %d of the %d map values differ" % (a.check, nd, max(mine.size, ref.size)))
            for i in np.nonzero(mine[:n] != ref[:n])[0][:5]:
                print("  %s: %.6g accumulated, %.6g in the target" % (cell(i), mine[i], ref[i]))
            rc = 1
        else:
            print("%s: 0 differences" % a.check)

    if a.out:
        provenance = read_provenance(a.provenance)
        provenance += "\n%d rays reached the selected volumes." % nray
        fpline = re.search(r"^fingerprint\s*:\s*(0x[0-9a-fA-F]+)", provenance, re.M)
        if fpline and int(fpline.group(1), 16) != a.fingerprint:
            sys.exit("provenance says fingerprint %s, --fingerprint is 0x%016x (stale provenance?)"
                     % (fpline.group(1), a.fingerprint))
        write_bin(a.out, v32, a.tag, a.beam_pipe, a.map_version, a.fingerprint, provenance,
                  read_positions(a.positions))
        print("wrote %s (0x%016x %s %s v%d)" % (a.out, a.fingerprint, a.tag, a.beam_pipe, a.map_version))
        if a.index:
            update_index(a.index, a.fingerprint, a.tag, a.beam_pipe, a.map_version, os.path.basename(a.out))
            print("index: %s" % a.index)
    sys.exit(rc)


if __name__ == "__main__":
    main()
