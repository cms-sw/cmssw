#!/usr/bin/env python3
"""Emit a binary BL-fit material map BLMaterialMap_<T>_BP<bp>_v<n>.bin (format: BLMaterialMapFile.h)
from the per-job accumulators of blMaterialMapBuild, from the retired compiled-in table, or from an
existing map (re-emit with a new header, provenance or reference appendix; values untouched); or check
a source's values against an existing .bin or .cc.

  # fresh rays: accumulators -> .bin (+ an index line)
  blMaterialMapEmit.py --bins DIR --provenance FILE --tag T36 --beam-pipe "2030/v3" \
      --fingerprint 0x0123456789abcdef --positions POSITIONS.txt \
      --out BLMaterialMap_T36_BP2030v3_v1.bin [--index BLMaterialMap.index]
  # the compiled-in table -> .bin, no value change
  blMaterialMapEmit.py --from-cc src/BLMaterialMapD121.cc --tag T35 --beam-pipe "2030/v3" \
      --fingerprint 0x... --positions ... --out BLMaterialMap_T35_BP2030v3_v1.bin [--index ...]
  # re-emit an existing map (values untouched)
  blMaterialMapEmit.py --from-bin OLD.bin --tag T35 --beam-pipe "2030/v3" \
      --fingerprint 0x... --positions POSITIONS.txt --out NEW.bin [--provenance FILE] [--index ...]
  # checks (exit nonzero on any difference; TARGET is .bin or .cc, by extension)
  blMaterialMapEmit.py --bins DIR --check TARGET
  blMaterialMapEmit.py --from-cc TABLE.cc --check TARGET.bin
  blMaterialMapEmit.py --from-bin OLD.bin --check TARGET.bin

Accumulators are summed in sorted file-name order; rho = num/den where the cell was sampled, 0
elsewhere. Every value written is np.float32(float("%.6g" % v)), zeros exactly 0.0: a six-significant-
digit decimal is never within a float64 ulp of a float32 rounding boundary, so these are the floats the
compiler made of the old table's "%.6gf" literals, and --from-cc reads those literals directly.
--positions is the BLMaterialMapFingerprintDump dumpPositions file; its LEAF lines become the sensor
reference appendix (0.1 mm buckets, sorted by rawId) the ESProducer matches against the job's geometry.
Provenance: --bins embeds PROVENANCE.txt plus the ray count; --from-cc uses the table's own PROVENANCE
comment block (prefixed by --provenance if given) plus a conversion line; --from-bin keeps the old text
unless --provenance replaces it. A provenance "fingerprint:" line must agree with --fingerprint.
--check compares all 560000 values (rho lattice and dE/dx triples) and prints the first differing cells.
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
KFLOATS = 4 * KSIZE  # blMaterialMap::kBufferFloats; the body is KFLOATS float32 = 2240000 bytes

MAGIC = b"BLMM0001"
# header, little-endian, no padding: magic@0, formatVersion@8, nR@12, nZ@16,
# dR@20, dZ@24, zMax@28, geometryTag@32, beamPipeTag@48, mapVersion@64, reserved@68,
# fingerprint@72, provenanceLen@80
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
    # per cell the triple (rho Z/A [mol/cm^3], <ln(I/eV)>, <ln(rho Z/A)>), xi-weighted log-means
    dedx = np.stack([rhoE, lnI, lnRhoE], axis=-1)
    return rho.reshape(-1), dedx.reshape(-1, 3), int(nray)


def canonical(flat):
    """Elementwise (float("%.6g" % v), np.float32 of it): the value a "%.6gf" literal denotes, and the
    float32 written to the .bin and used in every comparison."""
    v64 = np.array([float("%.6g" % v) for v in np.asarray(flat, dtype=np.float64).reshape(-1)])
    return v64, v64.astype(np.float32)


def table_tokens(path):
    """The float literals of a table's data array, as (density tokens, dE/dx tokens).

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


def token_value(tok):
    """The value a "%.6gf" table literal denotes (strip the trailing 'f')."""
    return float(tok[:-1] if tok.endswith("f") else tok)


def cc_provenance(path):
    """A table's own PROVENANCE: its leading '//' comment block, stripped to plain text."""
    lines = []
    with open(path) as f:
        for line in f:
            if not line.startswith("//"):
                break
            line = line.rstrip("\n")[2:]
            lines.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(lines).strip("\n")


def read_provenance(path):
    with open(path) as f:
        return f.read().strip("\n")


def fingerprint(text):
    """Argparse type of --fingerprint: the 64-bit geometry hash, with or without the 0x prefix."""
    try:
        fp = int(text, 16)  # an explicit base 16 still accepts the "0x" prefix
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
        f.flush()  # tofile writes through the file descriptor, below Python's buffer: no reordering
        body32.astype("<f4").tofile(f)  # C order: the rho lattice, then the kNZ-major dE/dx triples
        if positions is not None:
            positions.tofile(f)  # little-endian (rawId, x, y, z) int32 records, 16 bytes each


POS_DT = np.dtype([("rawId", "<u4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4")])


def read_positions(path):
    """The sensor reference positions from a BLMaterialMapFingerprintDump dumpPositions file: the LEAF
    lines, sorted by rawId and quantized to the 0.1 mm buckets of blMaterialMap::LeafPos."""
    # half away from zero, matching the std::llround of the producer (Python's round() is half-to-even)
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


def read_bin(path, with_positions=False):
    """The body of a .bin, header and size checked; prints the header identity. With with_positions
    also returns the sensor reference appendix (empty if the file carries none)."""
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
    values = np.frombuffer(data, dtype="<f4", count=KFLOATS, offset=head - KFLOATS * 4).astype(np.float32)
    if not with_positions:
        return values
    npos = (len(data) - head) // 16
    positions = np.frombuffer(data, dtype=POS_DT, count=npos, offset=head).copy()
    return values, positions


def read_bin_provenance(path):
    """Just the header identity fields and the provenance text of a .bin."""
    data = open(path, "rb").read(HEADER_SIZE)
    magic, version, nr, nz, dr, dz, zmax, tag, bp, map_version, reserved, fp, plen = struct.unpack(
        HEADER_FMT, data)
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        prov = f.read(plen).decode("utf-8")
    return tag.split(b"\0")[0].decode(), bp.split(b"\0")[0].decode(), map_version, fp, prov


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
                comments.append(old)  # comments are kept, hoisted above the entries
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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bins", help="directory of *.bin accumulators (or a glob), from fresh rays")
    src.add_argument("--from-cc", metavar="TABLE.cc",
                     help="the compiled-in table to convert (its literals are already canonical)")
    src.add_argument("--from-bin", metavar="MAP.bin",
                     help="an existing map to re-emit unchanged (its floats are already canonical), e.g. "
                     "to regenerate the header or the reference appendix")
    ap.add_argument("--provenance", metavar="FILE",
                    help="PROVENANCE.txt written by blMaterialMapRun.sh; embedded verbatim (required "
                    "with --bins --out), prepended to the table's own PROVENANCE block (with --from-cc), "
                    "or replacing the old file's text (with --from-bin, which otherwise keeps it)")
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
    ap.add_argument("--out", metavar="OUT.bin", help="write the material map (required in the write modes)")
    ap.add_argument("--check", metavar="TARGET",
                    help=".bin or .cc (auto-detected by extension) to compare the values with")
    a = ap.parse_args()
    if not (a.out or a.check):
        ap.error("give --out and/or --check")
    if a.out:
        for opt in ("tag", "beam_pipe", "fingerprint", "positions"):
            if getattr(a, opt) is None:
                ap.error("--out needs --%s" % opt.replace("_", "-"))
        if a.bins and not a.provenance:
            ap.error("--bins --out needs --provenance")
    if a.index and not a.out:
        ap.error("--index maintains the catalogue of written maps: give --out too")

    # the candidate value set: from fresh accumulators, or the compiled-in table's own literals
    nray = 0
    if a.bins:
        paths = sorted(
            glob.glob(os.path.join(a.bins, "*.bin")) if os.path.isdir(a.bins) else glob.glob(a.bins))
        if not paths:
            sys.exit("no accumulators under %s" % a.bins)
        rho, dedx, nray = read_bins(paths)
        flat = np.concatenate([rho, dedx.reshape(-1)])
        print("%d accumulators, %d rays, %d cells, %d non-zero"
              % (len(paths), nray, rho.size, int((rho > 0).sum())))
    elif a.from_cc:
        rtoks, dtoks = table_tokens(a.from_cc)
        if rtoks is None:
            sys.exit("%s: no data array" % a.from_cc)
        if len(rtoks) != KSIZE or len(dtoks) != 3 * KSIZE:
            sys.exit("%s: %d rho + %d dE/dx literals, expected %d + %d"
                     % (a.from_cc, len(rtoks), len(dtoks), KSIZE, 3 * KSIZE))
        flat = np.array([token_value(t) for t in rtoks + dtoks])
        print("%s: %d rho + %d dE/dx literals" % (a.from_cc, len(rtoks), len(dtoks)))
    else:
        flat = read_bin(a.from_bin).astype(np.float64)
    v64, v32 = canonical(flat)

    rc = 0
    if a.check:
        if a.check.endswith(".bin"):
            # both arrays are np.float32(float("%.6g" % v)) of the same cells: plain float equality
            mine, ref = v32, read_bin(a.check)
        elif a.check.endswith(".cc"):
            rtoks, dtoks = table_tokens(a.check)
            if rtoks is None:
                sys.exit("%s: no data array" % a.check)
            # token by token over BOTH lattices: the float value of each literal against the candidate's
            # canonical value (comparing values, a table's 0.0f matches a candidate's 0)
            mine, ref = v64, np.array([token_value(t) for t in rtoks + dtoks])
        else:
            sys.exit("%s: --check auto-detects .bin and .cc targets by extension" % a.check)
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
        if a.bins:
            provenance = read_provenance(a.provenance)
            provenance += "\n%d rays reached the selected volumes." % nray
        elif a.from_cc:
            provenance = cc_provenance(a.from_cc)
            if a.provenance:
                provenance = read_provenance(a.provenance) + "\n\n" + provenance
            provenance += "\nconverted from %s with no value change" % os.path.basename(a.from_cc)
        else:
            # --provenance replaces the old file's text; without it the old text is kept verbatim
            provenance = read_provenance(a.provenance) if a.provenance \
                else read_bin_provenance(a.from_bin)[4]
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
