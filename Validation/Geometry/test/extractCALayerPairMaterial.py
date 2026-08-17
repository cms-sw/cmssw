#!/usr/bin/env python3
"""Derive a single normal-incidence X/X0 per CA layer-pair from the Geant4
material-budget tree, driven by the CA layer/layerPairs configuration.

INPUTS
  --tree   matbdg_tree_<label>_<geom>.root  (MaterialBudgetAction, AllStepsToTree=True;
           run e.g. label=Tracker so ALL tracker material is in one tree)
  --cfi    RecoTracker/PixelSeeding/python/caHitNtupletAlpakaPhase2OTStubs_cfi.py
           (the `layers` table -> isBarrel, and the `layerPairs` table -> (i,o) pairs)
  --layer-geom  CSV "layer,isBarrel,r,z" giving each CA layer's representative
           position [cm].  If omitted, --auto derives a FIRST GUESS from the
           sensitive (silicon) crossings in the tree and prints it for you to verify.

METHOD
  1. Build rho(r,z) = sum(DeltaMB)/sum(step_length)  [X0/cm], phi-averaged, from the
     per-step branches (DeltaMB + Initial/Final X,Y,Z). Same per category.
  2. For each layer-pair (i,o), integrate rho between the two layers AT NORMAL INCIDENCE:
       barrel-barrel        -> radial integral  int rho(r, z~0) dr      (radial = normal)
       disk-disk (same side)-> axial  integral  int rho(r~rmid, z) dz   (axial  = normal)
       mixed/transition     -> straight chord integral (OBLIQUE, flagged: normal~=chord*cos)
  3. Emit caLayerPairMaterial.h (constexpr X/X0 per pair, indexed as in layerPairs),
     a CSV with the per-category breakdown, a rho(r,z) diagnostic PNG, and the coarsened
     rho(r,z) lookup (<out>_rhomap.h/.cc) that the broken-line fits integrate along the
     track (the production copy is RecoTracker/PixelTrackFitting's BLMaterialMap).

--phi-scan is a read-only diagnostic of the phi-averaging assumption (see build_phi_density).
"""
import argparse, ast, re, sys
import numpy as np

# ----------------------------------------------------------------------------- CA cfg
def _extract_list(text, name):
    """Pull the python list literal `name = [ ... ]` out of the cfi, strip # comments."""
    m = re.search(rf'{name}\s*=\s*\[', text)
    if not m:
        raise RuntimeError(f"'{name}' not found in cfi")
    i = text.index('[', m.start()); depth = 0
    for j in range(i, len(text)):
        depth += (text[j] == '[') - (text[j] == ']')
        if depth == 0:
            break
    body = '\n'.join(re.sub(r'#.*', '', ln) for ln in text[i:j+1].splitlines())
    return ast.literal_eval(body)

def parse_cfi(path):
    txt = open(path).read()
    layers = _extract_list(txt, 'layers')        # [idx,isBarrel,...]
    pairs  = _extract_list(txt, 'layerPairs')     # [i,o,...]
    isB = {row[0]: bool(row[1]) for row in layers}
    pr  = [(row[0], row[1]) for row in pairs]
    return isB, pr

# ----------------------------------------------------------------------------- tree -> rho(r,z)
RMAX, ZMAX, DR, DZ = 125.0, 280.0, 0.5, 1.0
CATS = ['', '_SEN', '_SUP', '_CAB', '_COL', '_ELE', '_OTH']  # '' = total

def build_density(tree_paths):
    """Build rho(r,z) summed over one OR MORE material trees (e.g. Tracker + BeamPipe), so the
    beam pipe / inner services are included for the upstream (inner) term."""
    import uproot
    if isinstance(tree_paths, str):
        tree_paths = [tree_paths]
    nr, nz = int(RMAX/DR), int(2*ZMAX/DZ)
    num = {c: np.zeros((nr, nz)) for c in CATS}   # sum DeltaMB
    den = np.zeros((nr, nz))                       # sum step length
    sil = np.zeros((nr, nz))                       # silicon X/X0 map (layer locator)
    MM2CM = 0.1   # Geant4 stores G4 positions in mm (CLHEP); we bin in cm
    # silicon radiation length window [mm] -> detect sensor layers independently of the
    # (often-unconfigured) material categorizer. Si X0 = 93.7 mm; tight window avoids Al (89 mm).
    SI_LO, SI_HI = 91.0, 97.0
    br = (['Initial X','Initial Y','Initial Z','Final X','Final Y','Final Z']
          + ['DeltaMB'+c for c in CATS] + ['Material X0'])
    rmax_seen = 0.0
    for tree_path in tree_paths:
        f = uproot.open(tree_path)
        cn = f.classnames(recursive=True)   # MaterialBudgetTree writes "T1" inside "TEST"
        trees = [name for name, cls in cn.items() if cls == 'TTree']
        if not trees:
            raise RuntimeError(f"no TTree in {tree_path}; file contents: {cn}")
        t = f[trees[0]]
        print(f"  reading {tree_path}: TTree '{trees[0]}' ({t.num_entries} entries)")
        for batch in t.iterate(br, library='np', step_size=2000):
            ix,iy,iz = batch['Initial X'],batch['Initial Y'],batch['Initial Z']
            fx,fy,fz = batch['Final X'],batch['Final Y'],batch['Final Z']
            dmb = {c: batch['DeltaMB'+c] for c in CATS}
            mx0 = batch['Material X0']
            for ev in range(len(ix)):
                x0,y0,z0 = ix[ev]*MM2CM, iy[ev]*MM2CM, iz[ev]*MM2CM
                x1,y1,z1 = fx[ev]*MM2CM, fy[ev]*MM2CM, fz[ev]*MM2CM
                dl = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)        # step length [cm]
                rm = 0.5*(np.hypot(x0,y0)+np.hypot(x1,y1)); zm = 0.5*(z0+z1)
                if rm.size: rmax_seen = max(rmax_seen, float(rm.max()))
                ir = (rm/DR).astype(int); izb = ((zm+ZMAX)/DZ).astype(int)
                ok = (ir>=0)&(ir<nr)&(izb>=0)&(izb<nz)
                np.add.at(den, (ir[ok], izb[ok]), dl[ok])
                for c in CATS:
                    np.add.at(num[c], (ir[ok], izb[ok]), dmb[c][ev][ok])
                si = ok & (mx0[ev] > SI_LO) & (mx0[ev] < SI_HI)        # silicon steps
                np.add.at(sil, (ir[si], izb[si]), dmb[''][ev][si])
    rho = {c: np.divide(num[c], den, out=np.zeros_like(den), where=den>0) for c in CATS}
    rho['_SI'] = sil    # silicon X/X0 map, used by auto_layer_geom for layer detection
    rho['_DEN'] = den   # path-length map, used to coarsen rho correctly for the exported map
    print(f"  density built: max r seen = {rmax_seen:.1f} cm, total path = {den.sum():.0f} cm")
    print(f"  X/X0 sums per category: " + ", ".join(f"{c or 'TOT'}={num[c].sum():.1f}" for c in CATS)
          + f", SI={sil.sum():.1f}")
    if den.sum() == 0:
        raise RuntimeError("empty density map -- check position units / branch names")
    return rho   # rho[c][ir,iz] = X0/cm  (+ rho['_SI'] = silicon X/X0 map)

def rbin(r): return min(max(int(r/DR),0), int(RMAX/DR)-1)
def zbin(z): return min(max(int((z+ZMAX)/DZ),0), int(2*ZMAX/DZ)-1)

# --------------------------------------------------------------- auto layer geometry (count-driven)
# Canonical CA layer order (from the cfi `layers` table comments). The link to the geometry is
# purely combinatorial: same physical layers, same order. We detect the silicon peaks, bucket
# them, sort, and assign by position -- the per-bucket COUNT is the consistency check.
R_ITOT = 20.0        # [cm] IT/OT radius split
Z_BARREL = 5.0       # [cm] |z| half-window for the barrel radial peaks (only barrel near z=0)
Z_OT_DISK = 118.0    # [cm] |z| above which OT crossings are disks (below = OT barrel)
# CA index ranges and their expected (count, kind, side) -- side: +1 fwd(z>0), -1 bwd(z<0), 0 barrel
CA_BUCKETS = [(range(0, 4),   4, 'barrel', 0, 'IT'),
              (range(4, 16),  12, 'disk', +1, 'IT'),   # IT forward disks
              (range(16, 28), 12, 'disk', -1, 'IT'),   # IT backward disks
              (range(28, 34), 6, 'barrel', 0, 'OT'),
              (range(34, 39), 5, 'disk', -1, 'OT'),    # OT backward disks
              (range(39, 44), 5, 'disk', +1, 'OT')]    # OT forward disks

def _peaks(prof, vals, min_frac=0.05, sep=2):
    """Local maxima above min_frac*max -> list of (position, height)."""
    thr = min_frac * prof.max() if prof.max() > 0 else 1e9
    out, i = [], 1
    while i < len(prof) - 1:
        if prof[i] > thr and prof[i] >= prof[i-1] and prof[i] > prof[i+1]:
            out.append((vals[i], prof[i])); i += sep
        else:
            i += 1
    return out

def _merge_to_n(peaks, n):
    """Count-driven: merge the closest adjacent (pos,height) peaks (height-weighted) until exactly
    n remain. Returns sorted positions, or None if there are fewer than n candidates to start."""
    pts = sorted(peaks)
    if len(pts) < n:
        return None
    while len(pts) > n:
        i = int(np.argmin([pts[k+1][0] - pts[k][0] for k in range(len(pts)-1)]))
        w = pts[i][1] + pts[i+1][1]
        v = ((pts[i][0]*pts[i][1] + pts[i+1][0]*pts[i+1][1]) / w) if w > 0 else 0.5*(pts[i][0]+pts[i+1][0])
        pts = pts[:i] + [(v, w)] + pts[i+2:]
    return [v for v, _ in pts]

def auto_layer_geom(det):
    """Detect physical layers from the silicon DENSITY map and assign to CA indices, merging the
    over-detected peaks down to each bucket's known count (handles 2-sensor stacks / double-disks).
    Returns geom{idx:(isBarrel,r,z)} or raises if a bucket has fewer candidates than needed."""
    rvals = np.arange(det.shape[0]) * DR
    zvals = np.arange(det.shape[1]) * DZ - ZMAX
    rcand = _peaks(det[:, zbin(-Z_BARREL):zbin(Z_BARREL)].sum(axis=1), rvals)   # barrel radii
    it_bar = _merge_to_n([p for p in rcand if p[0] < R_ITOT], 4)
    ot_bar = _merge_to_n([p for p in rcand if p[0] >= R_ITOT], 6)
    def disks(r_lo, r_hi, zmin, nf, nb):
        pk = _peaks(det[rbin(r_lo):rbin(r_hi), :].sum(axis=0), zvals)
        fwd = _merge_to_n([(p[0], p[1]) for p in pk if p[0] > zmin], nf)
        bwd = _merge_to_n([(abs(p[0]), p[1]) for p in pk if p[0] < -zmin], nb)
        return fwd, bwd, 0.5*(r_lo + r_hi)
    it_fwd, it_bwd, it_rmid = disks(4.0, R_ITOT, Z_BARREL + 20, 12, 12)
    ot_fwd, ot_bwd, ot_rmid = disks(R_ITOT, RMAX, Z_OT_DISK, 5, 5)
    fmt = lambda L: 'NONE(too few)' if L is None else [f'{x:.1f}' for x in L]
    print("  [auto] layer positions after merge-to-count:")
    print(f"    IT barrel r {fmt(it_bar)}   OT barrel r {fmt(ot_bar)}")
    print(f"    IT disk |z| fwd {fmt(it_fwd)}  bwd {fmt(it_bwd)}")
    print(f"    OT disk |z| fwd {fmt(ot_fwd)}  bwd {fmt(ot_bwd)}")
    found = {('IT','barrel',0): [(True, r, 0.0) for r in (it_bar or [])],
             ('IT','disk',+1):  [(False, it_rmid, z) for z in (it_fwd or [])],
             ('IT','disk',-1):  [(False, it_rmid, -z) for z in (it_bwd or [])],
             ('OT','barrel',0): [(True, r, 0.0) for r in (ot_bar or [])],
             ('OT','disk',+1):  [(False, ot_rmid, z) for z in (ot_fwd or [])],
             ('OT','disk',-1):  [(False, ot_rmid, -z) for z in (ot_bwd or [])]}
    geom, ok = {}, True
    for idxs, n, kind, side, reg in CA_BUCKETS:
        lst = found[(reg, kind, side)]
        if len(lst) != n:
            ok = False; print(f"    *** {reg} {kind} side={side:+d}: have {len(lst)}, need {n}")
        for k, idx in enumerate(idxs):
            if k < len(lst):
                _, r, z = lst[k]; geom[idx] = (kind == 'barrel', r, z)
    if not ok:
        raise RuntimeError("layer count mismatch -- inspect caLayerPairMaterial_rho.png (silicon panel) "
                           "and adjust R_ITOT/Z_BARREL/Z_OT_DISK or the peak threshold")
    print("  [auto] all buckets match the CA config -> layer<->geometry link established.")
    return geom

def load_layer_geom(csv):
    g = {}
    for ln in open(csv):
        ln = ln.split('#')[0].strip()
        if not ln or ln.lower().startswith('layer'):
            continue
        idx, isb, r, z = ln.split(',')
        g[int(idx)] = (bool(int(isb)), float(r), float(z))
    return g

# ----------------------------------------------------------------------------- rho(r,z) map export
def export_rho_map(rho, out, mdr, mdz, tag):
    """Export the material density rho(r,z) [X0/cm] as a device-ready constexpr lookup, coarsened to
    (mdr x mdz) cm bins. The BL fit integrates this along the actual hit->hit segments (and IP->first
    hit) -> handles any segment (configured pair or not) at the real track angle."""
    den = rho['_DEN']; numtot = rho[''] * den   # numtot = sum of DeltaMB per fine bin
    fr, fz = max(1, int(round(mdr/DR))), max(1, int(round(mdz/DZ)))
    nrc, nzc = rho[''].shape[0]//fr, rho[''].shape[1]//fz
    numc = numtot[:nrc*fr, :nzc*fz].reshape(nrc, fr, nzc, fz).sum(axis=(1, 3))
    denc = den[:nrc*fr, :nzc*fz].reshape(nrc, fr, nzc, fz).sum(axis=(1, 3))
    rhoc = np.divide(numc, denc, out=np.zeros_like(numc), where=denc > 0)   # path-weighted coarse rho
    drc, dzc = fr*DR, fz*DZ
    flat = rhoc.reshape(-1)   # row-major: index = ir*kNZ + iz
    def _flit(v):                       # valid C++ float literal (0 -> 0.0f, not the invalid 0f)
        s = f"{v:.5g}"
        if not any(c in s for c in '.eE'):
            s += '.0'
        return s + 'f'

    # Two-file layout: a LIGHT, device-safe header (metadata + pointer-based rhoAt + a host
    # blMaterialMapData() declaration) and a HOST-ONLY .cc holding the big array. The values are
    # uploaded once to a device buffer by the BL fit and rhoAt() reads them through the passed
    # pointer. Keeping the array out of device-compiled code avoids the nvcc compile blow-up and the
    # on-device host-address fault of a baked-in constexpr table; resolution becomes a data-only swap.
    # The production copies are RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h and
    # src/BLMaterialMap<tag>.cc; the accessor and the array carry the geometry tag (--tag, e.g. D121).
    with open(out+'_rhomap.h', 'w') as fh:
        fh.write("// auto-generated by extractCALayerPairMaterial.py\n")
        fh.write("// Geant4 material radiation-length density rho(r,z) [X0/cm] (Tracker + BeamPipe).\n")
        fh.write("// Integrate along a track segment -> its X/X0. The VALUES live in the host-only\n")
        fh.write(f"// companion BLMaterialMap{tag}.cc (blMaterialMapData{tag}()), uploaded to a device buffer by\n")
        fh.write("// the BL fit; rhoAt() reads that buffer through the passed pointer.\n")
        fh.write("#ifndef RecoTracker_PixelTrackFitting_BLMaterialMap_h\n")
        fh.write("#define RecoTracker_PixelTrackFitting_BLMaterialMap_h\n")
        fh.write("namespace blMaterialMap {\n")
        fh.write(f"  constexpr int   kNR   = {nrc};\n  constexpr int   kNZ   = {nzc};\n")
        fh.write(f"  constexpr float kDR   = {drc:.4f}f;  // cm\n  constexpr float kDZ   = {dzc:.4f}f;  // cm\n")
        fh.write(f"  constexpr float kZMAX = {ZMAX:.1f}f;  // cm; z in [-kZMAX, +kZMAX]\n")
        fh.write("  constexpr int   kSize = kNR * kNZ;\n")
        fh.write(f"  // Host pointer to the kNR*kNZ row-major density grid [X0/cm] (defined in BLMaterialMap{tag}.cc).\n")
        fh.write(f"  const float* blMaterialMapData{tag}();\n")
        fh.write("  // local density [X0/cm] at (r,z) read from rho (host array or uploaded device buffer);\n")
        fh.write("  // 0 outside the grid. constexpr so it is callable from device code (deref at run time).\n")
        fh.write("  constexpr inline float rhoAt(const float* rho, float r, float z) {\n")
        fh.write("    int ir = int(r / kDR), iz = int((z + kZMAX) / kDZ);\n")
        fh.write("    if (ir < 0 || ir >= kNR || iz < 0 || iz >= kNZ)\n      return 0.f;\n")
        fh.write("    return rho[ir * kNZ + iz];\n  }\n")
        fh.write("}  // namespace blMaterialMap\n#endif\n")
    with open(out+'_rhomap.cc', 'w') as fc:
        fc.write("// auto-generated companion to BLMaterialMap.h -- Geant4 material radiation-length density\n")
        fc.write("// rho(r,z) [X0/cm] (Tracker + BeamPipe). HOST-ONLY data table, compiled once by the host\n")
        fc.write("// compiler and uploaded to a device buffer by the BL fit. Kept OUT of device-compiled code.\n")
        fc.write('#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"\n\n')
        fc.write("namespace blMaterialMap {\n")
        fc.write("  namespace {\n")
        fc.write(f"    const float kRho{tag}[kNR * kNZ] = {{\n")
        for s in range(0, len(flat), 12):
            fc.write("      " + ", ".join(_flit(v) for v in flat[s:s+12]) + ",\n")
        fc.write("    };\n  }  // namespace\n")
        fc.write(f"  const float* blMaterialMapData{tag}() {{ return kRho{tag}; }}\n")
        fc.write("}  // namespace blMaterialMap\n")
    print(f"  wrote {out}_rhomap.h + {out}_rhomap.cc  ({nrc}x{nzc} grid @ {drc:.1f}x{dzc:.1f} cm, "
          f"{4*nrc*nzc/1024:.0f} KB data)")

# ----------------------------------------------------------------------------- per-pair integral
def pair_material(i, o, geom, isB, rho):
    """Material a track FROM THE IP sees between the two layers of the pair.
    Procedure: draw the straight line from (0,0) to the OUTER layer's
    mid-point; find where it crosses the INNER layer surface; integrate rho along that segment.
    This is a real IP trajectory (the inner hit is the line-inner-layer intersection, not the inner
    layer centre)."""
    a, b = geom[i], geom[o]
    inner, outer = (a, b) if (a[1]**2 + a[2]**2) <= (b[1]**2 + b[2]**2) else (b, a)
    b_in, r_in, z_in = inner
    b_out, r_out, z_out = outer
    if r_in == 0 and z_in == 0:                 # inner = virtual IP (the upstream/inner term)
        t, rx, zx, mode = 0.0, 0.0, 0.0, 'IP->layer'
    elif b_in:                                  # inner barrel cylinder at r_in
        t = r_in / r_out if r_out > 0 else 2.0
        rx, zx, mode = r_in, t * z_out, 'barrel-in'
    else:                                       # inner disk plane at z_in
        t = z_in / z_out if z_out != 0 else 2.0
        rx, zx, mode = t * r_out, z_in, 'disk-in'
    if not (0.0 <= t <= 1.0):                   # line doesn't cross the inner layer between IP & outer
        rx, zx, mode = r_in, z_in, mode + '(fallback:centre)'
    Llen = np.hypot(r_out - rx, z_out - zx)     # segment length [cm] along the IP line
    n = max(2, int(Llen / 0.1))                 # 0.1 cm sampling << 0.5 cm bin -> integral
    rs = np.linspace(rx, r_out, n); zs = np.linspace(zx, z_out, n); dl = Llen / (n - 1)  # alignment-insensitive
    out = {c: sum(rho[c][rbin(r), zbin(z)] * dl for r, z in zip(rs, zs)) for c in CATS}
    out['mode'] = mode
    return out

# ----------------------------------------------------------------------------- main
# ---------------------------------------------------------------------------------------------------
# --phi-scan: quantify the phi-spread (diagnostic; does NOT touch the emitted production map).
#
# The production map is phi-averaged: rho(r,z) = sum(DeltaMB)/sum(l) over all azimuths. That assumption
# is weakest at small r (few modules, so a large module/gap/overlap contrast; flat modules spanning
# several r-cells; asymmetric services), and it lands on sigma(d0) because the innermost scatterer has
# the longest lever arm to the IP. This mode rebuilds the same estimator with an extra phi axis and
# reports, per (r,z) cell,
#   spread(r,z)    = sigma_phi(rho)/<rho>_w  (path-weighted std over phi, normalised to the phi-mean)
#   occupancy(r,z) = fraction of phi bins with non-zero sampled path (low occupancy = azimuthal gaps).
# Two numbers because a true vacuum gap (rho=0 at some phi) and an unsampled bin both give den=0: the
# path-weighted spread ignores den=0 bins, the occupancy exposes exactly those gaps. phi comes from the
# per-step Initial/Final X,Y branches (atan2), so no Geant4 re-run is needed, only the step tree.
def build_phi_density(tree_paths, nphi):
    """rho(r,phi,z) [X0/cm], total category only, same binning as build_density (+ a phi axis)."""
    import uproot
    if isinstance(tree_paths, str):
        tree_paths = [tree_paths]
    nr, nz = int(RMAX/DR), int(2*ZMAX/DZ)
    num = np.zeros((nr, nphi, nz)); den = np.zeros((nr, nphi, nz))
    MM2CM = 0.1
    br = ['Initial X','Initial Y','Initial Z','Final X','Final Y','Final Z','DeltaMB']
    for tree_path in tree_paths:
        f = uproot.open(tree_path)
        trees = [name for name, cls in f.classnames(recursive=True).items() if cls == 'TTree']
        if not trees:
            raise RuntimeError(f"no TTree in {tree_path}")
        t = f[trees[0]]
        print(f"  [phi-scan] reading {tree_path}: '{trees[0]}' ({t.num_entries} entries), nphi={nphi}")
        for batch in t.iterate(br, library='np', step_size=2000):
            ix,iy,iz = batch['Initial X'],batch['Initial Y'],batch['Initial Z']
            fx,fy,fz = batch['Final X'],batch['Final Y'],batch['Final Z']
            dmb = batch['DeltaMB']
            for ev in range(len(ix)):
                x0,y0,z0 = ix[ev]*MM2CM, iy[ev]*MM2CM, iz[ev]*MM2CM
                x1,y1,z1 = fx[ev]*MM2CM, fy[ev]*MM2CM, fz[ev]*MM2CM
                dl = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
                xm, ym = 0.5*(x0+x1), 0.5*(y0+y1)
                rm = 0.5*(np.hypot(x0,y0)+np.hypot(x1,y1)); zm = 0.5*(z0+z1)
                ph = np.arctan2(ym, xm)                                   # [-pi, pi)
                ir  = (rm/DR).astype(int); izb = ((zm+ZMAX)/DZ).astype(int)
                iph = ((ph + np.pi)/(2*np.pi)*nphi).astype(int) % nphi
                ok = (ir>=0)&(ir<nr)&(izb>=0)&(izb<nz)
                np.add.at(den, (ir[ok], iph[ok], izb[ok]), dl[ok])
                np.add.at(num, (ir[ok], iph[ok], izb[ok]), dmb[ev][ok])
    return num, den   # rho(r,phi,z) = num/den where den>0

def phi_scan_report(num, den, out, layers_barrel_r, r_report=16.0, z_barrel=20.0):
    """Path-weighted phi-spread + phi-occupancy; per-layer numbers + inner-region plots. Read-only diagnostic."""
    nr, nphi, nz = num.shape
    sden = den.sum(axis=1)                                                # path summed over phi  (r,z)
    snum = num.sum(axis=1)
    mean = np.divide(snum, sden, out=np.zeros_like(sden), where=sden>0)   # = the phi-averaged map value
    rho  = np.divide(num, den, out=np.zeros_like(num), where=den>0)       # rho(r,phi,z)
    var  = np.divide((den*(rho - mean[:,None,:])**2).sum(axis=1), sden,
                     out=np.zeros_like(sden), where=sden>0)               # path-weighted variance over phi
    spread = np.divide(np.sqrt(var), mean, out=np.zeros_like(mean), where=mean>0)
    occ    = (den > 0).sum(axis=1) / float(nphi)                          # phi coverage fraction (r,z)
    # barrel slice |z| < z_barrel, profile vs r
    iz0, iz1 = int((-z_barrel+ZMAX)/DZ), int((z_barrel+ZMAX)/DZ)
    w = sden[:, iz0:iz1]
    prof_spread = np.divide((spread[:,iz0:iz1]*w).sum(1), w.sum(1), out=np.zeros(nr), where=w.sum(1)>0)
    prof_occ    = np.divide((occ[:,iz0:iz1]   *w).sum(1), w.sum(1), out=np.zeros(nr), where=w.sum(1)>0)
    rgrid = (np.arange(nr)+0.5)*DR
    print(f"\n  [phi-scan] barrel (|z|<{z_barrel:g} cm) phi-spread sigma_phi(rho)/<rho> and phi-occupancy at the layer radii:")
    print(f"  {'r[cm]':>7} {'spread':>9} {'phi-occ':>7}")
    for rl in layers_barrel_r:
        ib = min(max(int(rl/DR),0), nr-1)
        print(f"  {rl:7.2f} {prof_spread[ib]:9.2f} {prof_occ[ib]:7.2f}")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        irmax = int(r_report/DR)
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))
        im = ax[0].imshow(spread[:irmax].clip(1e-3), origin='lower', aspect='auto',
                          extent=[-ZMAX, ZMAX, 0, r_report], cmap='magma', norm=LogNorm(1e-2, spread.max()))
        ax[0].set_title(f'phi-spread sigma_phi(rho)/<rho>  (r < {r_report:g} cm)')
        ax[0].set_xlabel('z [cm]'); ax[0].set_ylabel('r [cm]'); fig.colorbar(im, ax=ax[0])
        for rl in layers_barrel_r: ax[0].axhline(rl, color='cyan', lw=0.7, ls=(0,(4,3)), alpha=.6)
        ax[1].plot(rgrid[:irmax], prof_spread[:irmax], label='sigma_phi(rho)/<rho>  (spread)')
        ax[1].plot(rgrid[:irmax], prof_occ[:irmax],    label='phi-occupancy  (coverage)')
        for rl in layers_barrel_r: ax[1].axvline(rl, color='0.6', lw=0.7, ls=(0,(4,3)))
        ax[1].set_xlabel('r [cm]'); ax[1].set_title(f'barrel |z|<{z_barrel:g} cm profile vs r')
        ax[1].set_xlim(0, r_report); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.savefig(out + '_phispread.png', dpi=100)
        print(f"  [phi-scan] wrote {out}_phispread.png")
    except Exception as e:
        print("  [phi-scan] plot skipped:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True, nargs='+',
                    help='One or more material trees, SUMMED (e.g. the Tracker tree AND the '
                         'BeamPipe tree, so the beam pipe / inner services enter the inner term).')
    ap.add_argument('--cfi', default='../../../HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2PixelTracksSoAWithStubs_cfi.py')
    ap.add_argument('--layer-geom', default=None,
                    help='CSV "layer,isBarrel,r,z". If omitted, the layer geometry is auto-derived '
                         'from the silicon peaks in the tree (count-checked against the CA config).')
    ap.add_argument('--auto', action='store_true',
                    help='Force the count-driven auto layer-geometry (same as omitting --layer-geom).')
    ap.add_argument('--out', default='caLayerPairMaterial')
    ap.add_argument('--map-dr', type=float, default=2.0, help='exported rho-map r bin [cm]')
    ap.add_argument('--map-dz', type=float, default=4.0, help='exported rho-map z bin [cm]')
    ap.add_argument('--tag', default='D121', help='geometry tag appended to the exported accessor/array names')
    ap.add_argument('--phi-scan', action='store_true',
                    help='Diagnostic only: rebuild rho(r,phi,z) and report the azimuthal spread '
                         'sigma_phi(rho)/<rho> + phi-occupancy per layer (quantifies the phi-symmetry assumption). '
                         'Does NOT emit/modify the production map.')
    ap.add_argument('--nphi', type=int, default=48, help='phi bins for --phi-scan')
    a = ap.parse_args()
    if a.auto:
        a.layer_geom = None   # --auto explicitly selects the auto-derivation path

    if a.phi_scan:
        # Layer radii for the per-layer report: from the CSV if given, else the auto-detected silicon peaks.
        if a.layer_geom:
            geom = load_layer_geom(a.layer_geom)
        else:
            rho = build_density(a.tree)
            det = rho['_SI'] if rho['_SI'].max() > 0 else rho['']
            geom = auto_layer_geom(det)
        barrel_r = sorted({round(r, 2) for (b, r, z) in geom.values() if b and r > 0 and r < 16.0})
        print(f"  [phi-scan] inner barrel layer radii: {barrel_r} cm")
        num, den = build_phi_density(a.tree, a.nphi)
        phi_scan_report(num, den, a.out, barrel_r)
        return

    isB, pairs = parse_cfi(a.cfi)
    print(f"parsed {len(isB)} layers, {len(pairs)} layer-pairs from {a.cfi}")
    rho = build_density(a.tree)

    # diagnostic maps up-front (so they exist even if the layer count-check below aborts)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))   # r-z view: z horizontal, r vertical
        for axi, key, ttl in [(ax[0], '', 'total rho(r,z)'), (ax[1], '_SI', 'silicon X/X0 (layer locator)')]:
            im = axi.imshow(np.log10(rho[key] + 1e-6), origin='lower', aspect='auto',
                            extent=[-ZMAX, ZMAX, 0, RMAX], cmap='viridis')
            axi.set_title(ttl); axi.set_xlabel('z [cm]'); axi.set_ylabel('r [cm]'); fig.colorbar(im, ax=axi)
        plt.tight_layout(); plt.savefig(a.out + '_rho.png', dpi=90)
        print(f"  wrote {a.out}_rho.png (total + silicon maps)")
    except Exception as e:
        print("  rho plot skipped:", e)

    if a.layer_geom:
        geom = load_layer_geom(a.layer_geom)
    else:
        if rho['_SI'].max() > 0:
            det = rho['_SI']; print("  [auto] detecting layers from the SILICON (Material X0) map")
        else:
            det = rho['']; print("  [auto] WARNING: no silicon found -> using TOTAL density (noisier)")
        geom = auto_layer_geom(det)           # count-driven: bucket peaks -> CA indices
        with open(a.out + '_layers.csv', 'w') as fg:   # dump for the record / manual override
            fg.write('layer,isBarrel,r,z\n')
            for idx in sorted(geom):
                b, r, z = geom[idx]; fg.write(f"{idx},{int(b)},{r:.2f},{z:.2f}\n")
        print(f"  [auto] wrote detected geometry to {a.out}_layers.csv")

    # Inner/upstream material per layer: beamline (virtual IP at r=0,z=0) -> each layer.
    # The track scatters here BEFORE its first measurement, so this MS sets the d0/z0 pulls and
    # must be attached to the FIRST hit on the track (e.g. innermost IT barrel layer = beam pipe
    # + everything inside it). Computed with the same path logic (radial for barrel, chord for disk).
    geom[-1] = (True, 0.0, 0.0)   # virtual IP on the beamline
    inner = {}
    for L in sorted(k for k in geom if k >= 0):
        inner[L] = pair_material(-1, L, geom, isB, rho)
    print(f"  inner material (beamline->layer), innermost IT barrel L0 = {inner[0]['']:.5f} X/X0 "
          f"(includes beam pipe)")

    rows = []
    for (i, o) in pairs:
        if i not in geom or o not in geom:
            print(f"  WARN pair ({i},{o}): missing layer geometry, skipped"); continue
        m = pair_material(i, o, geom, isB, rho)
        rows.append((i, o, m))
    # CSV: per-pair (between layers)
    with open(a.out+'.csv','w') as fcsv:
        fcsv.write('i,o,mode,XX0'+','.join('XX0'+c for c in CATS[1:])+'\n')
        for i,o,m in rows:
            fcsv.write(f"{i},{o},{m['mode']},{m['']:.5f},"+','.join(f"{m[c]:.5f}" for c in CATS[1:])+'\n')
    # CSV: per-layer inner/upstream (beamline -> layer)
    with open(a.out+'_inner.csv','w') as fci:
        fci.write('layer,mode,XX0'+','.join('XX0'+c for c in CATS[1:])+'\n')
        for L in sorted(inner):
            m = inner[L]
            fci.write(f"{L},{m['mode']},{m['']:.5f},"+','.join(f"{m[c]:.5f}" for c in CATS[1:])+'\n')
    # C++ header: per-pair + per-layer inner arrays
    with open(a.out+'.h','w') as fh:
        fh.write("// auto-generated by extractCALayerPairMaterial.py -- normal-incidence X/X0.\n")
        fh.write("// Feed to multScatt, scaling by the track angle (1/sin barrel, 1/cos disk).\n\n")
        fh.write("// (a) material BETWEEN layers, indexed as the CA layerPairs table:\n")
        fh.write("constexpr float kLayerPairXX0[] = {\n")
        for i,o,m in rows:
            fh.write(f"  {m['']:.5f}f,  // ({i:2d},{o:2d}) {m['mode']}\n")
        fh.write("};\n\n")
        fh.write("// (b) inner/upstream material (beam pipe + services), beamline->layer, indexed by CA\n")
        fh.write("//     layer; attach to the FIRST hit on the track (sets d0/z0 MS):\n")
        fh.write("constexpr float kLayerInnerXX0[] = {\n")
        for L in sorted(inner):
            fh.write(f"  {inner[L]['']:.5f}f,  // layer {L:2d} ({inner[L]['mode']})\n")
        fh.write("};\n")
    print(f"wrote {a.out}.csv and {a.out}.h  ({len(rows)} pairs)")
    export_rho_map(rho, a.out, a.map_dr, a.map_dz, a.tag)   # device-ready rho(r,z) for the runtime integral

if __name__ == '__main__':
    main()
