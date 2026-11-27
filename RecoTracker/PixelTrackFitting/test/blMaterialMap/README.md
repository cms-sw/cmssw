# Regenerating the BrokenLine-fit material maps

The BrokenLine fit reads one `blMaterialMap::Map` per tracker geometry from
`RecoTracker/PixelSeeding/data/BLMaterialMap/`. The file `BLMaterialMap_<T>_BP<bp>_v<n>.bin` is for tracker version
`<T>`, beam-pipe version `<bp>` and map version `<n>`. `BLMaterialMap.index` lists the files. The ESProducer computes
the fingerprint of the running geometry and loads the matching file. `interface/BLMaterialMapFile.h` gives the byte
format.

One script runs three steps:

1. `blMaterialMapRays_cfg.py` (cmsRun): sends straight rays through the Geant4 detector, one job per seed.
   MaterialBudgetAction writes every step in the tracker and the beam pipe. `BLMaterialTableDump` writes the
   material table. `BLMaterialMapFingerprintDump` writes the geometry fingerprint. Job 1 also writes the sensor
   positions.
2. `blMaterialMapBuild` (test binary): splits each step into the (r,z) cells it crosses and adds the material per
   cell. A step whose material is not in the table stops the job.
3. `blMaterialMapEmit.py`: adds the job outputs and writes the `.bin` and its index line (`--index`), or compares an
   existing `.bin` with the sums (`--check`).

```
cmsenv                                   # an area where RecoTracker/PixelTrackFitting is built
./blMaterialMapRun.sh /path/to/work      # the shipped recipe: T35 (D121), Phase2C22I13M9, 40 x 150000 rays

./blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era] [map-version]
  njobs         parallel single-core Geant4 jobs, seeds 1..njobs (40)
  rays-per-job  10 GeV neutrino rays per job, eta flat in [-6,6], phi flat, from the origin (150000)
  tag           the tracker version of the map (T35)
  geometry-cff  the geometry that the rays cross (Configuration.Geometry.GeometryExtendedRun4D121Reco_cff)
  era           the era of the geometry (Phase2C22I13M9); it sets the beam-pipe and materials versions
  map-version   the map version for this geometry and beam pipe (1)
```

The script writes `<outdir>/BLMaterialMap_<tag>_BP<bp>_v<n>.bin`, its line in `<outdir>/BLMaterialMap.index`, and
`<outdir>/PROVENANCE.txt`. The `.bin` also holds the provenance text.

To compare a new map with an existing one, value by value:

```
blMaterialMapEmit.py --bins <outdir>/bins --check <map>.bin
```

It prints `0 differences` when the maps agree. Another release can change a value at its last digit.

## What the map holds

Each ray is a 10 GeV muon neutrino from the origin, in a field-free detector with no interactions. Each step of a
ray is in one material, and the tree gives `dmb` = dl/X0 for the step.

A cell is a ring of 0.5 cm in r over [0, 125) cm and 1 cm in z over [-280, 280) cm, averaged over phi. The builder
cuts each step at the cell edges: a quadratic equation in the step parameter for r, a linear one for z. Each part of
length dl gets the weight w = dl rbar, with rbar its Simpson mean radius. The weight cancels the 1/r density of rays
flat in eta, because d(eta) ds = dr dz / r. Per cell:

```
den  += w
num  += (dmb / dl) w
numE += rhoE w,   numEI += rhoE ln(I/eV) w,   numEE += rhoE ln(rhoE) w
```

The emitter stores, per cell, 0 where the denominator is 0:

```
rho    = num / den       the area average of 1/X0 [1/cm]
rhoE   = numE / den      the area average of the electron density over N_A [mol/cm^3]
lnI    = numEI / numE    the rhoE-weighted mean of ln(I/eV)
lnRhoE = numEE / numE    the rhoE-weighted mean of ln(rhoE)
```

`elossLandau` in `interface/BLMaterialMap.h` uses the dE/dx values (PDG RPP 2026, Sec. 34.2).

The builder finds the material of a step by its (density, X0) pair in the table of `BLMaterialTableDump`. It takes
the nearest pair in relative distance. The job fails if that pair is farther than 10^-3.

## The geometry fingerprint

The index key is an FNV-1a hash over the sorted rawIds of the tracker sensors of the ideal `GeometricDet` tree
(`interface/BLMaterialMapFingerprint.h`). DDD, DD4hep and DB give the same key for one geometry. Tracker versions
with the same sensors share a key, so each file also holds the sensor positions in 0.1 mm buckets. The producer keeps
the file whose positions match the job within one bucket.

To get the fingerprint of a geometry without rays:

```
./blMaterialMapRun.sh --fingerprint-only /path/to/work [tag] [geometry-cff] [era]
```

This prints `FINGERPRINT 0x<16 hex>` and writes `PROVENANCE.txt` and `trees/fingerprint_001.txt`. A full run stops
if its jobs do not agree on the fingerprint.

## Adding a geometry `<Tnn>`

1. Run `./blMaterialMapRun.sh /path/to/work 40 150000 Tnn <geometry-cff> <era>`.
2. Run `cp /path/to/work/BLMaterialMap_Tnn_BP<bp>_v1.bin RecoTracker/PixelSeeding/data/BLMaterialMap/`.
3. Add the line from `/path/to/work/BLMaterialMap.index` to
   `RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index`. The format is
   `0x<16 hex> <tag> <beam-pipe> <version> <file>`, one space between fields, `#` comments allowed. Sort the lines by
   fingerprint, then by file.

No code change is necessary. The `indexFile` parameter of the ESProducer names the index.

## Beam-pipe or materials changes

The fingerprint does not see the beam pipe or the materials. After such a change, regenerate with map-version 2,
which gives `BLMaterialMap_<T>_BP<newbp>_v2.bin`. Replace the line of the tracker in the index: if two files match
the job, the producer throws an exception.

## Value format

The emitter writes each value as `np.float32(float("%.6g" % v))`, and zeros as exactly 0.0.
