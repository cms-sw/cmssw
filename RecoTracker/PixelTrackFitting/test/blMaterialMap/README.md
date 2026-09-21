# Regenerating the BL-fit material maps

The BrokenLine and GBL fits read one serialized `blMaterialMap::Map` per tracker geometry, served at run
time from `RecoTracker/PixelSeeding/data/BLMaterialMap/`: `BLMaterialMap_<T>_BP<bp>_v<n>.bin` for tracker
version `<T>`, beam-pipe version `<bp>` and map version `<n>`, catalogued in `BLMaterialMap.index`. The
ESProducer fingerprints the geometry it runs on and loads the matching file. The byte format is documented
in `interface/BLMaterialMapFile.h`. This directory holds the code that makes the maps, three steps driven
by one script:

1. `blMaterialMapRays_cfg.py` -- cmsRun: straight, non-interacting rays through the Geant4 detector of the
   chosen geometry; every step inside the Tracker and beam-pipe volumes is written by the release's
   MaterialBudgetAction watcher (one job per seed). The `BLMaterialMapFingerprintDump` analyzer runs in
   the same job and writes the geometry fingerprint (job 1 also dumps the sensor positions).
2. `blMaterialMapBuild` -- test binary: splits every step exactly into the (r,z) cells it crosses and
   accumulates the radius-weighted material per cell.
3. `blMaterialMapEmit.py` -- sums the per-job accumulators and writes the `.bin` (provenance text from the
   run, sensor reference positions appended), maintains the index line (`--index`), or checks an
   existing `.bin` or `.cc` against the accumulated values (`--check`).

    cmsenv                                        # an area in which RecoTracker/PixelTrackFitting is built
    ./blMaterialMapRun.sh /path/to/work            # the shipped recipe: T35 (D121), Phase2C22I13M9, 40 x 150000 rays

    ./blMaterialMapRun.sh <outdir> [njobs] [rays-per-job] [tag] [geometry-cff] [era]
      njobs         parallel single-core Geant4 jobs, seeds 1..njobs (shipped maps: 40)
      rays-per-job  straight 10 GeV neutrino rays per job, eta flat in [-6,6], phi flat, from z=0 (150000)
      tag           the tracker version the map is for (T35)
      geometry-cff  the geometry the rays cross (Configuration.Geometry.GeometryExtendedRun4D121Reco_cff)
      era           the era to load it with (Phase2C22I13M9); the beam-pipe and materials versions come
                    from its XML list and land in the provenance and in the beam-pipe tag

The script writes `<outdir>/BLMaterialMap_<tag>_BP<bp>_v<n>.bin` (`<n>` = `$MAPVERSION`, default 1) and the
matching index line in `<outdir>/BLMaterialMap.index`. Geometry, era, release, beam pipe, materials, ray
sample, fingerprint and sensor census are recorded in `<outdir>/PROVENANCE.txt` and embedded in the
file. `blMaterialMapEmit.py --bins <outdir>/bins --check <map>.bin` compares a regeneration with an
existing map value by value; a run in another release can move a value at its last printed digit
(Geant4 steps depend on the Geant4 build).

## The geometry fingerprint

The index key is an FNV-1a hash over the sorted rawIds of every tracker sensor of the ideal
`GeometricDet` tree (`interface/BLMaterialMapFingerprint.h`); no alignment and no positions enter it,
so DDD, DD4hep and DB sourcing of the same geometry give the same key. Tracker versions that share the
sensor inventory (T36/T37/T38, whose modules differ in placement by 0.6-1.4 mm) share the key: each
file therefore also carries its sensors' reference positions in 0.1 mm buckets, and the producer keeps
the candidate whose positions match the job's within one bucket. Sourcing modes move positions by a
few nm, which can cross a bucket boundary by one bucket at most. To get the fingerprint of a geometry
without generating rays:

    ./blMaterialMapRun.sh --fingerprint-only /path/to/work [tag] [geometry-cff] [era]

which runs one event, prints `FINGERPRINT 0x<16 hex>` and writes `PROVENANCE.txt` and
`trees/fingerprint_001.txt` (fingerprint, census, sensor positions) as a full run's first job does. A
full run refuses to build the map if its jobs disagree on the fingerprint.

## Adding a geometry `<Tnn>`

1. `./blMaterialMapRun.sh /path/to/work 40 150000 Tnn <geometry-cff> <era>` in the release.
2. `cp /path/to/work/BLMaterialMap_Tnn_BP<bp>_v1.bin RecoTracker/PixelSeeding/data/BLMaterialMap/`
3. Merge the line the run wrote in `/path/to/work/BLMaterialMap.index` into
   `RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index`
   (`0x<16 hex> <tag> <beam-pipe> <version> <file>`, one space separated, `#` comments allowed, sorted
   by fingerprint then file).

No code change: the ESProducer finds the index through FileInPath
(`RecoTracker/PixelSeeding/data/BLMaterialMap/BLMaterialMap.index`).

## Beam-pipe or materials changes

The fingerprint sees tracker sensors only, so a beam-pipe or materials change under an unchanged
tracker does not change the key. Regenerate with `MAPVERSION=2` (giving `BLMaterialMap_<T>_BP<newbp>_v2.bin`)
and replace the tracker's index line: two files of the same tracker would both match the job and the
producer throws on the ambiguity.

## Byte exactness

Every value is written as `np.float32(float("%.6g" % v))`, zeros exactly 0.0; a six-significant-digit
decimal is never within a float64 ulp of a float32 rounding boundary, so these are the floats the
compiler made of the retired table's `"%.6gf"` literals. The shipped T35 map is that table converted with
no value change (the table is no longer in the tree; the historical command was
`blMaterialMapEmit.py --from-cc src/BLMaterialMapD121.cc --tag T35 --beam-pipe 2030/v3 --fingerprint ...
--positions ... --provenance ... --out BLMaterialMap_T35_BP2030v3_v1.bin`). `--check` compares the whole
map, both lattices, and prints `0 differences` when the contents agree.
