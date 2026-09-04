# Regenerating the BL-fit material map

The compiled-in table `src/BLMaterialMap<geometry>.cc` is the material model the BrokenLine and GBL fits use
(accessor `blMaterialMap::blMaterialMapData()`). Its header records the geometry, era and release it was made
with and how. This directory holds the code that makes it, in three steps driven by one script:

1. `blMaterialMapRays_cfg.py` -- cmsRun: straight, non-interacting rays through the Geant4 detector of the
   chosen geometry, every step inside the Tracker and beam-pipe volumes written by the release's
   MaterialBudgetAction watcher (one job per seed);
2. `blMaterialMapBuild` -- test binary (built with the package): splits every step exactly into the (r,z)
   cells it crosses and accumulates the radius-weighted material per cell;
3. `blMaterialMapEmit.py` -- sums the per-job accumulators and writes `BLMaterialMap<tag>.cc` with a
   provenance header taken from the run, or checks an existing table against them (`--check`).

    cmsenv                                        # an area in which RecoTracker/PixelTrackFitting is built
    ./blMaterialMapRun.sh /path/to/work            # the shipped recipe: D121, Phase2C22I13M9, 40 x 150000 rays
    ./blMaterialMapRun.sh /path/to/work 40 150000 D127 Configuration.Geometry.GeometryExtendedRun4D127Reco_cff Phase2C22I13M9

The script writes `/path/to/work/BLMaterialMap<tag>.cc`, and when `src/BLMaterialMap<tag>.cc` exists it also
prints how many of the 140000 values differ from it (0 when run in the release that made the shipped table:
Geant4 steps depend on the Geant4 build, so another release can move a value at its last printed digit). To
ship a table: copy it to `src/`, remove the previous table file (exactly one is compiled in) and rebuild.
Nothing else in the code changes with the geometry.

`test/blMapsCheck_cfg.py` (BLMapsCheck) then verifies in a cmsRun job that the EventSetup product equals the
compiled-in table and that the field map equals the MagneticField resampled on its lattice.
