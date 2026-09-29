"""Triplet training dataset: one row per triplet the cellular automaton builds, as nanoAOD tables.

Needs the dump build (uncomment '#define CA_TRIPLET_DUMP' in CATripletDumpMacro.h and rebuild):
the CA producer then emits a per-event device 'Triplet' product. This config adds its tables to
an existing reconstruction process (the base config; see _import_chassis) and prunes validation/DQM:
  * TripletFeaturesTableProducer -> 'Trk<X>Triplet' (one row per built triplet, with h1/h2/h3 keys)
  * HitTruthTableProducer        -> 'HitTruth' (each merged hit -> its truth particle)
A triplet counts as real when its three hits point at the same truth particle (joined at training
time in train_triplet_dnn.py). The tables carry the event structure, so this path has no ordering
constraint and may run multi-threaded.

  DS_SAMPLE, DS_NEVT, DS_SKIP, DS_NTUPLE_ITER (prompt|disp), DS_NTUPLE_THREADS, DS_NTUPLE_OUT,
  DS_FILES, DS_CHASSIS, DS_BACKEND (cpu|gpu): see the env vars below.

Produced by:  retrain_prompt.sh dump --for triplet;  retrain_displaced.sh dump --for triplet
"""
import glob
import os

import FWCore.ParameterSet.Config as cms

_sample = os.environ.get("DS_SAMPLE", "ttbarPU")


def _import_chassis(candidates):
    """Import the base reconstruction config this dump adds its tables to.

    This config builds no reconstruction of its own. DS_CHASSIS names the module to use;
    otherwise the first importable name in `candidates` is taken. Generate one with

        ./make_base_config.sh --files <input.root,...> --out <dir>/base_reco_cfg.py

    and either pass DS_CHASSIS=base_reco_cfg with <dir> on PYTHONPATH, or let the retraining
    entry scripts do it. A tracking-only menu is the natural base: it runs the whole pixel and
    outer-tracker chain the dataset comes from and skips everything else, and the truth inputs
    it needs (truth particles and simulation links from the input, merged hits and stubs from
    the reconstruction) are all present in it."""
    import importlib
    forced = os.environ.get("DS_CHASSIS")
    names = [forced] if forced else list(candidates)
    tried = []
    for _n in names:
        try:
            _mod = importlib.import_module(_n)
        except ImportError as _e:
            tried.append("%s (%s)" % (_n, _e))
            continue
        print("tripletDump: base config = %s" % _n)
        return _mod.process
    raise ImportError(
        "triplet_dump_cfg: no base reconstruction config found. Tried: " + "; ".join(tried)
        + ".\n  Generate one, from the events you want to reconstruct, with:\n"
        "    RecoTracker/PixelSeeding/test/models/make_base_config.sh \\\n"
        "        --files <a.root,b.root|dir|list.txt> --out <dir>/base_reco_cfg.py\n"
        "  then run this config with DS_CHASSIS=base_reco_cfg and <dir> on PYTHONPATH.\n"
        "  The retraining entry scripts (retrain_prompt.sh / retrain_displaced.sh) do all of\n"
        "  that for you when they are given --input <event files>.")


# Only the generated chassis is a candidate; name a different module explicitly with DS_CHASSIS.
process = _import_chassis(["base_reco_cfg"])
if _sample != "ttbarPU":
    # Built-in input list for the displaced pile-up sample; DS_FILES replaces it.
    if _sample == "displacedPU" and not os.environ.get("DS_FILES"):
        _dir = ("/eos/cms/store/relval/CMSSW_16_1_0_pre4/RelValDisplacedSUSY_14TeV/GEN-SIM-DIGI-RAW/"
                "PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_20260505_140840-v1/2590000/")
        _files = sorted(glob.glob(_dir + "*.root"))
        if _files:
            process.source.fileNames = cms.untracked.vstring(["file:" + f for f in _files])
        else:
            print("tripletDump: the built-in displacedPU input list is not available here; "
                  "keeping the base config's own input files. Pass DS_FILES to choose them.")

_dsFiles = os.environ.get("DS_FILES")
if _dsFiles:
    process.source.fileNames = cms.untracked.vstring(
        [(f if f.startswith("file:") or f.startswith("/store") else "file:" + f)
         for f in _dsFiles.split(",") if f])

process.maxEvents.input = cms.untracked.int32(int(os.environ.get("DS_NEVT", "1000")))
process.source.skipEvents = cms.untracked.uint32(int(os.environ.get("DS_SKIP", "0")))  # parallel chunks

# CA backend: CPU (alpaka serial_sync) by default; DS_BACKEND=gpu to override.
if os.environ.get("DS_BACKEND", "cpu") == "cpu":
    process.options.accelerators = cms.untracked.vstring("cpu")
else:
    process.options.accelerators = cms.untracked.vstring("*")

# Prune validation and DQM paths: the dump needs only the HLT reconstruction. The base config's own
# output EndPath and output modules go too, since this dump writes its own NanoAOD file below and a
# chassis made without cmsDriver's '--output={}' would write a full GEN-SIM-DIGI-RAW file per dump.
_keep = []
for _p in process.schedule:
    _lbl = _p.label_()
    if any(k in _lbl for k in ("alidation", "DQM", "dqm")) or _lbl.endswith("output_step"):
        continue
    _keep.append(_p)
process.schedule = cms.Schedule(*_keep)
for _om in list(process.outputModules_()):
    delattr(process, _om)

# NanoAOD table pipeline, the per-built-triplet training dataset. Requires the CA_TRIPLET_DUMP build:
# the CA producer emits a per-event device 'Triplet' product with one row per built triplet.

# The displaced CA module label differs between menus (SoADisplaced / SoALowPt); resolve it once,
# falling through to the first name so the hasattr() guard below is the single missing-module check.
def _resolve_label(*names):
    for _n in names:
        if hasattr(process, _n):
            return _n
    return names[0]


_dispLabel = _resolve_label("hltPhase2PixelTracksSoADisplaced", "hltPhase2PixelTracksSoALowPt")

_iter = os.environ.get("DS_NTUPLE_ITER", "disp")
if _iter == "prompt":
    _caSrc, _tabName = "hltPhase2PixelTracksSoA", "TrkPromptTriplet"
else:
    _caSrc, _tabName = _dispLabel, "TrkDispTriplet"

# Only the iteration being tabled has its triplet reject compiled out and its triplets written
# (CATripletCuts.h: dumpIteration >= 0); every other CA module keeps the production reject, so the
# chain upstream of the dumped iteration is the production one. Set on both modules, since the name
# matches only the dumped iteration.
_dumpIterName = "promptHighPt" if _iter == "prompt" else "displaced"
for _l in ("hltPhase2PixelTracksSoA", _dispLabel):
    if hasattr(process, _l):
        getattr(process, _l).tripletDumpIteration = cms.string(_dumpIterName)
print("tripletDump: tripletDumpIteration=%s on the CA modules (the other iteration keeps the production reject)"
      % _dumpIterName)


def _scale_str_cap(mod, name, factor):
    """Scale a STRING cap by `factor`. These caps are math expressions in x (the hit count),
    evaluated per event by the CA sizing code, so we wrap the current value as 'factor*(expr)'.
    That scales ANY form correctly -- a plain number ('8388608'), a 'c*x' formula ('18.7492*x'),
    or a polynomial ('0.00022*pow(x,2)+0.53*x+10000'). No-op if the param is absent."""
    if not hasattr(mod, name):
        return
    old = getattr(mod, name).value()  # cms.string -> python str
    new = "%g*(%s)" % (factor, old)
    setattr(mod, name, cms.string(new))
    print("  %s: '%s' -> '%s' (x%g)" % (name, old, new, factor))


def _scale_dbl_cap(mod, name, factor):
    """Scale a DOUBLE cap (avgHitsPerTrack / avgCellsPerCell / ...) by `factor`. No-op if absent."""
    if not hasattr(mod, name):
        return
    old = float(getattr(mod, name).value())
    new = old * factor
    setattr(mod, name, cms.double(new))
    print("  %s: %g -> %g (x%g)" % (name, old, new, factor))

# The gates of the iteration being dumped are switched off, so the product holds every triplet the
# cuts accepted: with useTripletDNN on, the deployed model would filter the training set down to the
# triplets it accepts. useTrackDNN acts downstream of triplet building and is switched off only to
# save the work.
if hasattr(process, _caSrc):
    getattr(process, _caSrc).useTripletDNN = cms.bool(False)
    getattr(process, _caSrc).useTrackDNN = cms.bool(False)
    print("tripletDump: %s gates off -- dumping every triplet the cuts accepted" % _caSrc)
    # Without its gate the automaton builds three to four times as many triplets and tracks as
    # production does, overflowing buffers sized for production. The caps are scaled up from whatever
    # the arm carries, rather than assigned, so that they cannot silently shrink; fillStatistics makes
    # an overflow visible in the log.
    _mod = getattr(process, _caSrc)
    # Escape hatch: keep the production caps untouched, to see whether they would overflow.
    if os.environ.get("DS_DUMP_NO_CAP_OVERRIDE", "0") == "1":
        _mod.fillStatistics = cms.bool(True)
        print("tripletDump: DS_DUMP_NO_CAP_OVERRIDE=1 -> keeping the production caps of %s, "
              "with the occupancy summary on" % _caSrc)
    else:
        print("tripletDump[%s]: enlarging this arm's own buffers for the ungated dump:" % _iter)
        _scale_str_cap(_mod, "maxNumberOfDoublets", 1.5)
        _scale_str_cap(_mod, "maxNumberOfTuples", 16.0)
        _scale_dbl_cap(_mod, "avgHitsPerTrack", 2.0)
        _scale_dbl_cap(_mod, "avgCellsPerCell", 4.0)
        _scale_dbl_cap(_mod, "avgTracksPerCell", 8.0)
        _mod.fillStatistics = cms.bool(True)
        print("  fillStatistics = True")

    # Prompt arm: hold the distance-of-closest-approach cut on connections between two strip-strip
    # disk layers (both endpoints in the odd layer identifiers 35..53) at 5. A value tuned for purity
    # removes those connections outright, and the model has to see them to learn to reject them.
    if _iter == "prompt":
        _S2 = {35, 37, 39, 41, 43, 45, 47, 49, 51, 53}
        _g = list(_mod.geometry.pairGraph.value())
        _dca = list(_mod.geometry.caDCACutsPerPair.value())
        _n2s = 0
        for _k in range(len(_dca)):
            if _g[2 * _k] in _S2 and _g[2 * _k + 1] in _S2:
                _dca[_k] = 5.0
                _n2s += 1
        _mod.geometry.caDCACutsPerPair = cms.vdouble(_dca)
        print("tripletDump[prompt]: %d strip-strip layer pairs held at caDCACutsPerPair=5 "
              "so the training set contains the connections a tighter cut would remove" % _n2s)
        assert _n2s == 8, "expected 8 2S->2S chain pairs, got %d" % _n2s

process.tripletFeaturesTable = cms.EDProducer(
    "TripletFeaturesTableProducer",
    tableName=cms.string(_tabName),
    tripletDumpSrc=cms.InputTag(_caSrc),  # device 'Triplet' product (empty instance label)
)
process.hitTruthTable = cms.EDProducer(
    "HitTruthTableProducer",
    tableName=cms.string("HitTruth"),
    TP_signalOnly=cms.bool(False),  # labeling: ANY selected TP (incl. in/out-of-time PU) is real
    # |eta| coverage must reach the TEPX edge, or every 4.0-4.4 triplet is labelled fake; 4.5 matches
    # the track-level truth producers.
    TP_maxEta=cms.double(4.5),
)
process.tripletNanoPath = cms.Path(process.tripletFeaturesTable + process.hitTruthTable)
process.schedule.append(process.tripletNanoPath)

_nout = os.environ.get("DS_NTUPLE_OUT", "tripletNano_%s.root" % _sample)
process.tripletNanoOut = cms.OutputModule(
    "NanoAODOutputModule",
    fileName=cms.untracked.string(_nout),
    outputCommands=cms.untracked.vstring("drop *", "keep nanoaodFlatTable_*_*_*"),
)
process.tripletNanoEnd = cms.EndPath(process.tripletNanoOut)
process.schedule.append(process.tripletNanoEnd)

# The nano tables carry the event structure -> no ordering constraint; allow many threads/streams.
_nthr = int(os.environ.get("DS_NTUPLE_THREADS", "1"))
process.options.numberOfThreads = _nthr
process.options.numberOfStreams = _nthr

print("tripletDump: sample=%s nevt=%d iter=%s tab=%s out=%s threads=%d" % (
    _sample, process.maxEvents.input.value(), _iter, _tabName, _nout, _nthr))
