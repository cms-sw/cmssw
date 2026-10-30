"""Track-level training dataset: one row per reconstructed track, with the features the
selectors see and the truth the track validation uses.

It adds tables to an existing reconstruction process (the base config, see _import_chassis)
and writes them as nanoAOD. Two row spaces per iteration, joined offline by an
order-preserving (pt, phi) match:
  Trk<X>Full  : every track in the SoA, before selection      -> the fitted features
  Trk<X>CA    : the same tracks' hit and vector hit features
  Trk<X>Truth : the legacy conversion at quality minQuality -> matched/duplicate/truth kinematics

It feeds the track DNN (train_disp_nano.py) and the final high-purity selector (nano_loader.py +
train_merged_forest.py; the model-family comparison also uses build_tree_model.py and
train_prompt_hp_nano.py). Multithreaded.

It is steered by command-line options (FWCore.ParameterSet.VarParsing), e.g.

  cmsRun trackNano_config.py chassis=base_reco_cfg inputFiles=a.root,b.root maxEvents=100 \
      minQuality=loose mtvLabel=1 threads=8

'cmsRun trackNano_config.py help' lists them with their defaults. The main ones:

  sample      = label naming the output file (default ttbarPU)
  inputFiles  = comma-separated input files (default: those of the base config)
  maxEvents   = events (default 1000);  skipEvents = events to skip first (default 0)
  outputFile  = output file (default trackNano_<sample>.root)
  minQuality  = quality of the truth conversion: 'loose' (the population the track DNN decides
                on) | 'tight' (default; the population the final selector sees)
  threads     = threads/streams (default 1)
  mtvLabel    = 1 adds the columns matchedAny/duplicateAny to the truth tables: the match to ANY
                TrackingParticle, which is what the track validation calls "not a fake" and what
                the trainers target by default
  chassis     = base reconstruction config module (default base_reco_cfg, see _import_chassis;
                make_base_config.sh generates one)

Chain-state overrides, one set per iteration (default = inherit the producer configuration;
see the block below): promptGate / promptGateOff / promptTrackDNN / promptTrackThr / promptCaps
(+ promptCapsFactor) and the disp* equivalents. Every dump prints a '[trackNano CHAIN-STATE]' line
per iteration with the effective state; a dataset is only valid for the state on that line.

The retraining entry script sets the right state for each step:
  retrain_prompt.sh dump --for gate|hp

Iterations covered: prompt (hltPhase2PixelTracksSoA) and, when the base config runs one,
displaced (hltPhase2PixelTracksSoADisplaced, formerly hltPhase2PixelTracksSoALowPt -- either
label resolves, see _resolve_label). Without a displaced iteration only the TrkPrompt* tables are
written. The displaced truth uses a displaced-friendly truth-particle selection (transverse
impact parameter below 60 cm, no in-time requirement).
"""
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing()
_one, _list = VarParsing.multiplicity.singleton, VarParsing.multiplicity.list
_str, _int, _float, _bool = (VarParsing.varType.string, VarParsing.varType.int, VarParsing.varType.float,
                             VarParsing.varType.bool)
options.register("sample", "ttbarPU", _one, _str,
                 "label naming the output file (trackNano_<sample>.root)")
options.register("chassis", "", _one, _str,
                 "base reconstruction config module to add the tables to (default: base_reco_cfg)")
options.register("inputFiles", "", _list, _str,
                 "comma-separated input files (default: the base config's own input files)")
options.register("maxEvents", 1000, _one, _int, "number of events to process")
options.register("skipEvents", 0, _one, _int, "number of input events to skip first")
options.register("outputFile", "", _one, _str,
                 "output NanoAOD file (default: trackNano_<sample>.root)")
options.register("threads", 1, _one, _int, "number of threads and streams")
options.register("minQuality", "tight", _one, _str,
                 "quality of the truth conversion: loose (track-DNN dataset) | tight (selector dataset)")
options.register("mtvLabel", False, _one, _bool,
                 "add matchedAny/duplicateAny (match to any TrackingParticle) to the truth tables")
options.register("dispLoosen", "", _one, _str,
                 "factor scaling the displaced iteration's z0/dz doublet cuts (default: none)")
options.register("doStats", False, _one, _bool, "turn on the CA buffer occupancy summary")
for _arm, _what in (("prompt", "prompt"), ("disp", "displaced")):
    options.register(_arm + "Caps", "", _one, _str,
                     "'big' enlarges the %s CA's tuple and cell buffers (default: as configured)" % _what)
    options.register(_arm + "CapsFactor", 4., _one, _float,
                     "enlargement factor of %sCaps=big" % _arm)
    options.register(_arm + "Gate", "", _one, _str,
                     "tripletDNNThreshold of the %s CA (default: as configured)" % _what)
    options.register(_arm + "GateOff", False, _one, _bool,
                     "switch the %s CA's triplet gate off" % _what)
    options.register(_arm + "TrackDNN", "", _one, _str,
                     "1 / 0 switches the %s CA's track DNN on / off (default: as configured)" % _what)
    options.register(_arm + "TrackThr", "", _one, _str,
                     "trackDNNThreshold of the %s CA (default: as configured)" % _what)
options.parseArguments()

# The sample is a label: it names the output. inputFiles chooses the events.
_sample = options.sample


def _import_chassis(candidates):
    """Import the base reconstruction config this dump adds its tables to.

    This config builds no reconstruction of its own. The chassis option names the module to use;
    otherwise the first importable name in `candidates` is taken. Generate one with

        ./make_base_config.sh --files <input.root,...> --out <dir>/base_reco_cfg.py

    and either pass chassis=base_reco_cfg with <dir> on PYTHONPATH, or let the retraining
    entry scripts do it. Resolving by name (rather than a fixed import) means a renamed or
    missing base config gives a clear error listing what was tried. The validation and
    monitoring paths of the base config are pruned below, so a validation config is a fine
    base."""
    import importlib
    forced = options.chassis
    names = [forced] if forced else list(candidates)
    tried = []
    for _n in names:
        try:
            _mod = importlib.import_module(_n)
        except ImportError as _e:
            tried.append("%s (%s)" % (_n, _e))
            continue
        print("trackNano: base config = %s" % _n)
        return _mod.process
    raise ImportError(
        "trackNano_config: no base reconstruction config found. Tried: " + "; ".join(tried)
        + ".\n  Generate one, from the events you want to reconstruct, with:\n"
        "    RecoTracker/PixelSeeding/test/models/make_base_config.sh \\\n"
        "        --files <a.root,b.root|dir|list.txt> --out <dir>/base_reco_cfg.py\n"
        "  then run this config with chassis=base_reco_cfg and <dir> on PYTHONPATH.\n"
        "  The retraining entry script (retrain_prompt.sh) does all of that for you when it is\n"
        "  given --input <event files>.")


# Only the generated chassis is a candidate: a private configuration picked up from the checkout
# would reconstruct on whatever geometry, era and aging scenario it happens to carry. Name one
# explicitly with the chassis option if that is really what is wanted.
process = _import_chassis(["base_reco_cfg"])


# The displaced CA module label differs between menus (SoADisplaced / SoALowPt). Resolve it once
# (it is also what the InputTags / clone(trackSrc=) need). None when the base config runs no
# displaced iteration: the displaced tables are then left out.
def _resolve_label(*names):
    for _n in names:
        if hasattr(process, _n):
            return _n
    return None


_dispLabel = _resolve_label("hltPhase2PixelTracksSoADisplaced", "hltPhase2PixelTracksSoALowPt")
if _dispLabel is None:
    print("trackNano: the base config runs no displaced iteration; writing the prompt tables only")

# The input files are the base config's own unless inputFiles names others.
_files = [f for f in options.inputFiles if f]
if _files:
    process.source.fileNames = cms.untracked.vstring(
        [(f if f.startswith("file:") or f.startswith("/store") else "file:" + f) for f in _files])

process.maxEvents.input = cms.untracked.int32(options.maxEvents)
process.source.skipEvents = cms.untracked.uint32(options.skipEvents)

# dispLoosen scales the displaced iteration's longitudinal doublet cuts, so a dataset can be
# dumped on the looser population a retrained chain is meant to reach. It scales what the producer
# carries, so it stays correct when those cuts are retuned.
_dl = options.dispLoosen or None
if _dl is not None and _dispLabel:
    s = float(_dl); _dc = getattr(process, _dispLabel).doubletCuts
    _dc.maxZ0 = cms.vdouble([v * s for v in _dc.maxZ0])
    _dc.maxDZ = cms.vdouble([v * s for v in _dc.maxDZ])
    _dc.minDZ = cms.vdouble([v * s for v in _dc.minDZ])
    print("trackNano: displaced z0/dz cuts x%g" % s)

# doStats=1 turns on the occupancy summary, which reports a buffer that overflowed instead
# of letting it drop tracks silently. (To enlarge the buffers, use promptCaps / dispCaps below,
# which scale whatever the producer carries.)
if options.doStats:
    process.hltPhase2PixelTracksSoA.fillStatistics = cms.bool(True)
    if _dispLabel:
        getattr(process, _dispLabel).fillStatistics = cms.bool(True)
    print("trackNano: buffer occupancy summary on")

# ===========================================================================================
# CHAIN-STATE HOOKS -- one set per iteration
# ===========================================================================================
# Every training dump must be produced with the upstream models in the state the retrained
# chain will run in:
#   tripletDNNThreshold     (upstream of everything)    -> the deployed gate
#   useTrackDNN/Threshold   track-DNN dataset -> OFF; final-HP dataset -> ON at deployed thr
#   final HP selector        (downstream)                -> irrelevant to both dumps
# DEFAULT = inherit the producer config, so once a new working point lands in the cfi the next
# retrain picks it up. Overrides exist for dumping at a working point not in the config yet;
# every one is printed, so no dataset's provenance is ambiguous:
#   promptGate / promptGateOff / promptTrackDNN / promptTrackThr / promptCaps  and  disp* equivalents.
# A track the track DNN rejects keeps its quality, so it is present in a 'loose' dump either way;
# the useTrackDNN hook exists so the dump state is set explicitly rather than assumed.
# ===========================================================================================


def _scale_str_cap(mod, name, factor):
    """Scale a cap written as a formula in x (the hit count) by `factor`, by wrapping it as
    'factor*(expr)'. That is correct for any form -- a plain number, a linear or a quadratic
    formula -- whereas ASSIGNING a fixed formula can silently shrink a cap the arm has since
    outgrown. No-op if the parameter is absent."""
    if not hasattr(mod, name):
        return
    old = getattr(mod, name).value()
    setattr(mod, name, cms.string("%g*(%s)" % (factor, old)))


def _scale_dbl_cap(mod, name, factor):
    if not hasattr(mod, name):
        return
    setattr(mod, name, cms.double(float(getattr(mod, name).value()) * factor))


def _apply_chain_overrides(mod, prefix):
    """Apply the <prefix>* overrides (prefix = prompt | disp) to one CA producer; return what was
    applied."""
    def opt(name):
        return getattr(options, prefix + name)
    ov = []
    if opt("Caps") == "big":
        # doublets are NOT scaled: the doublet count does not depend on the (switched-off) track DNN,
        # and x4 pushes Kernel_connect past the 65535-block grid limit on GPU at PU200.
        # <prefix>CapsFactor (default 4) sets the enlargement; a training dump must not lose
        # items to container truncation (the dropped population would bias the model).
        _f = opt("CapsFactor")
        _scale_str_cap(mod, "maxNumberOfTuples", _f)
        _scale_dbl_cap(mod, "avgCellsPerCell", _f)
        _scale_dbl_cap(mod, "avgTracksPerCell", _f)
        mod.fillStatistics = cms.bool(True)
        ov.append("caps=tuples/cells x%g" % _f)
    if opt("Gate"):
        mod.tripletDNNThreshold = cms.double(float(opt("Gate")))
        ov.append("gate=%s" % opt("Gate"))
    if opt("GateOff"):
        mod.useTripletDNN = cms.bool(False)
        ov.append("gateOFF")
    if opt("TrackDNN"):
        mod.useTrackDNN = cms.bool(opt("TrackDNN") == "1")
        ov.append("trackDNN=%s" % opt("TrackDNN"))
    if opt("TrackThr"):
        mod.trackDNNThreshold = cms.double(float(opt("TrackThr")))
        ov.append("trackThr=%s" % opt("TrackThr"))
    return ov


def _shown(mod, name):
    """Value of a producer parameter for the provenance line. A parameter the python
    configuration does not set takes the default declared in the module's C++ fillDescriptions,
    which cannot be read from here -- say so instead of guessing or crashing."""
    p = getattr(mod, name, None)
    return p.value() if p is not None else "<C++ default>"


_pm = process.hltPhase2PixelTracksSoA
_dm = getattr(process, _dispLabel) if _dispLabel else None
_ovPrompt = _apply_chain_overrides(_pm, "prompt")
_ovDisp = _apply_chain_overrides(_dm, "disp") if _dm is not None else []

# The dump's provenance line, one per iteration. Read it in the log before using a dataset: the
# dataset is only valid for the chain state printed here.
for _label, _mod, _ov in (("prompt", _pm, _ovPrompt), ("displaced", _dm, _ovDisp)):
    if _mod is None:
        continue
    print("[trackNano CHAIN-STATE] %s: useTripletDNN=%s tripletDNNThreshold=%s | useTrackDNN=%s "
          "trackDNNThreshold=%s | maxNumberOfTuples=%s avgCellsPerCell=%s avgTracksPerCell=%s | "
          "minQuality=%s | overrides=[%s]"
          % (_label, _shown(_mod, "useTripletDNN"), _shown(_mod, "tripletDNNThreshold"),
             _shown(_mod, "useTrackDNN"), _shown(_mod, "trackDNNThreshold"),
             _shown(_mod, "maxNumberOfTuples"), _shown(_mod, "avgCellsPerCell"),
             _shown(_mod, "avgTracksPerCell"), options.minQuality,
             ",".join(_ov) if _ov else "none (inheriting the configuration)"))

# Prune validation/DQM paths: only the HLT reconstruction is needed.
# Also prune the base config's own output EndPath (<eventcontent>output_step) and drop every
# output module it declares: this dump writes its own NanoAOD file below, and a chassis made
# without cmsDriver's '--output={}' would otherwise write a full GEN-SIM-DIGI-RAW file next to
# every table -- tens of GB and a large share of the wall time, silently.
_keep = []
for _p in process.schedule:
    _lbl = _p.label_()
    if any(k in _lbl for k in ("alidation", "DQM", "dqm")) or _lbl.endswith("output_step"):
        continue
    _keep.append(_p)
process.schedule = cms.Schedule(*_keep)
for _om in list(process.outputModules_()):
    delattr(process, _om)

# ---------------------------------------------------------------------------
# Associator (same modules the validation uses)
from Validation.RecoTrack.associators_cff import hltTPClusterProducer, hltTrackAssociatorByHits
process.hltTPClusterProducer = hltTPClusterProducer
process.hltTrackAssociatorByHits = hltTrackAssociatorByHits

from HLTrigger.NGTScouting.hltTracks_cfi import (hltPixelTrackSoATable, hltPixelTrackTable,
                                                 pixelTrackAssoc, tpSelectorPixelTracks)

# Legacy conversions of the PRE-HP SoAs (truth row space; minQuality tight = the
# population the HP/tight selection actually decides on).
_convCommon = dict(
    beamSpot=cms.InputTag("hltOnlineBeamSpot"),
    pixelRecHitLegacySrc=cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc=cms.InputTag("hltSiPhase2RecHits"),
    otRecHitsSoASrc=cms.InputTag("hltCAOTRecHitsSoA"),
    vectorHitsSoASrc=cms.InputTag("hltPhase2OTVectorHitsSoA"),
    minNumberOfHits=cms.int32(0),
    minQuality=cms.string(options.minQuality),  # 'loose' = full candidate population (track-DNN dataset)
    useOTExtension=cms.bool(True),
    expandVectorHits=cms.bool(True),
    requireQuadsFromConsecutiveLayers=cms.bool(False),
)
process.trainPromptTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
                                           trackSrc=cms.InputTag("hltPhase2PixelTracksSoA"),
                                           **_convCommon)
if _dispLabel:
    process.trainDispTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
                                             trackSrc=cms.InputTag(_dispLabel),
                                             **_convCommon)

# Displaced-friendly TP selection: keep displaced TPs (tip<60cm), in/out-of-time PU real.
tpSelectorDisplaced = tpSelectorPixelTracks.clone(
    tip=cms.double(60.0),
    lip=cms.double(80.0),
    ptMin=cms.double(0.4),
    intimeOnly=cms.bool(False),
)

# All-inclusive TP selection, i.e. the track validation's fake definition: a track is fake only
# when it is associated to no TrackingParticle at all, while the narrow selections above are the
# efficiency denominators. A track matched to a real particle that fails one of their cuts is real
# for the validation and fake by the narrow label, so the truth tables can carry both `matched`
# (narrow, the efficiency class) and `matchedAny` (inclusive, the training target). Only charge is
# cut here, as the hit associator needs it.
tpSelectorAnyTP = tpSelectorPixelTracks.clone(
    ptMin=cms.double(0.0),
    tip=cms.double(1000.0),
    lip=cms.double(1000.0),
    minRapidity=cms.double(-10.0),
    maxRapidity=cms.double(10.0),
    intimeOnly=cms.bool(False),
    signalOnly=cms.bool(False),
    stableOnly=cms.bool(False),
    chargedOnly=cms.bool(True),
    minHit=cms.int32(0),
)

# mtvLabel=1 adds the inclusive association (and the matchedAny/duplicateAny columns) to
# the truth tables; off by default because it changes those tables' schema.
_MTV_LABEL = options.mtvLabel

process.trainPromptAssoc = pixelTrackAssoc.clone(trackCollection="trainPromptTracks")
if _MTV_LABEL:
    process.trainPromptAssocAny = pixelTrackAssoc.clone(trackCollection="trainPromptTracks",
                                                        tpSelectorPSet=tpSelectorAnyTP)
if _dispLabel:
    process.trainDispAssoc = pixelTrackAssoc.clone(trackCollection="trainDispTracks",
                                                   tpSelectorPSet=tpSelectorDisplaced)
    if _MTV_LABEL:
        process.trainDispAssocAny = pixelTrackAssoc.clone(trackCollection="trainDispTracks",
                                                          tpSelectorPSet=tpSelectorAnyTP)

# Feature tables: ALL SoA tracks, exact deployment ABI.
process.trainPromptFullTable = hltPixelTrackSoATable.clone(src="hltPhase2PixelTracksSoA", name="TrkPromptFull")
if _dispLabel:
    process.trainDispFullTable = hltPixelTrackSoATable.clone(src=_dispLabel, name="TrkDispFull")

# CA-features table: the hit/vector hit features of the in-kernel track DNN plus the extra columns,
# host-computed via the shared CATrackFeatures.h; same row space as TrkDispFull.
if _dispLabel:
    process.trainDispCATable = cms.EDProducer(
        "CATrackFeaturesTableProducer",
        tableName=cms.string("TrkDispCA"),
        trackSrc=cms.InputTag(_dispLabel),
        pixelRecHitSrc=cms.InputTag("hltPhase2SiPixelRecHitsSoA"),
        vectorHitsSrc=cms.InputTag("hltPhase2OTVectorHitsSoA"),
        otRecHitsSoASrc=cms.InputTag("hltCAOTRecHitsSoA"),
        otRecHitSrc=cms.InputTag("hltSiPhase2RecHits"),
        # Attach-purity: also emit a per-(track,hit) truth table (isOTExtra/isVectorHit/isTrueForOwnTP).
        emitHitTruth=cms.bool(True),
        hitTableName=cms.string("TrkDispCAHit"),
    )

# Same CA hit/vector hit features for the PROMPT iteration (the prompt CA also runs on the
# vector-hits-merged hit collection), for the prompt track DNN with the same 12-feature ABI as the
# displaced one. Index-aligned to TrkPromptFull.
process.trainPromptCATable = cms.EDProducer(
    "CATrackFeaturesTableProducer",
    tableName=cms.string("TrkPromptCA"),
    trackSrc=cms.InputTag("hltPhase2PixelTracksSoA"),
    pixelRecHitSrc=cms.InputTag("hltPhase2SiPixelRecHitsSoA"),
    vectorHitsSrc=cms.InputTag("hltPhase2OTVectorHitsSoA"),
    otRecHitsSoASrc=cms.InputTag("hltCAOTRecHitsSoA"),
    otRecHitSrc=cms.InputTag("hltSiPhase2RecHits"),
    # Attach-purity: also emit a per-(track,hit) truth table (isOTExtra/isVectorHit/isTrueForOwnTP).
    emitHitTruth=cms.bool(True),
    hitTableName=cms.string("TrkPromptCAHit"),
)

# Truth tables on the legacy conversions (kinematics for the join + truth columns).
# assocAny (optional): a second association run with tpSelectorAnyTP, contributing matchedAny /
# duplicateAny. `matched` keeps its meaning (the efficiency-selected match); matchedAny is the label
# a fake-rejection model is trained against, and by construction matchedAny >= matched row-wise.
def _truthTable(src, name, assoc, assocAny=None):
    t = hltPixelTrackTable.clone(src=src, name=name)
    for _v in t.externalVariables.parameterNames_():
        getattr(t.externalVariables, _v).src = cms.InputTag(assoc, getattr(t.externalVariables, _v).src.getProductInstanceLabel())
    if assocAny:
        t.externalVariables.matchedAny = cms.PSet(
            src=cms.InputTag(assocAny, "matched"),
            doc=cms.string("1 if matched to ANY TrackingParticle (the track validation calls a "
                           "track fake iff this is 0). Training target for fake rejection."),
            type=cms.string("uint8"))
        t.externalVariables.duplicateAny = cms.PSet(
            src=cms.InputTag(assocAny, "duplicate"),
            doc=cms.string("1 if multiple reco tracks map to the same TP, under the all-inclusive "
                           "TP selection"),
            type=cms.string("uint8"))
    return t

process.trainPromptTruthTable = _truthTable("trainPromptTracks", "TrkPromptTruth", "trainPromptAssoc",
                                            "trainPromptAssocAny" if _MTV_LABEL else None)
if _dispLabel:
    process.trainDispTruthTable = _truthTable("trainDispTracks", "TrkDispTruth", "trainDispAssoc",
                                              "trainDispAssocAny" if _MTV_LABEL else None)

process.trainNanoTask = cms.Task(
    process.hltTPClusterProducer,
    process.hltTrackAssociatorByHits,
    process.trainPromptTracks,
    process.trainPromptAssoc,
)
# The CA tables also emit the per-hit purity tables Trk<X>CAHit.
process.trainNanoPath = cms.Path(
    process.trainPromptFullTable + process.trainPromptCATable + process.trainPromptTruthTable,
    process.trainNanoTask,
)
if _MTV_LABEL:
    process.trainNanoTask.add(process.trainPromptAssocAny)
if _dispLabel:
    process.trainNanoTask.add(process.trainDispTracks, process.trainDispAssoc)
    if _MTV_LABEL:
        process.trainNanoTask.add(process.trainDispAssocAny)
    process.trainNanoPath += process.trainDispFullTable + process.trainDispCATable + process.trainDispTruthTable
process.schedule.append(process.trainNanoPath)

_out = options.outputFile or "trackNano_%s.root" % _sample
process.trainNanoOut = cms.OutputModule(
    "NanoAODOutputModule",
    fileName=cms.untracked.string(_out),
    outputCommands=cms.untracked.vstring(
        "drop *",
        "keep nanoaodFlatTable_train*_*_*",
    ),
)
process.trainNanoEnd = cms.EndPath(process.trainNanoOut)
process.schedule.append(process.trainNanoEnd)

# Thread/stream count: the nano tables carry the event structure (no ordering constraint), so
# allow many threads via the threads option for fast parallel dump jobs.
_nthr = options.threads
process.options.numberOfThreads = _nthr
process.options.numberOfStreams = _nthr

print("trackNano: sample=%s nevt=%d out=%s paths=%d threads=%d" % (
    _sample, process.maxEvents.input.value(), _out, len(list(process.schedule)), _nthr))
