import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi 
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
import Validation.RecoTrack.MultiTrackValidator_cfi
from Validation.RecoTrack.trajectorySeedTracks_cfi import trajectorySeedTracks as _trajectorySeedTracks
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import cutsRecoTracks_cfi

from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
from CommonTools.RecoAlgos.trackingParticleConversionRefSelector_cfi import trackingParticleConversionRefSelector as _trackingParticleConversionRefSelector
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *
from CommonTools.RecoAlgos.recoChargedRefCandidateToTrackRefProducer_cfi import recoChargedRefCandidateToTrackRefProducer as _recoChargedRefCandidateToTrackRefProducer

import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from Configuration.Eras.Modifier_fastSim_cff import fastSim

### First define the stuff for the standard validation sequence
## Track selectors
for _eraName, _postfix, _era in _cfg.allEras():
    _seedProd = ["initialStepSeedsPreSplitting"]
    _trackProd = ["initialStepTracksPreSplitting"]
    if _eraName in ["trackingLowPU", "trackingPhase1PU70", "trackingPhase2PU140"]: # these don't have preSplitting
        _seedProd = []
        _trackProd = []

    locals()["_algos"+_postfix] = ["generalTracks"] + _cfg.iterationAlgos(_postfix) + ["duplicateMerge"]
    locals()["_seedProducers"+_postfix] = _seedProd + _cfg.seedProducers(_postfix)
    locals()["_trackProducers"+_postfix] = _trackProd + _cfg.trackProducers(_postfix)

_removeForFastSimSeedProducers =["initialStepSeedsPreSplitting",
                                 "jetCoreRegionalStepSeeds",
                                 "muonSeededSeedsInOut",
                                 "muonSeededSeedsOutIn"]
_seedProducers_fastSim = [ x for x in _seedProducers if x not in _removeForFastSimSeedProducers]

_removeForFastTrackProducers = ["initialStepTracksPreSplitting",
                                "jetCoreRegionalStepTracks",
                                "muonSeededTracksInOut",
                                "muonSeededTracksOutIn"]
_trackProducers_fastSim = [ x for x in _trackProducers if x not in _removeForFastTrackProducers]

def _algoToSelector(algo):
    sel = ""
    if algo != "generalTracks":
        sel = algo[0].upper()+algo[1:]
    return "cutsRecoTracks"+sel

def _addSelectorsByAlgo(algos, modDict):
    names = []
    seq = cms.Sequence()
    for algo in algos:
        if algo == "generalTracks":
            continue
        modName = _algoToSelector(algo)
        if modName not in modDict:
            mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=[algo])
            modDict[modName] = mod
        else:
            mod = modDict[modName]
        names.append(modName)
        seq += mod
    return (names, seq)
def _addSelectorsByHp(algos, modDict):
    seq = cms.Sequence()
    names = []
    for algo in algos:
        modName = _algoToSelector(algo)
        modNameHp = modName+"Hp"
        if modNameHp not in modDict:
            if algo == "generalTracks":
                mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(quality=["highPurity"])
            else:
                mod = modDict[modName].clone(quality=["highPurity"])
            modDict[modNameHp] = mod
        else:
            mod = modDict[modNameHp]
        names.append(modNameHp)
        seq += mod
    return (names, seq)
def _addSelectorsBySrc(modules, midfix, src, modDict):
    seq = cms.Sequence()
    names = []
    for modName in modules:
        modNameNew = modName.replace("cutsRecoTracks", "cutsRecoTracks"+midfix)
        if modNameNew not in modDict:
            mod = modDict[modName].clone(src=src)
            modDict[modNameNew] = mod
        else:
            mod = modDict[modNameNew]
        names.append(modNameNew)
        seq += mod
    return (names, seq)
def _addSelectorsByOriginalAlgoMask(modules, midfix, algoParam,modDict):
    seq = cms.Sequence()
    names = []
    for modName in modules:
        if modName[-2:] == "Hp":
            modNameNew = modName[:-2] + midfix + "Hp"
        else:
            modNameNew = modName + midfix
        if modNameNew not in modDict:
            mod = modDict[modName].clone()
            setattr(mod, algoParam, mod.algorithm.value())
            mod.algorithm = []
            modDict[modNameNew] = mod
        else:
            mod = modDict[modNameNew]
        names.append(modNameNew)
        seq += mod
    return (names, seq)
def _addSeedToTrackProducers(seedProducers,modDict):
    names = []
    seq = cms.Sequence()
    for seed in seedProducers:
        modName = "seedTracks"+seed
        if modName not in modDict:
            mod = _trajectorySeedTracks.clone(src=seed)
            modDict[modName] = mod
        else:
            mod = modDict[modName]
        names.append(modName)
        seq += mod
    return (names, seq)

_relevantEras = _cfg.allEras()
_relevantErasAndFastSim = _relevantEras + [("fastSim", "_fastSim", fastSim)]
def _translateArgs(args, postfix, modDict):
    ret = []
    for arg in args:
        if isinstance(arg, list):
            ret.append(_translateArgs(arg, postfix, modDict))
        else:
            ret.append(modDict[arg+postfix])
    return ret
def _sequenceForEachEra(function, args, names, sequence, modDict, plainArgs=[], modifySequence=None, includeFastSim=False):
    if sequence[0] != "_":
        raise Exception("Sequence name is expected to begin with _")

    _eras = _relevantErasAndFastSim if includeFastSim else _relevantEras
    for eraName, postfix, _era in _eras:
        _args = _translateArgs(args, postfix, modDict)
        _args.extend(plainArgs)
        ret = function(*_args, modDict=modDict)
        if len(ret) != 2:
            raise Exception("_sequenceForEachEra is expected to return 2 values, but function returned %d" % len(ret))
        modDict[names+postfix] = ret[0]
        modDict[sequence+postfix] = ret[1]

    # The sequence of the first era will be the default one
    defaultSequenceName = sequence+_eras[0][0]
    defaultSequence = modDict[defaultSequenceName]
    modDict[defaultSequenceName[1:]] = defaultSequence # remove leading underscore

    # Optionally modify sequences before applying the era
    if modifySequence is not None:
        for eraName, postfix, _era in _eras:
            modifySequence(modDict[sequence+postfix])

    # Apply eras
    for _eraName, _postfix, _era in _eras[1:]:
        _era.toReplaceWith(defaultSequence, modDict[sequence+_postfix])
def _setForEra(module, eraName, era, **kwargs):
    if eraName == "":
        for key, value in kwargs.iteritems():
            setattr(module, key, value)
    else:
        era.toModify(module, **kwargs)

# Seeding layer sets
def _getSeedingLayers(seedProducers):
    import RecoTracker.IterativeTracking.iterativeTk_cff as _iterativeTk_cff

    seedingLayersMerged = []
    for seedName in seedProducers:
        seedProd = getattr(_iterativeTk_cff, seedName)
        if not hasattr(seedProd, "OrderedHitsFactoryPSet"):
            continue

        if hasattr(seedProd, "SeedMergerPSet"):
            seedingLayersName = seedProd.SeedMergerPSet.layerList.refToPSet_.value()
        else:
            seedingLayersName = seedProd.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
        seedingLayers = getattr(_iterativeTk_cff, seedingLayersName).layerList.value()
        for layerSet in seedingLayers:
            if layerSet not in seedingLayersMerged:
                seedingLayersMerged.append(layerSet)
    return seedingLayersMerged
for _eraName, _postfix, _era in _relevantEras:
    locals()["_seedingLayerSets"+_postfix] = _getSeedingLayers(locals()["_seedProducers"+_postfix])

# MVA selectors
def _getMVASelectors(postfix):
    import RecoTracker.IterativeTracking.iterativeTk_cff as _iterativeTk_cff

    # assume naming convention that the iteration name (when first
    # letter in lower case) is the selector name
    pset = cms.untracked.PSet()
    for iterName, seqName in _cfg.iterationAlgos(postfix, includeSequenceName=True):
        if hasattr(_iterativeTk_cff, iterName):
            mod = getattr(_iterativeTk_cff, iterName)
            seq = getattr(_iterativeTk_cff, seqName)

            # Ignore iteration if the MVA selector module is not in the sequence
            try:
                seq.index(mod)
            except:
                continue

            typeName = mod._TypedParameterizable__type
            classifiers = []
            if typeName == "ClassifierMerger":
                classifiers = mod.inputClassifiers.value()
            elif "TrackMVAClassifier" in typeName:
                classifiers = [iterName]
            if len(classifiers) > 0:
                setattr(pset, iterName+"Tracks", cms.untracked.vstring(classifiers))

    return pset
for _eraName, _postfix, _era in _relevantEras:
    locals()["_mvaSelectors"+_postfix] = _getMVASelectors(_postfix)

# Validation iterative steps
_sequenceForEachEra(_addSelectorsByAlgo, args=["_algos"], names="_selectorsByAlgo", sequence="_tracksValidationSelectorsByAlgo", modDict=globals())

# high purity
_sequenceForEachEra(_addSelectorsByHp, args=["_algos"], names="_selectorsByAlgoHp", sequence="_tracksValidationSelectorsByAlgoHp", modDict=globals())

for _eraName, _postfix, _era in _relevantEras:
    selectors = locals()["_selectorsByAlgoHp"+_postfix]
    locals()["_generalTracksHp"+_postfix] = selectors[0]
    locals()["_selectorsByAlgoHp"+_postfix] = selectors[1:]

# BTV-like selection
import PhysicsTools.RecoAlgos.btvTracks_cfi as btvTracks_cfi
cutsRecoTracksBtvLike = btvTracks_cfi.btvTrackRefs.clone()

# Select tracks associated to AK4 jets
import RecoJets.JetAssociationProducers.ak4JTA_cff as ak4JTA_cff
ak4JetTracksAssociatorExplicitAll = ak4JTA_cff.ak4JetTracksAssociatorExplicit.clone(
    jets = "ak4PFJets"
)
from JetMETCorrections.Configuration.JetCorrectors_cff import *
import CommonTools.RecoAlgos.jetTracksAssociationToTrackRefs_cfi as jetTracksAssociationToTrackRefs_cfi
cutsRecoTracksAK4PFJets = jetTracksAssociationToTrackRefs_cfi.jetTracksAssociationToTrackRefs.clone(
    association = "ak4JetTracksAssociatorExplicitAll",
    jets = "ak4PFJets",
    correctedPtMin = 10,
)


## Select signal TrackingParticles, and do the corresponding associations
trackingParticlesSignal = _trackingParticleRefSelector.clone(
    signalOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)

# select tracks with pT > 0.9 GeV (for upgrade fake rates)
generalTracksPt09 = cutsRecoTracks_cfi.cutsRecoTracks.clone(ptMin=0.9)
# and then the selectors
_sequenceForEachEra(_addSelectorsBySrc, modDict=globals(),
                    args=[["_generalTracksHp"]],
                    plainArgs=["Pt09", "generalTracksPt09"],
                    names="_selectorsPt09", sequence="_tracksValidationSelectorsPt09",
                    modifySequence=lambda seq:seq.insert(0, generalTracksPt09))

# select tracks from the PV
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import trackWithVertexRefSelector as _trackWithVertexRefSelector
generalTracksFromPV = _trackWithVertexRefSelector.clone(
    src = "generalTracks",
    ptMin = 0,
    ptMax = 1e10,
    ptErrorCut = 1e10,
    quality = "loose",
    vertexTag = "offlinePrimaryVertices",
    nVertices = 1,
    vtxFallback = False,
    zetaVtx = 0.1, # 1 mm
    rhoVtx = 1e10, # intentionally no dxy cut
)
# and then the selectors
_sequenceForEachEra(_addSelectorsBySrc, modDict=globals(),
                    args=[["_generalTracksHp"]],
                    plainArgs=["FromPV", "generalTracksFromPV"],
                    names="_selectorsFromPV", sequence="_tracksValidationSelectorsFromPV",
                    modifySequence=lambda seq: seq.insert(0, generalTracksFromPV))

# select tracks with pT > 0.9 GeV from the PV
generalTracksFromPVPt09 = generalTracksPt09.clone(src="generalTracksFromPV")
# and then the selectors
_sequenceForEachEra(_addSelectorsBySrc, modDict=globals(),
                    args=[["_generalTracksHp"]],
                    plainArgs=["FromPVPt09", "generalTracksFromPVPt09"],
                    names="_selectorsFromPVPt09", sequence="_tracksValidationSelectorsFromPVPt09",
                    modifySequence=lambda seq: seq.insert(0, generalTracksFromPVPt09))

## Select conversion TrackingParticles, and define the corresponding associator
trackingParticlesConversion = _trackingParticleConversionRefSelector.clone()

## Select electron TPs
trackingParticlesElectron = _trackingParticleRefSelector.clone(
    pdgId = [-11, 11],
    signalOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)

## MTV instances
trackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone(
    useLogPt = cms.untracked.bool(True),
    dodEdxPlots = True,
    doPVAssociationPlots = True
    #,minpT = cms.double(-1)
    #,maxpT = cms.double(3)
    #,nintpT = cms.int32(40)
)
fastSim.toModify(trackValidator, 
                      dodEdxPlots = False)

for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidator, _eraName, _era,
               label = ["generalTracks", locals()["_generalTracksHp"+_postfix]] +
                       locals()["_selectorsByAlgo"+_postfix] + locals()["_selectorsByAlgoHp"+_postfix] +
                       ["generalTracksPt09"] + locals()["_selectorsPt09"+_postfix] +
               [
                   "cutsRecoTracksBtvLike",
                   "cutsRecoTracksAK4PFJets"
               ]
    )
    _setForEra(trackValidator.histoProducerAlgoBlock, _eraName, _era, seedingLayerSets=locals()["_seedingLayerSets"+_postfix])

# For efficiency of signal TPs vs. signal tracks, and fake rate of
# signal tracks vs. signal TPs
trackValidatorFromPV = trackValidator.clone(
    dirName = "Tracking/TrackFromPV/",
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
    trackCollectionForDrCalculation = "generalTracksFromPV",
    doPlotsOnlyForTruePV = True,
    doPVAssociationPlots = False,
)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorFromPV, _eraName, _era, label = ["generalTracksFromPV"] + locals()["_selectorsFromPV"+_postfix] + ["generalTracksFromPVPt09"] + locals()["_selectorsFromPVPt09"+_postfix])

# For fake rate of signal tracks vs. all TPs, and pileup rate of
# signal tracks vs. non-signal TPs
trackValidatorFromPVAllTP = trackValidatorFromPV.clone(
    dirName = "Tracking/TrackFromPVAllTP/",
    label_tp_effic = trackValidator.label_tp_effic.value(),
    label_tp_fake = trackValidator.label_tp_fake.value(),
    label_tp_effic_refvector = False,
    label_tp_fake_refvector = False,
    associators = trackValidator.associators.value(),
    doSimPlots = False,
    doSimTrackPlots = False,
)

# For efficiency of all TPs vs. all tracks
trackValidatorAllTPEffic = trackValidator.clone(
    dirName = "Tracking/TrackAllTPEffic/",
    label = [x for x in trackValidator.label.value() if "Pt09" not in x],
    doSimPlots = False,
    doRecoTrackPlots = False, # Fake rate of all tracks vs. all TPs is already included in trackValidator
    doPVAssociationPlots = False,
)
trackValidatorAllTPEffic.histoProducerAlgoBlock.generalTpSelector.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR.signalOnly = False
trackValidatorAllTPEffic.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ.signalOnly = False
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorAllTPEffic, _eraName, _era, label = ["generalTracks", locals()["_generalTracksHp"+_postfix]])

# For conversions
trackValidatorConversion = trackValidator.clone(
    dirName = "Tracking/TrackConversion/",
    label = [
        "convStepTracks",
        "conversionStepTracks",
        "ckfInOutTracksFromConversions",
        "ckfOutInTracksFromConversions",
    ],
    label_tp_effic = "trackingParticlesConversion",
    label_tp_effic_refvector = True,
    associators = ["quickTrackAssociatorByHits"],
    UseAssociators = True,
    doSimPlots = True,
    dodEdxPlots = False,
    doPVAssociationPlots = False,
    calculateDrSingleCollection = False,
)
# relax lip and tip
for n in ["Eta", "Phi", "Pt", "VTXR", "VTXZ"]:
    pset = getattr(trackValidatorConversion.histoProducerAlgoBlock, "TpSelectorForEfficiencyVs"+n)
    pset.lip = trackValidatorConversion.lipTP.value()
    pset.tip = trackValidatorConversion.tipTP.value()

# For electrons
trackValidatorGsfTracks = trackValidatorConversion.clone(
    dirName = "Tracking/TrackGsf/",
    label = ["electronGsfTracks"],
    label_tp_effic = "trackingParticlesElectron",
)


# the track selectors
tracksValidationSelectors = cms.Sequence(
    tracksValidationSelectorsByAlgo +
    tracksValidationSelectorsByAlgoHp +
    cutsRecoTracksBtvLike +
    ak4JetTracksAssociatorExplicitAll +
    cutsRecoTracksAK4PFJets
)
tracksValidationTruth = cms.Sequence(
    tpClusterProducer +
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation +
    VertexAssociatorByPositionAndTracks +
    trackingParticleNumberOfLayersProducer
)
fastSim.toModify(tracksValidationTruth, lambda x: x.remove(tpClusterProducer))

tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationSelectorsPt09 +
    tracksValidationSelectorsFromPV +
    tracksValidationSelectorsFromPVPt09 +
    tracksValidationTruth +
    cms.ignore(trackingParticlesSignal) +
    cms.ignore(trackingParticlesElectron) +
    trackingParticlesConversion
)
fastSim.toReplaceWith(tracksPreValidation, tracksPreValidation.copyAndExclude([
    trackingParticlesElectron,
    trackingParticlesConversion
]))

tracksValidation = cms.Sequence(
    tracksPreValidation +
    trackValidator +
    trackValidatorFromPV +
    trackValidatorFromPVAllTP +
    trackValidatorAllTPEffic +
    trackValidatorConversion +
    trackValidatorGsfTracks
)
fastSim.toReplaceWith(tracksValidation, tracksValidation.copyAndExclude([trackValidatorConversion, trackValidatorGsfTracks]))

### Then define stuff for standalone mode (i.e. MTV with RECO+DIGI input)

# Select by originalAlgo and algoMask
for _eraName, _postfix, _era in _relevantEras:
    locals()["_selectorsByAlgoAndHp"+_postfix] = locals()["_selectorsByAlgo"+_postfix] + locals()["_selectorsByAlgoHp"+_postfix]
_sequenceForEachEra(_addSelectorsByOriginalAlgoMask, modDict = globals(),
                    args = ["_selectorsByAlgoAndHp"], plainArgs = ["ByOriginalAlgo", "originalAlgorithm"],
                    names = "_selectorsByOriginalAlgo", sequence = "_tracksValidationSelectorsByOriginalAlgoStandalone")
_sequenceForEachEra(_addSelectorsByOriginalAlgoMask, modDict = globals(),
                    args = ["_selectorsByAlgoAndHp"], plainArgs = ["ByAlgoMask", "algorithmMaskContains"],
                    names = "_selectorsByAlgoMask", sequence = "_tracksValidationSelectorsByAlgoMaskStandalone")

# Select pT>0.9 by iteration
_sequenceForEachEra(_addSelectorsBySrc, modDict = globals(),
                    args = ["_selectorsByAlgoAndHp"], plainArgs = ["Pt09", "generalTracksPt09"],
                    names = "_selectorsPt09Standalone", sequence = "_tracksValidationSelectorsPt09Standalone")

# Select fromPV by iteration
_sequenceForEachEra(_addSelectorsBySrc, modDict = globals(),
                    args = ["_selectorsByAlgoAndHp"], plainArgs = ["FromPV", "generalTracksFromPV"],
                    names = "_selectorsFromPVStandalone", sequence = "_tracksValidationSelectorsFromPVStandalone")

# Select pt>0.9 and fromPV by iteration
_sequenceForEachEra(_addSelectorsBySrc, modDict = globals(),
                    args = ["_selectorsByAlgoAndHp"], plainArgs = ["FromPVPt09", "generalTracksFromPVPt09"],
                    names = "_selectorsFromPVPt09Standalone", sequence = "_tracksValidationSelectorsFromPVPt09Standalone")

# MTV instances
trackValidatorStandalone = trackValidator.clone()
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorStandalone, _eraName, _era, label = trackValidator.label + locals()["_selectorsByOriginalAlgo"+_postfix] + locals()["_selectorsByAlgoMask"+_postfix] + locals()["_selectorsPt09Standalone"+_postfix])

trackValidatorFromPVStandalone = trackValidatorFromPV.clone()
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorFromPVStandalone, _eraName, _era, label = trackValidatorFromPV.label + locals()["_selectorsFromPVStandalone"+_postfix] + locals()["_selectorsFromPVPt09Standalone"+_postfix])

trackValidatorFromPVAllTPStandalone = trackValidatorFromPVAllTP.clone(
    label = trackValidatorFromPVStandalone.label.value()
)
trackValidatorAllTPEfficStandalone = trackValidatorAllTPEffic.clone(
    label = [ x for x in trackValidator.label.value() if x not in ["cutsRecoTracksBtvLike", "cutsRecoTracksAK4PFJets"] and "Pt09" not in x]
)

trackValidatorConversionStandalone = trackValidatorConversion.clone( label = [x for x in trackValidatorConversion.label if x != "convStepTracks"])

# sequences
tracksValidationSelectorsStandalone = cms.Sequence(
    tracksValidationSelectorsByOriginalAlgoStandalone +
    tracksValidationSelectorsByAlgoMaskStandalone +
    tracksValidationSelectorsPt09Standalone +
    tracksValidationSelectorsFromPVStandalone +
    tracksValidationSelectorsFromPVPt09Standalone
)

# we copy this for both Standalone and TrackingOnly
#  and later make modifications from it which change based on era
_trackValidatorsBase = cms.Sequence(
    trackValidatorStandalone +
    trackValidatorFromPVStandalone +
    trackValidatorFromPVAllTPStandalone +
    trackValidatorAllTPEfficStandalone +
    trackValidatorConversionStandalone +
    trackValidatorGsfTracks
)
trackValidatorsStandalone = _trackValidatorsBase.copy()
fastSim.toModify(trackValidatorsStandalone, lambda x: x.remove(trackValidatorConversionStandalone) )

tracksValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain +
    tracksPreValidation +
    tracksValidationSelectorsStandalone +
    trackValidatorsStandalone
)

### TrackingOnly mode (i.e. MTV with DIGI input + tracking-only reconstruction)

# selectors
tracksValidationSelectorsTrackingOnly = tracksValidationSelectors.copyAndExclude([ak4JetTracksAssociatorExplicitAll,cutsRecoTracksAK4PFJets]) # selectors using track information only (i.e. no PF)
_sequenceForEachEra(_addSeedToTrackProducers, args=["_seedProducers"], names="_seedSelectors", sequence="_tracksValidationSeedSelectorsTrackingOnly", includeFastSim=True, modDict=globals())

# MTV instances
trackValidatorTrackingOnly = trackValidatorStandalone.clone(label = [ x for x in trackValidatorStandalone.label if x != "cutsRecoTracksAK4PFJets"] )

_trackValidatorSeedingBuildingTrackingOnly = trackValidatorTrackingOnly.clone( # common for seeds and built tracks
    associators = ["quickTrackAssociatorByHits"],
    UseAssociators = True,
    dodEdxPlots = False,
    doPVAssociationPlots = False,
    doSimPlots = False,
)
trackValidatorBuildingTrackingOnly = _trackValidatorSeedingBuildingTrackingOnly.clone(
    dirName = "Tracking/TrackBuilding/",
    doMVAPlots = True,
)
for _eraName, _postfix, _era in _relevantErasAndFastSim:
    _setForEra(trackValidatorBuildingTrackingOnly, _eraName, _era, label = locals()["_trackProducers"+_postfix])
fastSim.toModify(trackValidatorBuildingTrackingOnly, doMVAPlots=False)
for _eraName, _postfix, _era in _relevantEras:
    _setForEra(trackValidatorBuildingTrackingOnly, _eraName, _era, mvaLabels = locals()["_mvaSelectors"+_postfix])

trackValidatorSeedingTrackingOnly = _trackValidatorSeedingBuildingTrackingOnly.clone(
    dirName = "Tracking/TrackSeeding/",
    label = _seedSelectors,
    doSeedPlots = True,
)
for _eraName, _postfix, _era in _relevantErasAndFastSim:
    _setForEra(trackValidatorSeedingTrackingOnly, _eraName, _era, label = locals()["_seedSelectors"+_postfix])


trackValidatorConversionTrackingOnly = trackValidatorConversion.clone(label = [x for x in trackValidatorConversion.label if x not in ["ckfInOutTracksFromConversions", "ckfOutInTracksFromConversions"]])

# sequences
tracksPreValidationTrackingOnly = tracksPreValidation.copy()
tracksPreValidationTrackingOnly.replace(tracksValidationSelectors, tracksValidationSelectorsTrackingOnly)

trackValidatorsTrackingOnly = _trackValidatorsBase.copy()
trackValidatorsTrackingOnly.replace(trackValidatorStandalone, trackValidatorTrackingOnly)
trackValidatorsTrackingOnly += (
    trackValidatorSeedingTrackingOnly +
    trackValidatorBuildingTrackingOnly
)
trackValidatorsTrackingOnly.replace(trackValidatorConversionStandalone, trackValidatorConversionTrackingOnly)
trackValidatorsTrackingOnly.remove(trackValidatorGsfTracks)
fastSim.toModify(trackValidatorsTrackingOnly, lambda x: x.remove(trackValidatorConversionTrackingOnly))


tracksValidationTrackingOnly = cms.Sequence(
    tracksPreValidationTrackingOnly +
    tracksValidationSelectorsStandalone +
    tracksValidationSeedSelectorsTrackingOnly +
    trackValidatorsTrackingOnly
)



### Lite mode (only generalTracks and HP)
trackValidatorLite = trackValidator.clone(
    label = ["generalTracks", "cutsRecoTracksHp"]
)
tracksValidationLite = cms.Sequence(
    cutsRecoTracksHp +
    tracksValidationTruth +
    trackValidatorLite
)
