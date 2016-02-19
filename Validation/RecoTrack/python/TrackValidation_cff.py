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
from PhysicsTools.RecoAlgos.trackingParticleSelector_cfi import trackingParticleSelector as _trackingParticleSelector
from CommonTools.RecoAlgos.trackingParticleConversionSelector_cfi import trackingParticleConversionSelector as _trackingParticleConversionSelector
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import sortedPrimaryVertices as _sortedPrimaryVertices
from CommonTools.RecoAlgos.recoChargedRefCandidateToTrackRefProducer_cfi import recoChargedRefCandidateToTrackRefProducer as _recoChargedRefCandidateToTrackRefProducer

from Configuration.StandardSequences.Eras import eras

### First define the stuff for the standard validation sequence
## Track selectors
_algos = [
    "generalTracks",
    "initialStep",
    "lowPtTripletStep",
    "pixelPairStep",
    "detachedTripletStep",
    "mixedTripletStep",
    "pixelLessStep",
    "tobTecStep",
    "jetCoreRegionalStep",
    "muonSeededStepInOut",
    "muonSeededStepOutIn",
    "duplicateMerge",
]
_algosForTrackingPhase1 = [
    "generalTracks",
    "initialStep",
    "highPtTripletStep",
    "lowPtQuadStep",
    "lowPtTripletStep",
    "detachedQuadStep",
    "mixedTripletStep",
    "pixelPairStep",
    "tobTecStep",
    "muonSeededStepInOut",
    "muonSeededStepOutIn",
    ]

_seedProducers = [
    "initialStepSeedsPreSplitting",
    "initialStepSeeds",
    "detachedTripletStepSeeds",
    "lowPtTripletStepSeeds",
    "pixelPairStepSeeds",
    "mixedTripletStepSeedsA",
    "mixedTripletStepSeedsB",
    "pixelLessStepSeeds",
    "tobTecStepSeedsPair",
    "tobTecStepSeedsTripl",
    "jetCoreRegionalStepSeeds",
    "muonSeededSeedsInOut",
    "muonSeededSeedsOutIn",
]

_removeForFastSimSeedProducers =["initialStepSeedsPreSplitting",
                                 "jetCoreRegionalStepSeeds",
                                 "muonSeededSeedsInOut",
                                 "muonSeededSeedsOutIn"]
_seedProducersForFastSim = [ x for x in _seedProducers if x not in _removeForFastSimSeedProducers]


_seedProducersForTrackingPhase1 = [
        "initialStepSeeds",
        "highPtTripletStepSeeds",
        "lowPtQuadStepSeeds",
        "lowPtTripletStepSeeds",
        "detachedQuadStepSeeds",
        "mixedTripletStepSeedsA",
        "mixedTripletStepSeedsB",
        "pixelPairStepSeeds",
        "tobTecStepSeeds",
        "muonSeededSeedsInOut",
        "muonSeededSeedsOutIn",
    ]


_trackProducers = [
    "initialStepTracksPreSplitting",
    "initialStepTracks",
    "lowPtTripletStepTracks",
    "pixelPairStepTracks",
    "detachedTripletStepTracks",
    "mixedTripletStepTracks",
    "pixelLessStepTracks",
    "tobTecStepTracks",
    "jetCoreRegionalStepTracks",
    "muonSeededTracksInOut",
    "muonSeededTracksOutIn",
]
_removeForFastTrackProducers = ["initialStepTracksPreSplitting",
                                "jetCoreRegionalStepTracks",
                                "muonSeededTracksInOut",
                                "muonSeededTracksOutIn"]
_trackProducersForFastSim = [ x for x in _trackProducers if x not in _removeForFastTrackProducers]

_trackProducersForTrackingPhase1 = [
    "initialStepTracks",
    "highPtTripletStepTracks",
    "lowPtQuadStepTracks",
    "lowPtTripletStepTracks",
    "detachedQuadStepTracks",
    "mixedTripletStepTracks",
    "pixelPairStepTracks",
    "tobTecStepTracks",
    "muonSeededTracksInOut",
    "muonSeededTracksOutIn",
]

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
            globals()[modName] = mod
        else:
            mod = modDict[modName]
        names.append(modName)
        seq += mod
    return (names, seq)

# Validation iterative steps
(_selectorsByAlgo, tracksValidationSelectorsByAlgo) = _addSelectorsByAlgo(_algos, globals())
(_selectorsByAlgo_trackingPhase1, _tracksValidationSelectorsByAlgo_trackingPhase1) = _addSelectorsByAlgo(_algosForTrackingPhase1, globals())
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsByAlgo, _tracksValidationSelectorsByAlgo_trackingPhase1)

# high purity
(_selectorsByAlgoHp, tracksValidationSelectorsByAlgoHp) = _addSelectorsByHp(_algos,globals())
(_selectorsByAlgoHp_trackingPhase1, _tracksValidationSelectorsByAlgoHp_trackingPhase1) = _addSelectorsByHp(_algosForTrackingPhase1, globals())
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsByAlgoHp, _tracksValidationSelectorsByAlgoHp_trackingPhase1)

_generalTracksHp = _selectorsByAlgoHp[0]
_generalTracksHp_trackingPhase1 = _selectorsByAlgoHp_trackingPhase1[0]
_selectorsByAlgoHp = _selectorsByAlgoHp[1:]
_selectorsByAlgoHp_trackingPhase1 = _selectorsByAlgoHp_trackingPhase1[1:]

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
trackingParticlesSignal = _trackingParticleSelector.clone(
    signalOnly = True,
    chargedOnly = False,
    tip = 1e5,
    lip = 1e5,
    minRapidity = -10,
    maxRapidity = 10,
    ptMin = 0,
)
tpClusterProducerSignal = tpClusterProducer.clone(
    trackingParticleSrc = "trackingParticlesSignal"
)
quickTrackAssociatorByHitsSignal = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "tpClusterProducerSignal"
)
trackingParticleRecoTrackAsssociationSignal = trackingParticleRecoTrackAsssociation.clone(
    label_tp = "trackingParticlesSignal",
    associator = "quickTrackAssociatorByHitsSignal"
)

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
(_selectorsFromPV, tracksValidationSelectorsFromPV) = _addSelectorsBySrc([_generalTracksHp], "FromPV", "generalTracksFromPV", globals())
tracksValidationSelectorsFromPV.insert(0, generalTracksFromPV)
(_selectorsFromPV_trackingPhase1, _tracksValidationSelectorsFromPV_trackingPhase1) = _addSelectorsBySrc([_generalTracksHp_trackingPhase1], "FromPV", "generalTracksFromPV", globals())
_tracksValidationSelectorsFromPV_trackingPhase1.insert(0, generalTracksFromPV)
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsFromPV, _tracksValidationSelectorsFromPV_trackingPhase1)

## Select conversion TrackingParticles, and define the corresponding associator
# (do not use associations because the collections of interest are not subsets of each other)
trackingParticlesConversion = _trackingParticleConversionSelector.clone()
tpClusterProducerConversion = tpClusterProducer.clone(
    trackingParticleSrc = "trackingParticlesConversion",
)
quickTrackAssociatorByHitsConversion = quickTrackAssociatorByHits.clone(
    cluster2TPSrc = "tpClusterProducerConversion"
)


## MTV instances
trackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone(
    label =  ["generalTracks", _generalTracksHp] + _selectorsByAlgo + _selectorsByAlgoHp +  [
    "cutsRecoTracksBtvLike",
    "cutsRecoTracksAK4PFJets"],
    useLogPt = cms.untracked.bool(True),
    dodEdxPlots = True,
    doPVAssociationPlots = True
    #,minpT = cms.double(-1)
    #,maxpT = cms.double(3)
    #,nintpT = cms.int32(40)
)
eras.fastSim.toModify(trackValidator, 
                      dodEdxPlots = False)
eras.trackingPhase1.toModify(trackValidator,
    label = ["generalTracks", _generalTracksHp_trackingPhase1] + _selectorsByAlgo_trackingPhase1 + _selectorsByAlgoHp_trackingPhase1 +  [
        "cutsRecoTracksBtvLike",
        "cutsRecoTracksAK4PFJets"
    ]
)

# For efficiency of signal TPs vs. signal tracks, and fake rate of
# signal tracks vs. signal TPs
trackValidatorFromPV = trackValidator.clone(
    dirName = "Tracking/TrackFromPV/",
    label = ["generalTracksFromPV"]+_selectorsFromPV,
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    associators = ["trackingParticleRecoTrackAsssociationSignal"],
    trackCollectionForDrCalculation = "generalTracksFromPV",
    doPlotsOnlyForTruePV = True,
    doPVAssociationPlots = False,
)
eras.trackingPhase1.toModify(trackValidatorFromPV, label = ["generalTracksFromPV"]+_selectorsFromPV_trackingPhase1)

# For fake rate of signal tracks vs. all TPs, and pileup rate of
# signal tracks vs. non-signal TPs
trackValidatorFromPVAllTP = trackValidatorFromPV.clone(
    dirName = "Tracking/TrackFromPVAllTP/",
    label_tp_effic = trackValidator.label_tp_effic.value(),
    label_tp_fake = trackValidator.label_tp_fake.value(),
    associators = trackValidator.associators.value(),
    doSimPlots = False,
    doSimTrackPlots = False,
)

# For efficiency of all TPs vs. all tracks
trackValidatorAllTPEffic = trackValidator.clone(
    dirName = "Tracking/TrackAllTPEffic/",
    label = [
        "generalTracks",
        _generalTracksHp,
    ],
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
eras.trackingPhase1.toModify(trackValidatorAllTPEffic, label = ["generalTracks", _generalTracksHp_trackingPhase1])

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
    label_tp_fake = "trackingParticlesConversion",
    associators = ["quickTrackAssociatorByHitsConversion"],
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
    VertexAssociatorByPositionAndTracks
)
eras.fastSim.toModify(tracksValidationTruth, lambda x: x.remove(tpClusterProducer))

tracksValidationTruthSignal = cms.Sequence(
    cms.ignore(trackingParticlesSignal) +
    tpClusterProducerSignal +
    quickTrackAssociatorByHitsSignal +
    trackingParticleRecoTrackAsssociationSignal
)
eras.fastSim.toModify(tracksValidationTruthSignal, lambda x: x.remove(tpClusterProducerSignal))


tracksValidationTruthConversion = cms.Sequence(
    trackingParticlesConversion +
    tpClusterProducerConversion +
    quickTrackAssociatorByHitsConversion
)

tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationSelectorsFromPV +
    tracksValidationTruth +
    tracksValidationTruthSignal +
    tracksValidationTruthConversion
)
eras.fastSim.toModify(tracksPreValidation, lambda x: x.remove(tracksValidationTruthConversion))

tracksValidation = cms.Sequence(
    tracksPreValidation +
    trackValidator +
    trackValidatorFromPV +
    trackValidatorFromPVAllTP +
    trackValidatorAllTPEffic +
    trackValidatorConversion
)
eras.fastSim.toModify(tracksValidation, lambda x: x.remove(trackValidatorConversion))


### Then define stuff for standalone mode (i.e. MTV with RECO+DIGI input)

# Select by originalAlgo and algoMask
_selectorsByAlgoAndHp = _selectorsByAlgo+_selectorsByAlgoHp
_selectorsByAlgoAndHp_trackingPhase1 = _selectorsByAlgo_trackingPhase1+_selectorsByAlgoHp_trackingPhase1
(_selectorsByOriginalAlgo, tracksValidationSelectorsByOriginalAlgoStandalone) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgoAndHp, "ByOriginalAlgo", "originalAlgorithm",globals())
(_selectorsByOriginalAlgo_trackingPhase1, _tracksValidationSelectorsByOriginalAlgoStandalone_trackingPhase1) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgoAndHp_trackingPhase1, "ByOriginalAlgo", "originalAlgorithm",globals())
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsByOriginalAlgoStandalone, _tracksValidationSelectorsByOriginalAlgoStandalone_trackingPhase1)

(_selectorsByAlgoMask, tracksValidationSelectorsByAlgoMaskStandalone) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgoAndHp, "ByAlgoMask", "algorithmMaskContains",globals())
(_selectorsByAlgoMask_trackingPhase1, _tracksValidationSelectorsByAlgoMaskStandalone_trackingPhase1) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgoAndHp_trackingPhase1, "ByAlgoMask", "algorithmMaskContains",globals())
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsByAlgoMaskStandalone, _tracksValidationSelectorsByAlgoMaskStandalone_trackingPhase1)

# Select fromPV by iteration
(_selectorsFromPVStandalone, tracksValidationSelectorsFromPVStandalone) = _addSelectorsBySrc(_selectorsByAlgoAndHp, "FromPV", "generalTracksFromPV",globals())
(_selectorsFromPVStandalone_trackingPhase1, _tracksValidationSelectorsFromPVStandalone_trackingPhase1) = _addSelectorsBySrc(_selectorsByAlgoAndHp_trackingPhase1, "FromPV", "generalTracksFromPV",globals())
eras.trackingPhase1.toReplaceWith(tracksValidationSelectorsFromPVStandalone, _tracksValidationSelectorsFromPVStandalone_trackingPhase1)

# MTV instances
trackValidatorStandalone = trackValidator.clone( label = trackValidator.label+ _selectorsByOriginalAlgo + _selectorsByAlgoMask)
eras.trackingPhase1.toModify(trackValidatorStandalone, label = trackValidator.label+ _selectorsByOriginalAlgo_trackingPhase1 + _selectorsByAlgoMask_trackingPhase1)

trackValidatorFromPVStandalone = trackValidatorFromPV.clone( label = trackValidatorFromPV.label+_selectorsFromPVStandalone)
eras.trackingPhase1.toModify(trackValidatorFromPVStandalone, label = trackValidatorFromPV.label+_selectorsFromPVStandalone_trackingPhase1)

trackValidatorFromPVAllTPStandalone = trackValidatorFromPVAllTP.clone(
    label = trackValidatorFromPVStandalone.label.value()
)
trackValidatorAllTPEfficStandalone = trackValidatorAllTPEffic.clone(
    label = [ x for x in trackValidator.label.value() if x not in ["cutsRecoTracksBtvLike", "cutsRecoTracksAK4PFJets"]]
)

trackValidatorConversionStandalone = trackValidatorConversion.clone( label = [x for x in trackValidatorConversion.label if x != "convStepTracks"])

# sequences
tracksValidationSelectorsStandalone = cms.Sequence(
    tracksValidationSelectorsByOriginalAlgoStandalone +
    tracksValidationSelectorsByAlgoMaskStandalone +
    tracksValidationSelectorsFromPVStandalone
)

# we copy this for both Standalone and TrackingOnly
#  and later make modifications from it which change based on era
_trackValidatorsBase = cms.Sequence(
    trackValidatorStandalone +
    trackValidatorFromPVStandalone +
    trackValidatorFromPVAllTPStandalone +
    trackValidatorAllTPEfficStandalone +
    trackValidatorConversionStandalone
)
trackValidatorsStandalone = _trackValidatorsBase.copy()
eras.fastSim.toModify(trackValidatorsStandalone, lambda x: x.remove(trackValidatorConversionStandalone) )

tracksValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain +
    tracksPreValidation +
    tracksValidationSelectorsStandalone +
    trackValidatorsStandalone
)

### TrackingOnly mode (i.e. MTV with DIGI input + tracking-only reconstruction)

# selectors
tracksValidationSelectorsTrackingOnly = tracksValidationSelectors.copyAndExclude([ak4JetTracksAssociatorExplicitAll,cutsRecoTracksAK4PFJets]) # selectors using track information only (i.e. no PF)
(_seedSelectors, tracksValidationSeedSelectorsTrackingOnly) = _addSeedToTrackProducers(_seedProducers, globals())
(_fastSimSeedSelectors, _fastSimTracksValidationSeedSelectorsTrackingOnly) = _addSeedToTrackProducers(_seedProducersForFastSim, globals())
(_trackingPhase1SeedSelectors, _trackingPhase1TracksValidationSeedSelectorsTrackingOnly) = _addSeedToTrackProducers(_seedProducersForTrackingPhase1, globals())
eras.fastSim.toReplaceWith(tracksValidationSeedSelectorsTrackingOnly, _fastSimTracksValidationSeedSelectorsTrackingOnly)
eras.trackingPhase1.toReplaceWith(tracksValidationSeedSelectorsTrackingOnly, _trackingPhase1TracksValidationSeedSelectorsTrackingOnly)

# MTV instances
trackValidatorTrackingOnly = trackValidatorStandalone.clone(label = [ x for x in trackValidatorStandalone.label if x != "cutsRecoTracksAK4PFJets"] )

trackValidatorBuildingTrackingOnly = trackValidatorTrackingOnly.clone(
    dirName = "Tracking/TrackBuilding/",
    associators = ["quickTrackAssociatorByHits"],
    UseAssociators = True,
    label = _trackProducers,
    dodEdxPlots = False,
    doPVAssociationPlots = False,
    doSimPlots = False,
)

eras.fastSim.toModify(trackValidatorBuildingTrackingOnly, label =  _trackProducersForFastSim )
eras.trackingPhase1.toModify(trackValidatorBuildingTrackingOnly, label = _trackProducersForTrackingPhase1)

trackValidatorSeedingTrackingOnly = trackValidatorBuildingTrackingOnly.clone(
    dirName = "Tracking/TrackSeeding/",
    label = _seedSelectors,
    doSeedPlots = True,
)
eras.fastSim.toModify(trackValidatorSeedingTrackingOnly, label= _fastSimSeedSelectors)
eras.trackingPhase1.toModify(trackValidatorSeedingTrackingOnly, label= _trackingPhase1SeedSelectors)


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
eras.fastSim.toModify(trackValidatorsTrackingOnly, lambda x: x.remove(trackValidatorConversionTrackingOnly))


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
