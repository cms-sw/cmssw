import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi 
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
import Validation.RecoTrack.MultiTrackValidator_cfi
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import cutsRecoTracks_cfi

from SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi import *
from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
from PhysicsTools.RecoAlgos.trackingParticleSelector_cfi import trackingParticleSelector as _trackingParticleSelector
from CommonTools.RecoAlgos.sortedPrimaryVertices_cfi import sortedPrimaryVertices as _sortedPrimaryVertices
from CommonTools.RecoAlgos.recoChargedRefCandidateToTrackRefProducer_cfi import recoChargedRefCandidateToTrackRefProducer as _recoChargedRefCandidateToTrackRefProducer

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
def _algoToSelector(algo):
    sel = ""
    if algo != "generalTracks":
        sel = algo[0].upper()+algo[1:]
    return "cutsRecoTracks"+sel

def _addSelectorsByAlgo():
    names = []
    seq = cms.Sequence()
    for algo in _algos:
        if algo == "generalTracks":
            continue
        modName = _algoToSelector(algo)
        mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=[algo])
        globals()[modName] = mod
        names.append(modName)
        seq += mod
    return (names, seq)
def _addSelectorsByHp():
    seq = cms.Sequence()
    names = []
    for algo in _algos:
        modName = _algoToSelector(algo)
        modNameHp = modName+"Hp"
        if algo == "generalTracks":
            mod = cutsRecoTracks_cfi.cutsRecoTracks.clone(quality=["highPurity"])
        else:
            mod = globals()[modName].clone(quality=["highPurity"])
        globals()[modNameHp] = mod
        names.append(modNameHp)
        seq += mod
    return (names, seq)
def _addSelectorsBySrc(modules, midfix, src):
    seq = cms.Sequence()
    names = []
    for modName in modules:
        modNameNew = modName.replace("cutsRecoTracks", "cutsRecoTracks"+midfix)
        mod = globals()[modName].clone(src=src)
        globals()[modNameNew] = mod
        names.append(modNameNew)
        seq += mod
    return (names, seq)
def _addSelectorsByOriginalAlgoMask(modules, midfix, algoParam):
    seq = cms.Sequence()
    names = []
    for modName in modules:
        if modName[-2:] == "Hp":
            modNameNew = modName[:-2] + midfix + "Hp"
        else:
            modNameNew = modName + midfix
        mod = globals()[modName].clone()
        setattr(mod, algoParam, mod.algorithm.value())
        mod.algorithm = []
        globals()[modNameNew] = mod
        names.append(modNameNew)
        seq += mod
    return (names, seq)

# Validation iterative steps
(_selectorsByAlgo, tracksValidationSelectorsByAlgo) = _addSelectorsByAlgo()

# high purity
(_selectorsByAlgoHp, tracksValidationSelectorsByAlgoHp) = _addSelectorsByHp()
_generalTracksHp = _selectorsByAlgoHp[0]
_selectorsByAlgoHp = _selectorsByAlgoHp[1:]

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
(_selectorsFromPV, tracksValidationSelectorsFromPV) = _addSelectorsBySrc([_generalTracksHp], "FromPV", "generalTracksFromPV")
tracksValidationSelectorsFromPV.insert(0, generalTracksFromPV)


## MTV instances
trackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()
trackValidator.label = ["generalTracks", _generalTracksHp] + _selectorsByAlgo + _selectorsByAlgoHp +  [
    "cutsRecoTracksBtvLike",
    "cutsRecoTracksAK4PFJets",
]
trackValidator.useLogPt = cms.untracked.bool(True)
trackValidator.dodEdxPlots = True
trackValidator.doPVAssociationPlots = True
#trackValidator.minpT = cms.double(-1)
#trackValidator.maxpT = cms.double(3)
#trackValidator.nintpT = cms.int32(40)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    trackValidator.dodEdxPlots = False


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

tracksValidationTruthSignal = cms.Sequence(
    cms.ignore(trackingParticlesSignal) +
    tpClusterProducerSignal +
    quickTrackAssociatorByHitsSignal +
    trackingParticleRecoTrackAsssociationSignal
)

if eras.fastSim.isChosen():
    tracksValidationTruth.remove(tpClusterProducer)
    tracksValidationTruthSignal.remove(tpClusterProducerSignal)


tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationSelectorsFromPV +
    tracksValidationTruth +
    tracksValidationTruthSignal
)

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence(
    trackValidator +
    trackValidatorFromPV +
    trackValidatorFromPVAllTP +
    trackValidatorAllTPEffic
)


### Then define stuff for standalone mode (i.e. MTV with RECO+DIGI input)

# Select by originalAlgo and algoMask
(_selectorsByOriginalAlgo, tracksValidationSelectorsByOriginalAlgoStandalone) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgo+_selectorsByAlgoHp, "ByOriginalAlgo", "originalAlgorithm")
(_selectorsByAlgoMask, tracksValidationSelectorsByAlgoMaskStandalone) = _addSelectorsByOriginalAlgoMask(_selectorsByAlgo+_selectorsByAlgoHp, "ByAlgoMask", "algorithmMaskContains")

# Select fromPV by iteration
(_selectorsFromPVStandalone, tracksValidationSelectorsFromPVStandalone) = _addSelectorsBySrc(_selectorsByAlgo+_selectorsByAlgoHp, "FromPV", "generalTracksFromPV")

# MTV instances
trackValidatorStandalone = trackValidator.clone()
trackValidatorStandalone.label.extend(_selectorsByOriginalAlgo + _selectorsByAlgoMask)
trackValidatorFromPVStandalone = trackValidatorFromPV.clone()
trackValidatorFromPVStandalone.label.extend(_selectorsFromPVStandalone)
trackValidatorFromPVAllTPStandalone = trackValidatorFromPVAllTP.clone(
    label = trackValidatorFromPVStandalone.label.value()
)
trackValidatorAllTPEfficStandalone = trackValidatorAllTPEffic.clone(
    label = trackValidator.label.value()
)
for _label in ["cutsRecoTracksBtvLike", "cutsRecoTracksAK4PFJets"]:
    trackValidatorAllTPEfficStandalone.label.remove(_label)

# sequences
tracksValidationSelectorsStandalone = cms.Sequence(
    tracksValidationSelectorsByOriginalAlgoStandalone +
    tracksValidationSelectorsByAlgoMaskStandalone +
    tracksValidationSelectorsFromPVStandalone
)
tracksPreValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain+
    tracksPreValidation +
    tracksValidationSelectorsStandalone
)
trackValidatorsStandalone = cms.Sequence(
    trackValidatorStandalone +
    trackValidatorFromPVStandalone +
    trackValidatorFromPVAllTPStandalone +
    trackValidatorAllTPEfficStandalone
)
tracksValidationStandalone = cms.Sequence(
    tracksPreValidationStandalone+
    trackValidatorsStandalone
)

### TrackingOnly mode (i.e. MTV with DIGI input + tracking-only reconstruction

# selectors
tracksValidationSelectorsTrackingOnly = tracksValidationSelectors.copyAndExclude([ak4JetTracksAssociatorExplicitAll,cutsRecoTracksAK4PFJets]) # selectors using track information only (i.e. no PF)

# MTV instances
trackValidatorTrackingOnly = trackValidatorStandalone.clone()
trackValidatorTrackingOnly.label.remove("cutsRecoTracksAK4PFJets")

# sequences
tracksPreValidationTrackingOnly = tracksPreValidation.copy()
tracksPreValidationTrackingOnly.replace(tracksValidationSelectors, tracksValidationSelectorsTrackingOnly)
tracksPreValidationTrackingOnly += tracksValidationSelectorsStandalone

trackValidatorsTrackingOnly = trackValidatorsStandalone.copy()
trackValidatorsTrackingOnly.replace(trackValidatorStandalone, trackValidatorTrackingOnly)

tracksValidationTrackingOnly = cms.Sequence(
    trackValidatorsTrackingOnly
)


### 'slim' sequences that only depend on track and tracking particle collections
tracksValidationSelectorsSlim = tracksValidationSelectorsTrackingOnly.copyAndExclude([cutsRecoTracksBtvLike])

tracksPreValidationSlim = cms.Sequence(
    tracksValidationSelectorsSlim +
    tracksValidationTruth
)

trackValidatorSlim = trackValidator.clone(
    doPVAssociationPlots = cms.untracked.bool(False),
    dodEdxPlots = False
)
for _label in ["cutsRecoTracksBtvLike", "cutsRecoTracksAK4PFJets"]:
    trackValidatorSlim.label.remove(_label)

tracksValidationSlim = cms.Sequence(
    tracksPreValidationSlim+
    trackValidatorSlim
)
