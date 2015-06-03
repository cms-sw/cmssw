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

## Track selectors
# Validation iterative steps
cutsRecoTracksInitialStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["initialStep"])
cutsRecoTracksLowPtTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["lowPtTripletStep"])
cutsRecoTracksPixelPairStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["pixelPairStep"])
cutsRecoTracksDetachedTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["detachedTripletStep"])
cutsRecoTracksMixedTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["mixedTripletStep"])
cutsRecoTracksPixelLessStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["pixelLessStep"])
cutsRecoTracksTobTecStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["tobTecStep"])
cutsRecoTracksJetCoreRegionalStep = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["jetCoreRegionalStep"])
cutsRecoTracksMuonSeededStepInOut = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["muonSeededStepInOut"])
cutsRecoTracksMuonSeededStepOutIn = cutsRecoTracks_cfi.cutsRecoTracks.clone(algorithm=["muonSeededStepOutIn"])

# high purity
cutsRecoTracksHp = cutsRecoTracks_cfi.cutsRecoTracks.clone(quality=["highPurity"])
cutsRecoTracksInitialStepHp = cutsRecoTracksInitialStep.clone(quality=["highPurity"])
cutsRecoTracksLowPtTripletStepHp = cutsRecoTracksLowPtTripletStep.clone(quality=["highPurity"])
cutsRecoTracksPixelPairStepHp = cutsRecoTracksPixelPairStep.clone(quality=["highPurity"])
cutsRecoTracksDetachedTripletStepHp = cutsRecoTracksDetachedTripletStep.clone(quality=["highPurity"])
cutsRecoTracksMixedTripletStepHp = cutsRecoTracksMixedTripletStep.clone(quality=["highPurity"])
cutsRecoTracksPixelLessStepHp = cutsRecoTracksPixelLessStep.clone(quality=["highPurity"])
cutsRecoTracksTobTecStepHp = cutsRecoTracksTobTecStep.clone(quality=["highPurity"])
cutsRecoTracksJetCoreRegionalStepHp = cutsRecoTracksJetCoreRegionalStep.clone(quality=["highPurity"])
cutsRecoTracksMuonSeededStepInOutHp = cutsRecoTracksMuonSeededStepInOut.clone(quality=["highPurity"])
cutsRecoTracksMuonSeededStepOutInHp = cutsRecoTracksMuonSeededStepOutIn.clone(quality=["highPurity"])

# BTV-like selection
import PhysicsTools.RecoAlgos.btvTracks_cfi as btvTracks_cfi
cutsRecoTracksBtvLike = btvTracks_cfi.btvTrackRefs.clone()

# Select tracks associated to AK4 jets
import RecoJets.JetAssociationProducers.ak4JTA_cff as ak4JTA_cff
ak4JetTracksAssociatorAtVertexPFAll = ak4JTA_cff.ak4JetTracksAssociatorAtVertexPF.clone(
    jets = "ak4PFJets"
)
from JetMETCorrections.Configuration.JetCorrectors_cff import *
import CommonTools.RecoAlgos.jetTracksAssociationToTrackRefs_cfi as jetTracksAssociationToTrackRefs_cfi
cutsRecoTracksAK4PFJets = jetTracksAssociationToTrackRefs_cfi.jetTracksAssociationToTrackRefs.clone(
    association = "ak4JetTracksAssociatorAtVertexPFAll",
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
# first convert to RecoChargedRefCandidates
trackRefsForValidation = cms.EDProducer("ChargedRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("generalTracks")
)
# then use the "PV sorting" module to select the candidates associated to PV
trackRefsFromPV = _sortedPrimaryVertices.clone(
    particles = "trackRefsForValidation",
    produceAssociationToOriginalVertices = True,
    produceNoPileUpCollection = True,
    produceSortedVertices = False,
    jets = "ak4CaloJets",
    vertices = "offlinePrimaryVertices"
)
# and finally extract tracks from there
generalTracksFromPV = _recoChargedRefCandidateToTrackRefProducer.clone(
    src = cms.InputTag("trackRefsFromPV", "originalNoPileUp")
)
# and then the selectors
cutsRecoTracksFromPVInitialStep         = cutsRecoTracksInitialStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVLowPtTripletStep    = cutsRecoTracksLowPtTripletStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVPixelPairStep       = cutsRecoTracksPixelPairStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVDetachedTripletStep = cutsRecoTracksDetachedTripletStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMixedTripletStep    = cutsRecoTracksMixedTripletStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVPixelLessStep       = cutsRecoTracksPixelLessStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVTobTecStep          = cutsRecoTracksTobTecStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVJetCoreRegionalStep = cutsRecoTracksJetCoreRegionalStep.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMuonSeededStepInOut = cutsRecoTracksMuonSeededStepInOut.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMuonSeededStepOutIn = cutsRecoTracksMuonSeededStepOutIn.clone(src="generalTracksFromPV")
# high purity
cutsRecoTracksFromPVHp                    = cutsRecoTracksHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVInitialStepHp         = cutsRecoTracksInitialStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVLowPtTripletStepHp    = cutsRecoTracksLowPtTripletStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVPixelPairStepHp       = cutsRecoTracksPixelPairStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVDetachedTripletStepHp = cutsRecoTracksDetachedTripletStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMixedTripletStepHp    = cutsRecoTracksMixedTripletStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVPixelLessStepHp       = cutsRecoTracksPixelLessStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVTobTecStepHp          = cutsRecoTracksTobTecStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVJetCoreRegionalStepHp = cutsRecoTracksJetCoreRegionalStepHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMuonSeededStepInOutHp = cutsRecoTracksMuonSeededStepInOutHp.clone(src="generalTracksFromPV")
cutsRecoTracksFromPVMuonSeededStepOutInHp = cutsRecoTracksMuonSeededStepOutInHp.clone(src="generalTracksFromPV")


## MTV instances
trackValidator= Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()

trackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                   cms.InputTag("cutsRecoTracksHp"),
                                   cms.InputTag("cutsRecoTracksInitialStep"),
                                   cms.InputTag("cutsRecoTracksInitialStepHp"),
                                   cms.InputTag("cutsRecoTracksLowPtTripletStep"),
                                   cms.InputTag("cutsRecoTracksLowPtTripletStepHp"),
                                   cms.InputTag("cutsRecoTracksPixelPairStep"),
                                   cms.InputTag("cutsRecoTracksPixelPairStepHp"),
                                   cms.InputTag("cutsRecoTracksDetachedTripletStep"),
                                   cms.InputTag("cutsRecoTracksDetachedTripletStepHp"),
                                   cms.InputTag("cutsRecoTracksMixedTripletStep"),
                                   cms.InputTag("cutsRecoTracksMixedTripletStepHp"),
                                   cms.InputTag("cutsRecoTracksPixelLessStep"),
                                   cms.InputTag("cutsRecoTracksPixelLessStepHp"),
                                   cms.InputTag("cutsRecoTracksTobTecStep"),
                                   cms.InputTag("cutsRecoTracksTobTecStepHp"),
                                   cms.InputTag("cutsRecoTracksJetCoreRegionalStep"),
                                   cms.InputTag("cutsRecoTracksJetCoreRegionalStepHp"),
                                   cms.InputTag("cutsRecoTracksMuonSeededStepInOut"),
                                   cms.InputTag("cutsRecoTracksMuonSeededStepInOutHp"),
                                   cms.InputTag("cutsRecoTracksMuonSeededStepOutIn"),
                                   cms.InputTag("cutsRecoTracksMuonSeededStepOutInHp"),
                                   cms.InputTag("cutsRecoTracksBtvLike"),
                                   cms.InputTag("cutsRecoTracksAK4PFJets"),
                                   )
trackValidator.useLogPt=cms.untracked.bool(True)
trackValidator.dodEdxPlots = True
#trackValidator.minpT = cms.double(-1)
#trackValidator.maxpT = cms.double(3)
#trackValidator.nintpT = cms.int32(40)

trackValidatorFromPV = trackValidator.clone(
    dirName = "Tracking/TrackFromPV/",
    label = [
        "generalTracksFromPV",
        "cutsRecoTracksFromPVHp",
        "cutsRecoTracksFromPVInitialStep",
        "cutsRecoTracksFromPVInitialStepHp",
        "cutsRecoTracksFromPVLowPtTripletStep",
        "cutsRecoTracksFromPVLowPtTripletStepHp",
        "cutsRecoTracksFromPVPixelPairStep",
        "cutsRecoTracksFromPVPixelPairStepHp",
        "cutsRecoTracksFromPVDetachedTripletStep",
        "cutsRecoTracksFromPVDetachedTripletStepHp",
        "cutsRecoTracksFromPVMixedTripletStep",
        "cutsRecoTracksFromPVMixedTripletStepHp",
        "cutsRecoTracksFromPVPixelLessStep",
        "cutsRecoTracksFromPVPixelLessStepHp",
        "cutsRecoTracksFromPVTobTecStep",
        "cutsRecoTracksFromPVTobTecStepHp",
        "cutsRecoTracksFromPVJetCoreRegionalStep",
        "cutsRecoTracksFromPVJetCoreRegionalStepHp",
        "cutsRecoTracksFromPVMuonSeededStepInOut",
        "cutsRecoTracksFromPVMuonSeededStepInOutHp",
        "cutsRecoTracksFromPVMuonSeededStepOutIn",
        "cutsRecoTracksFromPVMuonSeededStepOutInHp",
    ],
    label_tp_effic = "trackingParticlesSignal",
    label_tp_fake = "trackingParticlesSignal",
    associators = ["trackingParticleRecoTrackAsssociationSignal"],
    trackCollectionForDrCalculation = "generalTracksFromPV"
)

# the track selectors
tracksValidationSelectors = cms.Sequence(
    cutsRecoTracksHp*
    cutsRecoTracksInitialStep*
    cutsRecoTracksInitialStepHp*
    cutsRecoTracksLowPtTripletStep*
    cutsRecoTracksLowPtTripletStepHp*
    cutsRecoTracksPixelPairStep*
    cutsRecoTracksPixelPairStepHp*
    cutsRecoTracksDetachedTripletStep*
    cutsRecoTracksDetachedTripletStepHp*
    cutsRecoTracksMixedTripletStep*
    cutsRecoTracksMixedTripletStepHp*
    cutsRecoTracksPixelLessStep*
    cutsRecoTracksPixelLessStepHp*
    cutsRecoTracksTobTecStep*
    cutsRecoTracksTobTecStepHp*
    cutsRecoTracksJetCoreRegionalStep*
    cutsRecoTracksJetCoreRegionalStepHp*
    cutsRecoTracksMuonSeededStepInOut*
    cutsRecoTracksMuonSeededStepInOutHp*
    cutsRecoTracksMuonSeededStepOutIn*
    cutsRecoTracksMuonSeededStepOutInHp*
    cutsRecoTracksBtvLike*
    ak4JetTracksAssociatorAtVertexPFAll*
    cutsRecoTracksAK4PFJets
)
tracksValidationSelectorsFromPV = cms.Sequence(
    trackRefsForValidation*
    trackRefsFromPV*
    generalTracksFromPV*
    cutsRecoTracksFromPVHp*
    cutsRecoTracksFromPVInitialStep*
    cutsRecoTracksFromPVInitialStepHp*
    cutsRecoTracksFromPVLowPtTripletStep*
    cutsRecoTracksFromPVLowPtTripletStepHp*
    cutsRecoTracksFromPVPixelPairStep*
    cutsRecoTracksFromPVPixelPairStepHp*
    cutsRecoTracksFromPVDetachedTripletStep*
    cutsRecoTracksFromPVDetachedTripletStepHp*
    cutsRecoTracksFromPVMixedTripletStep*
    cutsRecoTracksFromPVMixedTripletStepHp*
    cutsRecoTracksFromPVPixelLessStep*
    cutsRecoTracksFromPVPixelLessStepHp*
    cutsRecoTracksFromPVTobTecStep*
    cutsRecoTracksFromPVTobTecStepHp*
    cutsRecoTracksFromPVJetCoreRegionalStep*
    cutsRecoTracksFromPVJetCoreRegionalStepHp*
    cutsRecoTracksFromPVMuonSeededStepInOut*
    cutsRecoTracksFromPVMuonSeededStepInOutHp*
    cutsRecoTracksFromPVMuonSeededStepOutIn*
    cutsRecoTracksFromPVMuonSeededStepOutInHp
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
tracksValidationTruthFS = cms.Sequence(
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation
)

tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationSelectorsFromPV +
    tracksValidationTruth +
    tracksValidationTruthSignal
)
tracksPreValidationFS = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationTruthFS
)

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( trackValidator + trackValidatorFromPV )
tracksValidationFS = cms.Sequence( trackValidator )

tracksValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain+
    tracksPreValidation+
    tracksValidation
)
