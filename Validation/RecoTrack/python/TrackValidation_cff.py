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
tracksValidationTruth = cms.Sequence(
    tpClusterProducer +
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation +
    VertexAssociatorByPositionAndTracks
)
tracksValidationTruthFS = cms.Sequence(
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation
)

tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationTruth
)
tracksPreValidationFS = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationTruthFS
)

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( trackValidator)
tracksValidationFS = cms.Sequence( trackValidator )

tracksValidationStandalone = cms.Sequence(
    ak4PFL1FastL2L3CorrectorChain+
    tracksPreValidation+
    tracksValidation
)
