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

# Validation iterative steps
cutsRecoTracksInitialStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksInitialStep.algorithm=cms.vstring("initialStep")

cutsRecoTracksLowPtTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksLowPtTripletStep.algorithm=cms.vstring("lowPtTripletStep")

cutsRecoTracksPixelPairStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksPixelPairStep.algorithm=cms.vstring("pixelPairStep")

cutsRecoTracksDetachedTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksDetachedTripletStep.algorithm=cms.vstring("detachedTripletStep")

cutsRecoTracksMixedTripletStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMixedTripletStep.algorithm=cms.vstring("mixedTripletStep")

cutsRecoTracksPixelLessStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksPixelLessStep.algorithm=cms.vstring("pixelLessStep")

cutsRecoTracksTobTecStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksTobTecStep.algorithm=cms.vstring("tobTecStep")

cutsRecoTracksJetCoreRegionalStep = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksJetCoreRegionalStep.algorithm=cms.vstring("jetCoreRegionalStep")

cutsRecoTracksMuonSeededStepInOut = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMuonSeededStepInOut.algorithm=cms.vstring("muonSeededStepInOut")

cutsRecoTracksMuonSeededStepOutIn = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMuonSeededStepOutIn.algorithm=cms.vstring("muonSeededStepOutIn")

# high purity
cutsRecoTracksHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksInitialStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksInitialStepHp.algorithm=cms.vstring("initialStep")
cutsRecoTracksInitialStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksLowPtTripletStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksLowPtTripletStepHp.algorithm=cms.vstring("lowPtTripletStep")
cutsRecoTracksLowPtTripletStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksPixelPairStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksPixelPairStepHp.algorithm=cms.vstring("pixelPairStep")
cutsRecoTracksPixelPairStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksDetachedTripletStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksDetachedTripletStepHp.algorithm=cms.vstring("detachedTripletStep")
cutsRecoTracksDetachedTripletStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksMixedTripletStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMixedTripletStepHp.algorithm=cms.vstring("mixedTripletStep")
cutsRecoTracksMixedTripletStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksPixelLessStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksPixelLessStepHp.algorithm=cms.vstring("pixelLessStep")
cutsRecoTracksPixelLessStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksTobTecStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksTobTecStepHp.algorithm=cms.vstring("tobTecStep")
cutsRecoTracksTobTecStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksJetCoreRegionalStepHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksJetCoreRegionalStepHp.algorithm=cms.vstring("jetCoreRegionalStep")
cutsRecoTracksJetCoreRegionalStepHp.quality=cms.vstring("highPurity")

cutsRecoTracksMuonSeededStepInOutHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMuonSeededStepInOutHp.algorithm=cms.vstring("muonSeededStepInOut")
cutsRecoTracksMuonSeededStepInOutHp.quality=cms.vstring("highPurity")

cutsRecoTracksMuonSeededStepOutInHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksMuonSeededStepOutInHp.algorithm=cms.vstring("muonSeededStepOutIn")
cutsRecoTracksMuonSeededStepOutInHp.quality=cms.vstring("highPurity")

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
                                   )
trackValidator.skipHistoFit=cms.untracked.bool(True)
trackValidator.useLogPt=cms.untracked.bool(True)
#trackValidator.minpT = cms.double(-1)
#trackValidator.maxpT = cms.double(3)
#trackValidator.nintpT = cms.int32(40)

# the track selectors
tracksValidationSelectors = cms.Sequence( cutsRecoTracksHp*
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
                                cutsRecoTracksMuonSeededStepOutInHp )
tracksValidationTruth = cms.Sequence(
    tpClusterProducer +
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation
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

