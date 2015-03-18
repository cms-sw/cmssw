import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi 
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
import Validation.RecoTrack.MultiTrackValidator_cfi
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import cutsRecoTracks_cfi

from SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi import *

# Validation iterative steps
cutsRecoTracksZero = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksZero.algorithm=cms.vstring("initialStep")

cutsRecoTracksFirst = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFirst.algorithm=cms.vstring("lowPtTripletStep")

cutsRecoTracksSecond = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSecond.algorithm=cms.vstring("pixelPairStep")

cutsRecoTracksThird = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksThird.algorithm=cms.vstring("detachedTripletStep")

cutsRecoTracksFourth = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFourth.algorithm=cms.vstring("mixedTripletStep")

cutsRecoTracksFifth = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFifth.algorithm=cms.vstring("pixelLessStep")

cutsRecoTracksSixth = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSixth.algorithm=cms.vstring("tobTecStep")

cutsRecoTracksSeventh = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSeventh.algorithm=cms.vstring("jetCoreRegionalStep")

cutsRecoTracksNinth = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksNinth.algorithm=cms.vstring("muonSeededStepInOut")

cutsRecoTracksTenth = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksTenth.algorithm=cms.vstring("muonSeededStepOutIn")

# high purity
cutsRecoTracksHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksZeroHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksZeroHp.algorithm=cms.vstring("initialStep")
cutsRecoTracksZeroHp.quality=cms.vstring("highPurity")

cutsRecoTracksFirstHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFirstHp.algorithm=cms.vstring("lowPtTripletStep")
cutsRecoTracksFirstHp.quality=cms.vstring("highPurity")

cutsRecoTracksSecondHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSecondHp.algorithm=cms.vstring("pixelPairStep")
cutsRecoTracksSecondHp.quality=cms.vstring("highPurity")

cutsRecoTracksThirdHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksThirdHp.algorithm=cms.vstring("detachedTripletStep")
cutsRecoTracksThirdHp.quality=cms.vstring("highPurity")

cutsRecoTracksFourthHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFourthHp.algorithm=cms.vstring("mixedTripletStep")
cutsRecoTracksFourthHp.quality=cms.vstring("highPurity")

cutsRecoTracksFifthHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksFifthHp.algorithm=cms.vstring("pixelLessStep")
cutsRecoTracksFifthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSixthHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSixthHp.algorithm=cms.vstring("tobTecStep")
cutsRecoTracksSixthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSeventhHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksSeventhHp.algorithm=cms.vstring("jetCoreRegionalStep")
cutsRecoTracksSeventhHp.quality=cms.vstring("highPurity")

cutsRecoTracksNinthHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksNinthHp.algorithm=cms.vstring("muonSeededStepInOut")
cutsRecoTracksNinthHp.quality=cms.vstring("highPurity")

cutsRecoTracksTenthHp = cutsRecoTracks_cfi.cutsRecoTracks.clone()
cutsRecoTracksTenthHp.algorithm=cms.vstring("muonSeededStepOutIn")
cutsRecoTracksTenthHp.quality=cms.vstring("highPurity")

trackValidator= Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()

trackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                   cms.InputTag("cutsRecoTracksHp"),
                                   cms.InputTag("cutsRecoTracksZero"),
                                   cms.InputTag("cutsRecoTracksZeroHp"),
                                   cms.InputTag("cutsRecoTracksFirst"),
                                   cms.InputTag("cutsRecoTracksFirstHp"),
                                   cms.InputTag("cutsRecoTracksSecond"),
                                   cms.InputTag("cutsRecoTracksSecondHp"),
                                   cms.InputTag("cutsRecoTracksThird"),
                                   cms.InputTag("cutsRecoTracksThirdHp"),
                                   cms.InputTag("cutsRecoTracksFourth"),
                                   cms.InputTag("cutsRecoTracksFourthHp"),
                                   cms.InputTag("cutsRecoTracksFifth"),
                                   cms.InputTag("cutsRecoTracksFifthHp"),
                                   cms.InputTag("cutsRecoTracksSixth"),
                                   cms.InputTag("cutsRecoTracksSixthHp"),
                                   cms.InputTag("cutsRecoTracksSeventh"),
                                   cms.InputTag("cutsRecoTracksSeventhHp"),
                                   cms.InputTag("cutsRecoTracksNinth"),
                                   cms.InputTag("cutsRecoTracksNinthHp"),
                                   cms.InputTag("cutsRecoTracksTenth"),
                                   cms.InputTag("cutsRecoTracksTenthHp"),
                                   )
trackValidator.skipHistoFit=cms.untracked.bool(True)
trackValidator.useLogPt=cms.untracked.bool(True)
#trackValidator.minpT = cms.double(-1)
#trackValidator.maxpT = cms.double(3)
#trackValidator.nintpT = cms.int32(40)

# the track selectors
tracksValidationSelectors = cms.Sequence( cutsRecoTracksHp*
                                cutsRecoTracksZero*
                                cutsRecoTracksZeroHp*
                                cutsRecoTracksFirst*
                                cutsRecoTracksFirstHp*
                                cutsRecoTracksSecond*
                                cutsRecoTracksSecondHp*
                                cutsRecoTracksThird*
                                cutsRecoTracksThirdHp*
                                cutsRecoTracksFourth*
                                cutsRecoTracksFourthHp*
                                cutsRecoTracksFifth*
                                cutsRecoTracksFifthHp*
                                cutsRecoTracksSixth*
                                cutsRecoTracksSixthHp* 
                                cutsRecoTracksSeventh*
                                cutsRecoTracksSeventhHp* 
                                cutsRecoTracksNinth*
                                cutsRecoTracksNinthHp* 
                                cutsRecoTracksTenth*
                                cutsRecoTracksTenthHp )
tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tpClusterProducer +
    quickTrackAssociatorByHits
)
tracksPreValidationFS = cms.Sequence(
    tracksValidationSelectors +
    quickTrackAssociatorByHits
)

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( trackValidator)
tracksValidationFS = cms.Sequence( trackValidator )

