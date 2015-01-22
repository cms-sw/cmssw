import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi 
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi 
import Validation.RecoTrack.MultiTrackValidator_cfi
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

from SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi import *

trackAssociatorByHitsRecoDenom= SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone(
    )

# Validation iterative steps
cutsRecoTracksZero = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksZero.algorithm=cms.vstring("initialStep")

cutsRecoTracksFirst = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirst.algorithm=cms.vstring("lowPtTripletStep")

cutsRecoTracksSecond = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecond.algorithm=cms.vstring("pixelPairStep")

cutsRecoTracksThird = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThird.algorithm=cms.vstring("detachedTripletStep")

cutsRecoTracksFourth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourth.algorithm=cms.vstring("mixedTripletStep")

cutsRecoTracksFifth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifth.algorithm=cms.vstring("pixelLessStep")

cutsRecoTracksSixth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSixth.algorithm=cms.vstring("tobTecStep")

cutsRecoTracksSeventh = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSeventh.algorithm=cms.vstring("jetCoreRegionalStep")

cutsRecoTracksNinth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksNinth.algorithm=cms.vstring("muonSeededStepInOut")

cutsRecoTracksTenth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksTenth.algorithm=cms.vstring("muonSeededStepOutIn")

# high purity
cutsRecoTracksHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksZeroHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksZeroHp.algorithm=cms.vstring("initialStep")
cutsRecoTracksZeroHp.quality=cms.vstring("highPurity")

cutsRecoTracksFirstHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirstHp.algorithm=cms.vstring("lowPtTripletStep")
cutsRecoTracksFirstHp.quality=cms.vstring("highPurity")

cutsRecoTracksSecondHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecondHp.algorithm=cms.vstring("pixelPairStep")
cutsRecoTracksSecondHp.quality=cms.vstring("highPurity")

cutsRecoTracksThirdHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThirdHp.algorithm=cms.vstring("detachedTripletStep")
cutsRecoTracksThirdHp.quality=cms.vstring("highPurity")

cutsRecoTracksFourthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourthHp.algorithm=cms.vstring("mixedTripletStep")
cutsRecoTracksFourthHp.quality=cms.vstring("highPurity")

cutsRecoTracksFifthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifthHp.algorithm=cms.vstring("pixelLessStep")
cutsRecoTracksFifthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSixthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSixthHp.algorithm=cms.vstring("tobTecStep")
cutsRecoTracksSixthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSeventhHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSeventhHp.algorithm=cms.vstring("jetCoreRegionalStep")
cutsRecoTracksSeventhHp.quality=cms.vstring("highPurity")

cutsRecoTracksNinthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksNinthHp.algorithm=cms.vstring("muonSeededStepInOut")
cutsRecoTracksNinthHp.quality=cms.vstring("highPurity")

cutsRecoTracksTenthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
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

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( tpClusterProducer * trackAssociatorByHitsRecoDenom * trackValidator)
tracksValidationFS = cms.Sequence( trackAssociatorByHitsRecoDenom * trackValidator )

