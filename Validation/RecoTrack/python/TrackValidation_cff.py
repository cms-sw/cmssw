import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoTrack.MultiTrackValidator_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

# Validation iterative steps
cutsRecoTracksFirst = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirst.algorithm=cms.vstring("ctf")

cutsRecoTracksSecond = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecond.algorithm=cms.vstring("iter2")

cutsRecoTracksThird = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThird.algorithm=cms.vstring("iter3")

cutsRecoTracksFourth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourth.algorithm=cms.vstring("iter4")

cutsRecoTracksFifth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifth.algorithm=cms.vstring("iter5")

# high purity
cutsRecoTracksHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksFirstHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirstHp.algorithm=cms.vstring("ctf")
cutsRecoTracksFirstHp.quality=cms.vstring("highPurity")

cutsRecoTracksSecondHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecondHp.algorithm=cms.vstring("iter2")
cutsRecoTracksSecondHp.quality=cms.vstring("highPurity")

cutsRecoTracksThirdHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThirdHp.algorithm=cms.vstring("iter3")
cutsRecoTracksThirdHp.quality=cms.vstring("highPurity")

cutsRecoTracksFourthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourthHp.algorithm=cms.vstring("iter4")
cutsRecoTracksFourthHp.quality=cms.vstring("highPurity")

cutsRecoTracksFifthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifthHp.algorithm=cms.vstring("iter5")
cutsRecoTracksFifthHp.quality=cms.vstring("highPurity")


multiTrackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                        cms.InputTag("cutsRecoTracksHp"),
                                        cms.InputTag("cutsRecoTracksFirst"),
                                        cms.InputTag("cutsRecoTracksFirstHp"),
                                        cms.InputTag("cutsRecoTracksSecond"),
                                        cms.InputTag("cutsRecoTracksSecondHp"),
                                        cms.InputTag("cutsRecoTracksThird"),
                                        cms.InputTag("cutsRecoTracksThirdHp"),
                                        cms.InputTag("cutsRecoTracksFourth"),
                                        cms.InputTag("cutsRecoTracksFourthHp"),
                                        cms.InputTag("cutsRecoTracksFifth"),
                                        cms.InputTag("cutsRecoTracksFifthHp")
                                        )


tracksValidation = cms.Sequence(cutsRecoTracksHp*
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
                                multiTrackValidator)

