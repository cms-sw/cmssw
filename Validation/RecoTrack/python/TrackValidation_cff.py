import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoTrack.MultiTrackValidator_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

# Validation iterative steps
cutsRecoTracksFirst = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirst.algorithm=cms.string("ctf")

cutsRecoTracksSecond = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecond.algorithm=cms.string("iter2")

cutsRecoTracksThird = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThird.algorithm=cms.string("iter3")

cutsRecoTracksFourth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourth.algorithm=cms.string("iter4")

cutsRecoTracksFifth = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifth.algorithm=cms.string("iter5")

# high purity
cutsRecoTracksHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksHp.quality=cms.string("highPurity")

cutsRecoTracksFirstHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFirstHp.algorithm=cms.string("ctf")
cutsRecoTracksFirstHp.quality=cms.string("highPurity")

cutsRecoTracksSecondHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksSecondHp.algorithm=cms.string("iter2")
cutsRecoTracksSecondHp.quality=cms.string("highPurity")

cutsRecoTracksThirdHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksThirdHp.algorithm=cms.string("iter3")
cutsRecoTracksThirdHp.quality=cms.string("highPurity")

cutsRecoTracksFourthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFourthHp.algorithm=cms.string("iter4")
cutsRecoTracksFourthHp.quality=cms.string("highPurity")

cutsRecoTracksFifthHp = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRecoTracksFifthHp.algorithm=cms.string("iter5")
cutsRecoTracksFifthHp.quality=cms.string("highPurity")


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

