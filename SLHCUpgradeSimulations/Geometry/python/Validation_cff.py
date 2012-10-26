import FWCore.ParameterSet.Config as cms
# Preparing a cff to keep all of our MTV settings in one place

from Validation.RecoTrack.cutsTPEffic_cfi import *
from Validation.RecoTrack.cutsTPFake_cfi import *

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi import *


quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

from Configuration.StandardSequences.Validation_cff import *

#cutsRecoTracksHpUpg = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
#cutsRecoTracksHpUpg.quality=cms.vstring("highPurity")
#cutsRecoTracksHpUpg.ptMin = cms.double(0.9)

#cutsRecoTracksZeroHpUpg = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
#cutsRecoTracksZeroHpUpg.algorithm=cms.vstring("iter0")
#cutsRecoTracksZeroHpUpg.quality=cms.vstring("highPurity")
#cutsRecoTracksZeroHpUpg.ptMin = cms.double(0.9)

trackValidator.label=cms.VInputTag(	cms.InputTag("generalTracks")#,
                                  #	cms.InputTag("cutsRecoTracksHp"),
                                  #      cms.InputTag("cutsRecoTracksHpwbtagc"),
                                  #      cms.InputTag("cutsRecoTracksHpUpg"),
                                  #      cms.InputTag("cutsRecoTracksZeroHpUpg"),
                                  #      cms.InputTag("cutsRecoTracksFirstHpUpg"),
                                  #      cms.InputTag("cutsRecoTracksSecondHpUpg"),
                                  #      cms.InputTag("cutsRecoTracksThirdHpUpg"),
                                  #      cms.InputTag("cutsRecoTracksFourthHpUpg")
                                        )
trackValidator.associators = cms.vstring('quickTrackAssociatorByHits')
trackValidator.UseAssociators = True
trackValidator.histoProducerAlgoBlock.nintEta = cms.int32(20)
trackValidator.histoProducerAlgoBlock.nintPt = cms.int32(100)
trackValidator.histoProducerAlgoBlock.maxPt = cms.double(200.0)
trackValidator.histoProducerAlgoBlock.useLogPt = cms.untracked.bool(True)
trackValidator.histoProducerAlgoBlock.minDxy = cms.double(-3.0)
trackValidator.histoProducerAlgoBlock.maxDxy = cms.double(3.0)
trackValidator.histoProducerAlgoBlock.nintDxy = cms.int32(100)
trackValidator.histoProducerAlgoBlock.minDz = cms.double(-10.0)
trackValidator.histoProducerAlgoBlock.maxDz = cms.double(10.0)
trackValidator.histoProducerAlgoBlock.nintDz = cms.int32(100)
trackValidator.histoProducerAlgoBlock.maxVertpos = cms.double(5.0)
trackValidator.histoProducerAlgoBlock.nintVertpos = cms.int32(100)
trackValidator.histoProducerAlgoBlock.minZpos = cms.double(-10.0)
trackValidator.histoProducerAlgoBlock.maxZpos = cms.double(10.0)
trackValidator.histoProducerAlgoBlock.nintZpos = cms.int32(100)
trackValidator.histoProducerAlgoBlock.phiRes_rangeMin = cms.double(-0.003)
trackValidator.histoProducerAlgoBlock.phiRes_rangeMax = cms.double(0.003)
trackValidator.histoProducerAlgoBlock.phiRes_nbin = cms.int32(100)
trackValidator.histoProducerAlgoBlock.cotThetaRes_rangeMin = cms.double(-0.01)
trackValidator.histoProducerAlgoBlock.cotThetaRes_rangeMax = cms.double(+0.01)
trackValidator.histoProducerAlgoBlock.cotThetaRes_nbin = cms.int32(120)
trackValidator.histoProducerAlgoBlock.dxyRes_rangeMin = cms.double(-0.01)
trackValidator.histoProducerAlgoBlock.dxyRes_rangeMax = cms.double(0.01)
trackValidator.histoProducerAlgoBlock.dxyRes_nbin = cms.int32(100)
trackValidator.tipTP = cms.double(3.5)
trackValidator.ptMinTP = cms.double(0.9)

slhcTracksValidation = cms.Sequence(cutsRecoTracksHp*
                                 #cutsRecoTracksHpwbtagc*
                                 #cutsRecoTracksHpUpg*
                                 #cutsRecoTracksZeroHpUpg*
                                 #cutsRecoTracksFirstHpUpg*
                                 #cutsRecoTracksSecondHpUpg*
                                 #cutsRecoTracksThirdHpUpg*
                                 #cutsRecoTracksFourthHpUpg*
                                 trackValidator)


