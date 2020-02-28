import FWCore.ParameterSet.Config as cms

# track associator settings
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
# to do the track<->TP association with TrackerHitAssociator
trackAssociatorByHitsRecoDenom = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHitsTrackerHitAssociator.clone()
from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cff import *

# reco track quality cuts
from Validation.RecoTrack.cuts_cff import *
cutsRecoTracks.src = "hiGeneralTracks"
cutsRecoTracks.ptMin = 2.0
cutsRecoTracks.quality = []

# high purity selection
cutsRecoTracksHP = cutsRecoTracks.clone( quality = cms.vstring("highPurity") )

# sim track quality cuts
from Validation.RecoHI.selectSimTracks_cff import *
findableSimTracks.ptMin = 2.0

# setup multi-track validator
from Validation.RecoTrack.MultiTrackValidator_cff import *
hiTrackValidator = multiTrackValidator.clone(
    associators = ["trackAssociatorByHitsRecoDenom"],
    UseAssociators = True,
    label_tp_effic = cms.InputTag("primaryChgSimTracks"),
    label_tp_fake  = cms.InputTag("cutsTPFake"),
    label_tp_effic_refvector = True,
    label_tp_fake_refvector = True,
    signalOnlyTP = cms.bool(False),
    trackCollectionForDrCalculation = cms.InputTag("cutsRecoTracks"),
    minpT = cms.double(1.0),
    maxpT = cms.double(100.0),
    nintpT = cms.int32(40),
    useLogPt = cms.untracked.bool(True),
    cores = cms.InputTag("")
    )

hiTrackValidator.label = cms.VInputTag(cms.InputTag('cutsRecoTracks'),
                                       cms.InputTag('cutsRecoTracksHP')
                                       )

# track prevalidation
hiTrackPrevalidation = cms.Sequence(
    primaryChgSimTracks
    * cutsTPFake
    * cutsRecoTracks
    * cutsRecoTracksHP
    * trackingParticleNumberOfLayersProducer
    )

# track validation sequence
hiTrackValidation = cms.Sequence( trackAssociatorByHitsRecoDenom * hiTrackValidator )
