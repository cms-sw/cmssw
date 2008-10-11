import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.TrackingParticleSelectionForMuonEfficiency_cfi import *
multiL3TrackValidator = cms.EDFilter("MultiTrackValidator",
    TrackingParticleSelectionForMuonEfficiency,
    useFabsEta = cms.bool(True),
    minpT = cms.double(0.0),
    nintHit = cms.int32(35),
    associatormap = cms.InputTag("tpToL3TrackAssociation"),
    label_tp_fake = cms.InputTag("mergedtruth","MergedTrackTruth"),
    out = cms.string('validationPlots.root'),
    min = cms.double(0.0),
    nintpT = cms.int32(200),
    label = cms.VInputTag(cms.InputTag("hltL3Muons")),
    maxHit = cms.double(35.0),
    label_tp_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    useInvPt = cms.bool(False),
    dirName = cms.string('RecoMuonV/MultiTrack/'),
    minHit = cms.double(0.0),
    sim = cms.string('g4SimHits'),
    associators = cms.vstring('TrackAssociatorByDeltaR1'),
    max = cms.double(2.5),
    maxpT = cms.double(1100.0),
    nint = cms.int32(25),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    UseAssociators = cms.bool(False)
)


