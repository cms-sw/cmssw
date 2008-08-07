import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.TrackingParticleSelectionForMuonEfficiency_cfi import *
multiGlbTrackValidator = cms.EDFilter("MultiTrackValidator",
    TrackingParticleSelectionForMuonEfficiency,
    useFabsEta = cms.bool(False),
    minpT = cms.double(0.0),
    nintHit = cms.int32(75),
    associatormap = cms.InputTag("tpToGlbTrackAssociation"),
    label_tp_fake = cms.InputTag("mergedtruth","MergedTrackTruth"),
    out = cms.string('validationPlots.root'),
    min = cms.double(-2.5),
    nintpT = cms.int32(300),
    label = cms.VInputTag(cms.InputTag("globalMuons")),
    maxHit = cms.double(75.0),
    label_tp_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    useInvPt = cms.bool(False),
    dirName = cms.string('RecoMuonV/MultiTrack/'),
    minHit = cms.double(0.0),
    sim = cms.string('g4SimHits'),
    associators = cms.vstring('TrackAssociatorByDeltaR1'),
    max = cms.double(2.5),
    maxpT = cms.double(3100.0),
    nint = cms.int32(50),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    UseAssociators = cms.bool(False)
)


