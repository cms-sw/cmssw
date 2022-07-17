import FWCore.ParameterSet.Config as cms

standAloneMuonsMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"),
    tracks = cms.InputTag("standAloneMuons"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('trackAssociatorByHits'),
    throwOnMissingTPCollection = cms.bool(True)
)


