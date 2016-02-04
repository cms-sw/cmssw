import FWCore.ParameterSet.Config as cms

globalMuonsMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("mergedtruth","MergedTrackTruth"),
    tracks = cms.InputTag("globalMuons"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


