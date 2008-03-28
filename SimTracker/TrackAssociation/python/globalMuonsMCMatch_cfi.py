import FWCore.ParameterSet.Config as cms

globalMuonsMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("trackingtruthprod"),
    tracks = cms.InputTag("globalMuons"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


