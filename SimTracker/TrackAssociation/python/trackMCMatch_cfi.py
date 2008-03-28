import FWCore.ParameterSet.Config as cms

trackMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("trackingtruthprod"),
    tracks = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


