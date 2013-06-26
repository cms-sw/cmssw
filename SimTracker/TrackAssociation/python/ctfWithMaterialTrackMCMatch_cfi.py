import FWCore.ParameterSet.Config as cms

ctfWithMaterialTrackMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"),
    tracks = cms.InputTag("ctfWithMaterialTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


