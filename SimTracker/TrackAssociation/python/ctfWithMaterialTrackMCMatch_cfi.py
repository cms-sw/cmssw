import FWCore.ParameterSet.Config as cms

ctfWithMaterialTrackMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("mergedtruth","MergedTrackTruth"),
    tracks = cms.InputTag("ctfWithMaterialTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


