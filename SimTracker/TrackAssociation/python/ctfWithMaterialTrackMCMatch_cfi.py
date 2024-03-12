import FWCore.ParameterSet.Config as cms

ctfWithMaterialTrackMCMatch = cms.EDProducer("MCTrackMatcher",
    trackingParticles = cms.InputTag("mix","MergedTrackTruth"),
    tracks = cms.InputTag("ctfWithMaterialTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)


# foo bar baz
# Xe8TMTzvQ2lMD
# Ql5AnE6OTn5ya
