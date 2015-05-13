import FWCore.ParameterSet.Config as cms

trackMCQuality = cms.EDProducer("TrackMCQuality",
                                tracks = cms.InputTag('generalTracks'),
                                trackingParticles = cms.InputTag('mix','MergedTrackTruth'),
                                associator = cms.InputTag('quickTrackAssociatorByHits')
                                )

