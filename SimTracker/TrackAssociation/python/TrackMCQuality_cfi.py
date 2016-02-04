import FWCore.ParameterSet.Config as cms

TrackMCQuality = cms.EDProducer("TrackMCQuality",
                                label_tr = cms.InputTag('generalTracks'),
                                label_tp = cms.InputTag('mergedtruth','MergedTrackTruth'),
                                associator = cms.string('TrackAssociatorByHits')
                                )

