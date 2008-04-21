import FWCore.ParameterSet.Config as cms

trackingParticleRecoTrackAsssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tr = cms.InputTag("generalTracks")
)


