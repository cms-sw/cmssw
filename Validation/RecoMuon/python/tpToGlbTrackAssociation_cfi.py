import FWCore.ParameterSet.Config as cms

tpToGlbTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByDeltaR1'),
    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tr = cms.InputTag("globalMuons")
)


