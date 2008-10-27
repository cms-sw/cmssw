import FWCore.ParameterSet.Config as cms

tpToStaTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByDeltaR2'),
    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tr = cms.InputTag("standAloneMuons","UpdatedAtVtx")
)


