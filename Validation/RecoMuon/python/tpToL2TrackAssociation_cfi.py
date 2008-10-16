import FWCore.ParameterSet.Config as cms

tpToL2TrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByDeltaR2'),
    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tr = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)


