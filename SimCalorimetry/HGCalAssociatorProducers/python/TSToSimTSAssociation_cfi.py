import FWCore.ParameterSet.Config as cms

tracksterSimClusterAssociation = cms.EDProducer("TSToSimTSAssociatorEDProducer",
    associator = cms.InputTag('simTsAssocByEnergyScoreProducer'),
    label_tst = cms.InputTag("ticlTrackstersMerge"),
    label_simTst = cms.InputTag("ticlSimTracksters"),
    label_lcl = cms.InputTag("hgcalLayerClusters")
)
