import FWCore.ParameterSet.Config as cms

TTClusterAssociatorFromPixelDigis = cms.EDProducer("TTClusterAssociator_Phase2TrackerDigi_",
    TTClusters = cms.VInputTag( cms.InputTag("TTClustersFromPhase2TrackerDigis", "ClusterInclusive"),
                                cms.InputTag("TTStubsFromPhase2TrackerDigis", "ClusterAccepted"),
    ),
    simTrackHits = cms.InputTag( "g4SimHits" ),
    digiSimLinks = cms.InputTag( "mix","Tracker" ), 
    trackingParts= cms.InputTag( "mix","MergedTrackTruth" )
)

