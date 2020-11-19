import FWCore.ParameterSet.Config as cms

JetCoreMCtruthSeedGenerator = cms.EDProducer("JetCoreMCtruthSeedGenerator",
    vertices=    cms.InputTag("offlinePrimaryVertices"),
    pixelClusters=    cms.InputTag("siPixelClustersPreSplitting"),
    cores= cms.InputTag("jetsForCoreTracking"),
    ptMin= cms.double(300),
    deltaR= cms.double(0.3),
    chargeFractionMin= cms.double(18000.0),
    simTracks= cms.InputTag("g4SimHits"),
    simVertex= cms.InputTag("g4SimHits"),
    simHit= cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    centralMIPCharge= cms.double(2),
    pixelCPE= cms.string( "PixelCPEGeneric" )
)
