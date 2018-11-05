import FWCore.ParameterSet.Config as cms

JetCorePerfectSeedGenerator = cms.EDProducer("JetCorePerfectSeedGenerator",
    vertices=    cms.InputTag("offlinePrimaryVertices"),
    pixelClusters=    cms.InputTag("siPixelClustersPreSplitting"),
    cores= cms.InputTag("jetsForCoreTracking"),
    ptMin= cms.double(300),
    deltaR= cms.double(0.1),
    chargeFractionMin= cms.double(18000.0),
    centralMIPCharge= cms.double(2),
    pixelCPE= cms.string( "PixelCPEGeneric" )
)
