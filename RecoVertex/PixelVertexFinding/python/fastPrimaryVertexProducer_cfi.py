import FWCore.ParameterSet.Config as cms

fastPrimaryVertexProducer = cms.EDProducer("FastPrimaryVertexProducer",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusters = cms.InputTag("hltSiPixelClusters"),
    jets  = cms.InputTag("hltBLifetimeL25JetsHbb"),
    maxZ = cms.double(18),
    pixelCPE = cms.string("hltESPPixelCPEGeneric"),
    maxSizeX = cms.double(3),
    maxDeltaPhi = cms.double(0.2),
    clusterLength = cms.double(2.0)
)


