import FWCore.ParameterSet.Config as cms

RegionPSetWithVerticesBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        useMultipleScattering = cms.bool(False),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        useFixedError = cms.bool(True),
        originRadius = cms.double(0.2),
        sigmaZVertex = cms.double(3.0),
        fixedError = cms.double(0.2),
        VertexCollection = cms.InputTag("firstStepPrimaryVertices"),
        ptMin = cms.double(0.9),
        useFoundVertices = cms.bool(True),
        useFakeVertices = cms.bool(False),
        maxNVertices = cms.int32(-1),
        nSigmaZ = cms.double(4.0),

        #parameters for HI collisions; doesn't do anything if all 3 booleans are False
       
        originRScaling4BigEvts = cms.bool(False),
        ptMinScaling4BigEvts = cms.bool(False),
        halfLengthScaling4BigEvts = cms.bool(False),
        pixelClustersForScaling = cms.InputTag("siPixelClusters"),
        minOriginR = cms.double(0),
        maxPtMin = cms.double(1000),
        minHalfLength = cms.double(0),
        scalingStartNPix = cms.double(0),
        scalingEndNPix = cms.double(1)
    )
)
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(RegionPSetWithVerticesBlock,
    RegionPSet = dict(VertexCollection = "pixelVertices")
)
