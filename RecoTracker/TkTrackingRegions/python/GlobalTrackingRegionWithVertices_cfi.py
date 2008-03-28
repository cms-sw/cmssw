import FWCore.ParameterSet.Config as cms

RegionPSet = cms.PSet(
    precise = cms.bool(True),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    useFixedError = cms.bool(True),
    originRadius = cms.double(0.2),
    sigmaZVertex = cms.double(3.0),
    fixedError = cms.double(0.2),
    VertexCollection = cms.string('pixelVertices'),
    ptMin = cms.double(0.9),
    useFoundVertices = cms.bool(True),
    nSigmaZ = cms.double(3.0)
)

