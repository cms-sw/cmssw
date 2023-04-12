import FWCore.ParameterSet.Config as cms

StripSubClusterShapeFilterParams = cms.PSet(
    maxNSat = cms.uint32(3),
    trimMaxADC = cms.double(30.),
    trimMaxFracTotal = cms.double(0.15),
    trimMaxFracNeigh = cms.double(0.25),
    maxTrimmedSizeDiffPos = cms.double(0.7),
    maxTrimmedSizeDiffNeg = cms.double(1.0),
    subclusterWindow = cms.double(0.7),
    seedCutMIPs = cms.double(0.35),
    seedCutSN = cms.double(7.),
    subclusterCutMIPs = cms.double(0.45),
    subclusterCutSN = cms.double(12.),
)
