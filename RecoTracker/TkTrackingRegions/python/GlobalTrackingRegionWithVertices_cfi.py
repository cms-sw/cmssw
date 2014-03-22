import FWCore.ParameterSet.Config as cms

# moving to block.  Will delete this one
# when transition is done
# NO: must be completely removed, as the CFI is included at top level
#     otherwise we get duplicate definitions
# PSet RegionPSet = {
#   double ptMin = 0.9
#   double originRadius = 0.2
#   double nSigmaZ = 3.0
#   InputTag beamSpot = "offlineBeamSpot"
#   bool precise = true
# 
#   bool useFoundVertices =  true
#   string VertexCollection = "pixelVertices"
# 
#   double sigmaZVertex = 3.0
#   bool useFixedError = true
#   double fixedError = 0.2
# }
RegionPSetWithVerticesBlock = cms.PSet(
    RegionPSet = cms.PSet(
        precise = cms.bool(True),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        useFixedError = cms.bool(True),
        originRadius = cms.double(0.2),
        sigmaZVertex = cms.double(3.0),
        fixedError = cms.double(0.2),
        VertexCollection = cms.InputTag("pixelVertices"),
        ptMin = cms.double(0.9),
        useFoundVertices = cms.bool(True),
        useFakeVertices = cms.bool(False),
        nSigmaZ = cms.double(4.0)
    )
)

