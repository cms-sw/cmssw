from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import *
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017

pp_on_XeXe_2017.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(
        VertexCollection = "firstStepPrimaryVertices",
        beamSpot = cms.InputTag("offlineBeamSpot"),
        maxNVertices = cms.int32(-1),
        nSigmaZ = cms.double(4.),
        precise = cms.bool(True),
        sigmaZVertex = cms.double(4.),
        useFakeVertices = cms.bool(False),
        useFixedError = cms.bool(True),
        useFoundVertices = cms.bool(True),
        useMultipleScattering = cms.bool(False)                                                  
        )
)
