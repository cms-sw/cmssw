from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import *
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingLowPU.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
trackingPhase1PU70.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
trackingPhase2PU140.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
