from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import *
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(globalTrackingRegionWithVertices, RegionPSet = dict(VertexCollection = "pixelVertices"))
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

(pp_on_XeXe_2017 | pp_on_AA).toModify(globalTrackingRegionWithVertices, RegionPSet = dict(
            VertexCollection = "firstStepPrimaryVertices",
            beamSpot = "offlineBeamSpot",
            maxNVertices = -1,
            nSigmaZ = 4.,
            precise = True,
            sigmaZVertex = 4.,
            useFakeVertices = False,
            useFixedError = True,
            useFoundVertices = True,
            useMultipleScattering = False                                                  
            )
                             )
