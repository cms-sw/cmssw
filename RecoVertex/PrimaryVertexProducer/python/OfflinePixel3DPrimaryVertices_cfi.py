import FWCore.ParameterSet.Config as cms

pixelVertices = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(2), ## 2 cms in case there is no beamspot
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'), ## 100 is for when the beamspot is not well known (ONLY FOR STARTUP)
    useBeamConstraint = cms.bool(False),
    minNdof  = cms.double(2.0),               # new 
    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(100.0),
        maxD0Significance = cms.double(100.0), ## 100 is for when the beamspot is not well known (ONLY FOR STARTUP)
        minPt = cms.double(0.0), ## better for softish events
        minPixelLayersWithHits = cms.int32(3),   # >=3  three or more pixel layers
        minSiliconLayersWithHits = cms.int32(3), # >=3  (includes pixel layers)
        trackQuality = cms.string("any")
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    TrackLabel = cms.InputTag("pixelTracks"),
##     TkClusParameters = cms.PSet(
##         algorithm   = cms.string("DA"),
##         TkDAClusParameters = cms.PSet(
##             coolingFactor = cms.double(0.6),  #  moderate annealing speed
##             Tmin = cms.double(4.),            #  freeze-out, stop producing new vertices
##             vertexSize = cms.double(0.01),    #  ~ resolution
##             d0CutOff = cms.double(0.),        # don't downweight high IP tracks
##             dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
##         )
##     )
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(0.1)        # 1 mm max separation betw.tracks inside clusters
        )
    )

)


