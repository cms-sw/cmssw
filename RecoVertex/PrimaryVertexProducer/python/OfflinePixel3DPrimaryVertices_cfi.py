import FWCore.ParameterSet.Config as cms

pixelVertices = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(2), ## 2 cms in case there is no beamspot (ONLY FOR STARTUP!)

        minVertexFitProb = cms.double(0.01) ## 1% vertex fit probability, not used (and meaningless with the AdaptiveFitter)

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

        # no longer used
        #minSiliconHits = cms.int32(2), ## hits > 2 - for when the beamspot is not well known (ONLY FOR STARTUP)
        # minPixelHits = cms.int32(2) ## hits > 2

    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("pixelTracks"),

    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(0.1)        # 1 cm max separation betw.tracks inside clusters
        )
    ),
                               
    # probably not needed here ...
    VtxFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
        trackCompatibilityToPVcut = cms.double(0.05), ## 5%
        maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster

    )

)


