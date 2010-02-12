import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesDA = cms.EDProducer("PrimaryVertexProducer",
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(True),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    minNdof  = cms.double(2.0),
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(1.0) # meaningless for constrained fits
    ),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconLayersWithHits = cms.int32(6),
        maxD0Significance = cms.double(5.0), ## keep most primary tracks
        minPt = cms.double(0.0), ## better for softish events
        minPixelLayersWithHits=cms.int32(2),
        trackQuality = cms.string("any")
    ),
    # label of tracks to be used
    #VtxFinderParameters = cms.PSet(
    #    ptCut = cms.double(0.0),
    #    vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
    #	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
    #    trackCompatibilityToPVcut = cms.double(0.05), ## 5%
    #    maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster
    #),
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("DA"),
        TkDAClusParameters = cms.PSet( 
            coolingFactor = cms.double(0.8),  #  rather slow annealing for now
            Tmin = cms.double(9.),            #  end of annealing
            vertexSize = cms.double(0.05)     #  ~ resolution
        )
    )
)


