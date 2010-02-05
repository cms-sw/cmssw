import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesDA = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(1.0) # meaningless for constrained fits

        # minVertexFitProb = cms.double(0.01) ## 1% vertex fit probability /NA  obsolete

    ),
    verbose = cms.untracked.bool(True),
    algorithm = cms.string('AdaptiveVertexFitter'),
    minNdof  = cms.double(1.0),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(7), ## hits > 7
        minSiliconLayersWithHits = cms.int32(-1),

        maxD0Significance = cms.double(5.0), ## keep most primary tracks

        minPt = cms.double(0.0), ## better for softish events

        minPixelHits = cms.int32(2), ## hits > 2
        minPixelLayersWithHits=cms.int32(-1)

    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(True),
    VtxFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
        trackCompatibilityToPVcut = cms.double(0.05), ## 5%
        maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster

    ),
    DAFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        TMin  = cms.double(10.)
    ),
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("DA"),
        TkDAClusParameters = cms.PSet( 
            coolingFactor = cms.double(0.8),  #  rather slow annealing for now
            Tmin = cms.double(10.),           #  end of annealing
            zSeparation = cms.double(0.2)            #  minimal allowed cluster separation in cm
        )
    )
)


