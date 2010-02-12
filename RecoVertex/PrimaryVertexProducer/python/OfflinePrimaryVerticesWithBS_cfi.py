import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesWithBS = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(1.0) # meaningless for constrained fits
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(True),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    minNdof  = cms.double(2.0),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconLayersWithHits = cms.int32(6), # >=6  (TDR > 7 hits)
        minPixelLayersWithHits = cms.int32(2),   # >=2  (TDR > 2 hits)
        maxD0Significance = cms.double(5.0),     # keep most primary tracks
        minPt = cms.double(0.0),                 # better for softish events
        trackQuality= cms.string("any")
    ),
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameters = cms.PSet( 
            zSeparation = cms.double(0.1)        # 1 mm max separation betw.tracks inside clusters
        )

    )
    #VtxFinderParameters = cms.PSet(
    #    ptCut = cms.double(0.0),
    #    vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
    #	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
    #     trackCompatibilityToPVcut = cms.double(0.05), ## 5%
    #    maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster
    #
    #),
)


