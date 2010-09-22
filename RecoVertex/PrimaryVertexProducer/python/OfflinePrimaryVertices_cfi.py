import FWCore.ParameterSet.Config as cms

offlinePrimaryVertices = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(2) ## 0.1
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    minNdof  = cms.double(0.0),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(20.0),     # 
        minSiliconLayersWithHits = cms.int32(5), # >= 5
        minPixelLayersWithHits = cms.int32(2),   # >= 2 
        maxD0Significance = cms.double(100.0),     # keep most primary tracks
        minPt = cms.double(0.0),                 # better for softish events
        trackQuality = cms.string("any")
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(False),


    # clustering
    TkClusParameters = cms.PSet(
        algorithm   = cms.string('gap'),
        TkGapClusParameters = cms.PSet( 
            zSeparation = cms.double(0.2) 
        )
    )
)


