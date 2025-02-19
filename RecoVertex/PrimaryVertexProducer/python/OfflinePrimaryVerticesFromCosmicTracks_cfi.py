import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesFromCosmicTracks = cms.EDProducer("PrimaryVertexProducer",

    verbose = cms.untracked.bool(False),
    TrackLabel = cms.InputTag("ctfWithMaterialTracksP5"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
                                        
    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconLayersWithHits = cms.int32(7), ## hits > 7
        maxD0Significance = cms.double(5.0), ## keep most primary tracks
        minPt = cms.double(0.0), ## better for softish events
        minPixelLayersWithHits = cms.int32(2), ## hits > 2
        trackQuality = cms.string("any")
    ),

    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameters = cms.PSet( 
            zSeparation = cms.double(0.1) ## 1 mm max separation betw. clusters
        )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )
                                        

                                        
)


