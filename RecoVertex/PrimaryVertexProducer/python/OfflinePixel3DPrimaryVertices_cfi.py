import FWCore.ParameterSet.Config as cms

pixelVertices = cms.EDProducer("PrimaryVertexProducer",

    verbose = cms.untracked.bool(False),
    TrackLabel = cms.InputTag("pixelTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
                                        
    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(100.0),
        minPixelLayersWithHits=cms.int32(3),
        minSiliconLayersWithHits = cms.int32(3),
        maxD0Significance = cms.double(100.0), 
        minPt = cms.double(0.0),
        trackQuality = cms.string("any")
    ),

    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(0.1)        # 1 mm max separation betw.tracks inside clusters
        )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(2.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(2.0)
               )
      ]
    )
                                        

                                        
)


