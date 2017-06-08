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
        maxEta = cms.double(100.),
        trackQuality = cms.string("any")
    ),

    TkClusParameters = cms.PSet(
        algorithm = cms.string('DA_vect'),
        TkDAClusParameters = cms.PSet(
            dzCutOff = cms.double(4.0),
            d0CutOff = cms.double(3.0),
            Tmin = cms.double(2.4),
            Tpurge = cms.double(2.0),
            Tstop = cms.double(0.5),
            coolingFactor = cms.double(0.6),
            vertexSize = cms.double(0.01),
            zmerge = cms.double(0.01),
            uniquetrkweight = cms.double(0.9)
        )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               chi2cutoff = cms.double(3.0),
               minNdof=cms.double(2.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(2.0)
               )
      ]
    )
                                        

                                        
)


