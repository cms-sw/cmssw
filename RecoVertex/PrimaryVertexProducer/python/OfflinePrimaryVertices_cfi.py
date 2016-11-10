import FWCore.ParameterSet.Config as cms

from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA_vectParameters

offlinePrimaryVertices = cms.EDProducer(
    "PrimaryVertexProducer",

    verbose = cms.untracked.bool(False),
    TrackLabel = cms.InputTag("generalTracks"),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    
    TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(20.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(5.0), 
        minPt = cms.double(0.0),
        trackQuality = cms.string("any")
    ),

    TkClusParameters = DA_vectParameters,

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               ),
      cms.PSet(label=cms.string("WithBS"),
               algorithm = cms.string('AdaptiveVertexFitter'),
               minNdof=cms.double(2.0),
               useBeamConstraint = cms.bool(True),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )
                                        

                                        
)

# This customization is needed in the trackingLowPU era to be able to
# produce vertices also in the cases in which the pixel detector is
# not included in data-taking, like it was the case for "Quiet Beam"
# collisions on 2016 with run 269207.

from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(offlinePrimaryVertices,
                            TkFilterParameters = dict(minPixelLayersWithHits = 0))
