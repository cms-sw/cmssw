import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

offlinePrimaryVerticesFromCosmicTracks = offlinePrimaryVertices.clone(
    TrackLabel    = "ctfWithMaterialTracksP5",
    beamSpotLabel = "offlineBeamSpot",

    TkFilterParameters = dict(
        maxNormalizedChi2 = 5.0,
        minSiliconLayersWithHits = 7, ## hits > 7
        maxD0Significance = 5.0, ## keep most primary tracks
        maxD0Error = 10.0,
        maxDzError = 10.0,
        maxEta = 5.0, 
        minPixelLayersWithHits = 2, ## hits > 2
    ),

    TkClusParameters = cms.PSet(
      algorithm = cms.string("gap"),
      TkGapClusParameters = cms.PSet(
        zSeparation = cms.double(0.1) ## 1 mm max separation betw. clusters
      )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               chi2cutoff = cms.double(3.0),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )
)
# foo bar baz
# hHSR9UaLsyoyO
# peCgMiHrgY1nV
