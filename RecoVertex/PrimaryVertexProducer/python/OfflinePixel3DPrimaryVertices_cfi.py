import FWCore.ParameterSet.Config as cms
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

pixelVertices = offlinePrimaryVertices.clone(
    TrackLabel = "pixelTracks",

    TkFilterParameters = dict(
        maxNormalizedChi2 = 100.0,
        minPixelLayersWithHits=3,
        minSiliconLayersWithHits = 3,
        maxD0Significance = 100.0, 
        maxEta = 100.,
    ),

    TkClusParameters = dict(
        TkDAClusParameters = dict(
            dzCutOff = 4.0,
            Tmin = 2.4,
            vertexSize = 0.01,
            uniquetrkweight = 0.9
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
# foo bar baz
