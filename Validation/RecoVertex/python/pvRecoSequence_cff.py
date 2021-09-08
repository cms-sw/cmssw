import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff import *
from Configuration.StandardSequences.GeometryRecoDB_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *

#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *

# 2010-like PV reconstruction 
import RecoVertex.PrimaryVertexProducer.primaryVertexProducer_cfi as _mod
offlinePrimaryVerticesGAP = _mod.primaryVertexProducer.clone(
       TrackLabel = "generalTracks",             # label of tracks to be used
       TkFilterParameters = dict(
                    maxNormalizedChi2 = 20.0,    #
                    minSiliconLayersWithHits = 5,# >= 5
                    minPixelLayersWithHits = 2,  # >= 2
                    maxD0Significance = 100.0,   # keep most primary tracks
                    minPt = 0.0,                 # better for softish events
                    maxEta = 5.0
                ),
       # clustering
       TkClusParameters = dict(
                   algorithm   = 'gap',
                   TkGapClusParameters = dict(
                         zSeparation = 0.2
                   )
       ),
       vertexCollections = cms.VPSet(
              [cms.PSet(label=cms.string(""),
                        chi2cutoff = cms.double(3.0),
                        algorithm = cms.string('AdaptiveVertexFitter'),
                        minNdof=cms.double(0.0),
                        useBeamConstraint = cms.bool(False),
                        maxDistanceToBeam = cms.double(2.0)
                       )
              ]
       )
)

offlinePrimaryVerticesD0s5 = offlinePrimaryVerticesGAP.clone(
    TkFilterParameters = dict( 
           maxD0Significance = 5 
    )
)
offlinePrimaryVerticesD0s51mm = offlinePrimaryVerticesGAP.clone(
    TkFilterParameters = dict(
           maxD0Significance = 5
    ),
    TkClusParameters = dict(
           TkGapClusParameters = dict(
                  zSeparation = 0.1
           )
    )
)

offlinePrimaryVerticesDA100um = _mod.primaryVertexProducer.clone(
    TrackLabel = "generalTracks",
    TkFilterParameters = dict(
        maxNormalizedChi2 = 20.0,
        maxD0Significance = 5.0,
        maxEta = 5.0
    ),

    TkClusParameters = dict(
        algorithm   = "DA",
        TkDAClusParameters = dict(
            coolingFactor = 0.6,  #  moderate annealing speed
            Tmin = 4.0,           #  end of annealing
            vertexSize = 0.01,    #  
            d0CutOff = 3.,        # downweight high IP tracks
            dzCutOff = 4.         # outlier rejection after freeze-out (T<Tmin)
        )
    ),

    vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               chi2cutoff = cms.double(3.0),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
              )
     ]
    )
)

offlinePrimaryVerticesDA100umV7 = offlinePrimaryVerticesDA100um.clone(
    vertexCollections = {0: dict(maxDistanceToBeam = 2.0)},
    TkFilterParameters = dict(
           maxNormalizedChi2 = 5.0
    ),
    TkClusParameters = dict(
           TkDAClusParameters = dict(
                  coolingFactor = 0.8,
                  Tmin = 9.
           )
    )
)
offlinePrimaryVerticesDA100umV8 = offlinePrimaryVerticesDA100um.clone(
    vertexCollections = {0: dict(maxDistanceToBeam = 1.0)},
    TkFilterParameters = dict(
           maxNormalizedChi2 = 5.0
    ),
    TkClusParameters = dict(
           TkDAClusParameters = dict(
                  coolingFactor = 0.6,
                  Tmin = 4.
           )
    )
)

seqPVReco = cms.Sequence(offlinePrimaryVerticesGAP + offlinePrimaryVerticesD0s5 + offlinePrimaryVerticesD0s51mm +
                         offlinePrimaryVerticesDA100um + offlinePrimaryVerticesDA100umV7 + offlinePrimaryVerticesDA100umV8 )
