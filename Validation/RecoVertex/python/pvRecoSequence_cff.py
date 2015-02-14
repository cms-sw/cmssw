import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff import *
from Configuration.StandardSequences.GeometryRecoDB_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *

#from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *

# 2010-like PV reconstruction 

offlinePrimaryVerticesGAP = cms.EDProducer("PrimaryVertexProducer",
       verbose = cms.untracked.bool(False),
       beamSpotLabel = cms.InputTag("offlineBeamSpot"),    
       TrackLabel = cms.InputTag("generalTracks"),         # label of tracks to be used
       TkFilterParameters = cms.PSet(
                    algorithm=cms.string('filter'),
                    maxNormalizedChi2 = cms.double(20.0),    #
                    minSiliconLayersWithHits = cms.int32(5), # >= 5
                    minPixelLayersWithHits = cms.int32(2),   # >= 2
                    maxD0Significance = cms.double(100.0),     # keep most primary tracks
                    minPt = cms.double(0.0),                 # better for softish events
                    trackQuality = cms.string("any")
                ),
       # clustering
       TkClusParameters = cms.PSet(
                   algorithm   = cms.string('gap'),
                   TkGapClusParameters = cms.PSet(
                         zSeparation = cms.double(0.2)
                   )
       ),
       vertexCollections = cms.VPSet(
              [cms.PSet(label=cms.string(""),
                        algorithm = cms.string('AdaptiveVertexFitter'),
                        minNdof=cms.double(0.0),
                        useBeamConstraint = cms.bool(False),
                        maxDistanceToBeam = cms.double(2.0)
               )]
    )
)



offlinePrimaryVerticesD0s5 = offlinePrimaryVerticesGAP.clone()
offlinePrimaryVerticesD0s5.TkFilterParameters.maxD0Significance = cms.double(5)

offlinePrimaryVerticesD0s51mm = offlinePrimaryVerticesGAP.clone()
offlinePrimaryVerticesD0s51mm.TkFilterParameters.maxD0Significance = cms.double(5)
offlinePrimaryVerticesD0s51mm.TkClusParameters.TkGapClusParameters.zSeparation = cms.double(0.1) 


offlinePrimaryVerticesDA100um = cms.EDProducer("PrimaryVertexProducer",

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

    TkClusParameters = cms.PSet(
        algorithm   = cms.string("DA"),
        TkDAClusParameters = cms.PSet(
            coolingFactor = cms.double(0.6),  #  moderate annealing speed
            Tmin = cms.double(4.),            #  end of annealing
            vertexSize = cms.double(0.01),    #  
            d0CutOff = cms.double(3.),        # downweight high IP tracks
            dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
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

offlinePrimaryVerticesDA100umV7 = offlinePrimaryVerticesDA100um.clone()
offlinePrimaryVerticesDA100umV7.vertexCollections[0].maxDistanceToBeam = cms.double(2.0)
offlinePrimaryVerticesDA100umV7.TkFilterParameters.maxNormalizedChi2 = cms.double(5.0)
offlinePrimaryVerticesDA100umV7.TkClusParameters.TkDAClusParameters.coolingFactor = cms.double(0.8)
offlinePrimaryVerticesDA100umV7.TkClusParameters.TkDAClusParameters.Tmin = cms.double(9.)

offlinePrimaryVerticesDA100umV8 = offlinePrimaryVerticesDA100um.clone()
offlinePrimaryVerticesDA100umV8.vertexCollections[0].maxDistanceToBeam = cms.double(1.0)
offlinePrimaryVerticesDA100umV8.TkFilterParameters.maxNormalizedChi2 = cms.double(5.0)
offlinePrimaryVerticesDA100umV8.TkClusParameters.TkDAClusParameters.coolingFactor = cms.double(0.6)
offlinePrimaryVerticesDA100umV8.TkClusParameters.TkDAClusParameters.Tmin = cms.double(4.)


seqPVReco = cms.Sequence(offlinePrimaryVerticesGAP + offlinePrimaryVerticesD0s5 + offlinePrimaryVerticesD0s51mm +
                         offlinePrimaryVerticesDA100um + offlinePrimaryVerticesDA100umV7 + offlinePrimaryVerticesDA100umV8 )
