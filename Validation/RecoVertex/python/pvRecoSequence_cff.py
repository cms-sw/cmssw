import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff import *
from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *

offlinePrimaryVerticesD0s5 = offlinePrimaryVertices.clone()
offlinePrimaryVerticesD0s5.TkFilterParameters.maxD0Significance = cms.double(5)

offlinePrimaryVerticesD0s51mm = offlinePrimaryVertices.clone()
offlinePrimaryVerticesD0s51mm.TkFilterParameters.maxD0Significance = cms.double(5)
offlinePrimaryVerticesD0s51mm.TkClusParameters.TkGapClusParameters.zSeparation = cms.double(0.1) 


from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesDA_cfi import *

offlinePrimaryVerticesDA100um = cms.EDProducer("PrimaryVertexProducer",
                                               verbose = cms.untracked.bool(False),
                                               algorithm = cms.string('AdaptiveVertexFitter'),
                                               TrackLabel = cms.InputTag("generalTracks"),
                                               useBeamConstraint = cms.bool(False),
                                               beamSpotLabel = cms.InputTag("offlineBeamSpot"),
                                               minNdof  = cms.double(0.0),
                                               PVSelParameters = cms.PSet(
    maxDistanceToBeam = cms.double(1.0)
    ),
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
    vertexSize = cms.double(0.01),    #  ~ resolution / sqrt(Tmin)
    d0CutOff = cms.double(3.),        # downweight high IP tracks
    dzCutOff = cms.double(4.)         # outlier rejection after freeze-out (T<Tmin)
    )
    )
                                               )



#offlinePrimaryVerticesDA100um.TkClusParameters.TkDAClusParameters.vertexSize = cms.double(0.01)
#offlinePrimaryVerticesDA100um.TkClusParameters.verbose = cms.untracked.bool(False)
#offlinePrimaryVerticesDA100um.PVSelParameters.maxDistanceToBeam = cms.double(2.0)

offlinePrimaryVerticesDA100umV7 = offlinePrimaryVerticesDA100um.clone()
offlinePrimaryVerticesDA100umV7.PVSelParameters.maxDistanceToBeam = cms.double(2.0)
offlinePrimaryVerticesDA100umV7.TkFilterParameters.maxNormalizedChi2 = cms.double(5.0)
offlinePrimaryVerticesDA100umV7.TkClusParameters.TkDAClusParameters.coolingFactor = cms.double(0.8)
offlinePrimaryVerticesDA100umV7.TkClusParameters.TkDAClusParameters.Tmin = cms.double(9.)

offlinePrimaryVerticesDA100umV8 = offlinePrimaryVerticesDA100um.clone()
offlinePrimaryVerticesDA100umV8.PVSelParameters.maxDistanceToBeam = cms.double(1.0)
offlinePrimaryVerticesDA100umV8.TkFilterParameters.maxNormalizedChi2 = cms.double(5.0)
offlinePrimaryVerticesDA100umV8.TkClusParameters.TkDAClusParameters.coolingFactor = cms.double(0.6)
offlinePrimaryVerticesDA100umV8.TkClusParameters.TkDAClusParameters.Tmin = cms.double(4.)


seqPVReco = cms.Sequence(offlinePrimaryVerticesD0s5 + offlinePrimaryVerticesD0s51mm +
                         offlinePrimaryVerticesDA100um + offlinePrimaryVerticesDA100umV7 + offlinePrimaryVerticesDA100umV8 )
