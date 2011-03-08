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

offlinePrimaryVerticesDA100um = offlinePrimaryVerticesDA.clone()
offlinePrimaryVerticesDA100um.TkClusParameters=cms.PSet(algorithm=cms.string('DA'),
                                                        TkDAClusParameters = cms.PSet(
    verbose = cms.untracked.bool(False),
    coolingFactor = cms.double(0.6),
    Tmin = cms.double(4.0),           #  end of annealing
    vertexSize = cms.double(0.01)      #  ~ resolution / sqrt(Tmin)
    )
                                                        )
#offlinePrimaryVerticesDA100um.TkClusParameters.TkDAClusParameters.vertexSize = cms.double(0.01)
#offlinePrimaryVerticesDA100um.TkClusParameters.verbose = cms.untracked.bool(False)
#offlinePrimaryVerticesDA100um.PVSelParameters.maxDistanceToBeam = cms.double(2.0)


seqPVReco = cms.Sequence(offlinePrimaryVerticesD0s5 + offlinePrimaryVerticesD0s51mm + offlinePrimaryVerticesDA100um )
