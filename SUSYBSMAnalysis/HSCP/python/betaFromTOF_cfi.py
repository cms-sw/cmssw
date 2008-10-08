import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.MuonSegmentMatcher_cff import *

betaFromTOF = cms.EDProducer("BetaFromTOF",
    MuonSegmentMatcher,
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite'),
        RPCLayers = cms.bool(True)
    ),
    DTsegments = cms.untracked.InputTag("dt4DSegments"),
    PruneCut = cms.double(0.2),
    HitsMin = cms.int32(3),
    debug = cms.bool(False),
    Muons = cms.untracked.InputTag("muons")
)


