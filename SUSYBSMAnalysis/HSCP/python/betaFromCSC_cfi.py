import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.MuonSegmentMatcher_cff import *

betaFromCSC = cms.EDProducer("BetaFromCSC",
    MuonSegmentMatcher,
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite'),
        RPCLayers = cms.bool(True)
    ),
    CSCsegments = cms.untracked.InputTag("CSCSegments"),
    PruneCut = cms.double(0.1),
    HitsMin = cms.int32(3),
    debug = cms.bool(False),
    Muons = cms.untracked.InputTag("muons")
)


