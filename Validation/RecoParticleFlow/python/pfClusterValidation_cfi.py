import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

pfClusterValidation = DQMEDAnalyzer('PFClusterValidation',
    pflowClusterECAL = cms.untracked.InputTag('particleFlowClusterECAL'),
    pflowClusterHCAL = cms.untracked.InputTag('particleFlowClusterHCAL'),
    pflowClusterHO = cms.untracked.InputTag('particleFlowClusterHO'),
    pflowClusterHF = cms.untracked.InputTag('particleFlowClusterHF'),
)
