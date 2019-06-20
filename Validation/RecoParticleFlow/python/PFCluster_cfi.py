import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pfclusterAnalyzer = DQMEDAnalyzer('PFClusterValidation',
                                   outputFile               = cms.untracked.string(''),
                                   pflowClusterECAL = cms.untracked.InputTag('particleFlowClusterECAL'),
                                   pflowClusterHCAL = cms.untracked.InputTag('particleFlowClusterHCAL'),
                                   pflowClusterHO = cms.untracked.InputTag('particleFlowClusterHO'),
                                   pflowClusterHF = cms.untracked.InputTag('particleFlowClusterHF'),
                                   #hcalselector             = cms.untracked.string('all'),
                                   mc                       = cms.untracked.bool(True)
                               
)


