import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pfClusterHBHEOnlyAlpakaComparison = DQMEDAnalyzer("PFCaloGPUComparisonTask",
                                                    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHEOnlyLegacy'),
                                                    pfClusterToken_target = cms.untracked.InputTag('particleFlowClusterHBHEOnly'),
                                                    pfCaloGPUCompDir = cms.untracked.string("pfClusterHBHEAlpakaV")
)

pfClusterHBHEAlpakaComparison = DQMEDAnalyzer("PFCaloGPUComparisonTask",
                                                    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHE'),
                                                    pfClusterToken_target = cms.untracked.InputTag('legacyPFClusterProducer'),
                                                    pfCaloGPUCompDir = cms.untracked.string("pfClusterHBHEAlpakaV")
)
