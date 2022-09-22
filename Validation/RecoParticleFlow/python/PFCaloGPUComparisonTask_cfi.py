
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

pfHBHEGPUComparisonTask = DQMEDAnalyzer("PFCaloGPUComparisonTask",
                                        pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHE@cpu'),
                                        pfClusterToken_target = cms.untracked.InputTag('particleFlowClusterHBHE@cuda'),
                                        pfCaloGPUCompDir = cms.untracked.string("pfHBHEGPUv")
                                        )
