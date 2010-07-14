import FWCore.ParameterSet.Config as cms


# Upgrade Cal Trigger
from SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysis_cfi import *
from L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff import *
from SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff import *
from SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysis_cfi import *
SLHC_L1 = cms.Sequence(   SLHCCaloTrigger+
                           mcSequence+
                           analysisSequence
                       )      
