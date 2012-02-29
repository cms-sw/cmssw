import sys

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1Tproducer")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

process.L1CaloTriggerSetup.InputXMLFile=cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')

#process.L1CaloTowerProducer.HCALDigis = cms.InputTag("simHcalUpgradeTriggerPrimitiveDigis")
#process.L1CaloTowerProducer.UseUpgradeHCAL = cms.bool(True)

process.L1TowerJetProducer.JetSize = cms.uint32(9)
process.L1TowerJetProducer.JetShape = cms.string("circle")

process.p1 = cms.Path(
				process.L1CaloTowerProducer+
                process.L1TowerJetProducer
			)

process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
                                        'file:EventData.root'
                                )
                        )


# Keep the logging output to a nice level #
process.load("FWCore/MessageService/MessageLogger_cfi")

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

