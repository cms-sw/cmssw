import sys

import FWCore.ParameterSet.Config as cms

process = cms.Process("L1Tproducer")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

process.L1CaloTriggerSetup.InputXMLFile=cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')


#-------------------------------------------------------------------------------------------------------------------------
# To use the upgraded HCAL Trigger Primitives, instead of the regular Trigger Primitives, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
#process.L1CaloTowerProducer.HCALDigis = cms.InputTag("simHcalUpgradeTriggerPrimitiveDigis")
#process.L1CaloTowerProducer.UseUpgradeHCAL = cms.bool(True)
#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------
# To configure a manually added Ring-subtraction algorithm, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
#process.L1RingSubtractionProducer.RingSubtractionType = cms.string("mean") # "mean", "median" or "constant"
#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------
# To configure a manually added Tower-Jet algorithm, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
#process.L1TowerJetProducer.src = cms.InputTag("L1CaloTowerProducer")
#process.L1TowerJetProducer.src = cms.InputTag("L1RingSubtractionProducer")
#process.L1TowerJetProducer.JetDiameter = cms.uint32(9)
#process.L1TowerJetProducer.JetShape = cms.string("circle") # "circle" or "square"
#-------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------
# To automatically add all permutations of the Ring-Subtraction and Tower-Jet algorithms, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
#from SLHCUpgradeSimulations.L1CaloTrigger.AllTowerJetPermutations import *
#process.AllTowerJetPermutations = CreateAllTowerJetPermutations( process )	#We are taking advantage of the fact that the config files are python
#-------------------------------------------------------------------------------------------------------------------------


process.p1 = cms.Path(
				process.L1CaloTowerProducer+
#				process.L1RingSubtractionProducer+
                process.L1TowerJetProducer+
                process.L1TowerJetFilter1D
#-------------------------------------------------------------------------------------------------------------------------
# To use manually added Ring-subtraction and Tower-Jet algorithms, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
				#process.L1RingSubtractionProducer+
                #process.L1TowerJetProducer
#-------------------------------------------------------------------------------------------------------------------------
# To use the automatically generated permutations of the Ring-subtraction and Tower-Jet algorithms, uncomment these lines
#-------------------------------------------------------------------------------------------------------------------------
#				process.AllTowerJetPermutations
			)

process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
                                        'file:/vols/cms04/rel08/EventData.root'
                                )
                        )


# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *_*_*_*',
                                                                     'keep *_L1CaloTower*_*_*',
                                                                     'keep *_L1RingSubtraction*_*_*',
                                                                     'keep *_L1TowerJet*_*_*',
                                                                      ),
    fileName = cms.untracked.string('TowerJet.root')
)
process.outpath = cms.EndPath(process.o1)


# Keep the logging output to a nice level #
process.load("FWCore/MessageService/MessageLogger_cfi")

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

