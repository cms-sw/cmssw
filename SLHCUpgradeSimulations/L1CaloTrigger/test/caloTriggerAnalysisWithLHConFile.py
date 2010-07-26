import FWCore.ParameterSet.Config as cms
import glob

process = cms.Process("PROD2")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# Simulation sequence
#process.simulation = cms.Sequence(process.ProductionFilterSequence*process.simulationWithFamos)
#process.source = cms.Source("EmptySource")
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:SLHC_LHC_Output.root')
                           )

# Get frontier conditions   - not applied in the HCAL, see below
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

# Attention ! for the HCAL IDEAL==STARTUP
# process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_0.0.xml"

process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysis_cfi")

#Load Scales
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

process.ecalRecHit.doDigis = True
process.hbhereco.doDigis = True
process.horeco.doDigis = True
process.hfreco.doDigis = True

process.p1 = cms.Path(
    process.mcSequence+
    process.analysisSequence
    )

#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring("file:UpgdCaloTrigMinBPU25-test.root")
#                            )  
#process.simulation = cms.Sequence(process.simulationWithFamos)

# To write out events 
#process.load("FastSimulation.Configuration.EventContent_cff")
#process.o1 = cms.OutputModule("PoolOutputModule",
#	outputCommands = cms.untracked.vstring('drop *_*_*_*',
#                                 'keep *_L1Calo*_*_*',
#                                 'keep *_SLHCL1ExtraParticles_*_*',
#                                 'keep *_l1extraParticles_*_*'),
#    fileName = cms.untracked.string('SLHC_LHC_Output.root')
#)
#process.outpath = cms.EndPath(process.o1)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('SLHC_LHC_AnalyzerOutput.root')
                                   )

# Add endpaths to the schedule
#process.schedule.append(process.outpath)

# Keep the logging output to a nice level #
# process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
# process.MessageLogger.categories.append('L1GtTrigReport')
# process.MessageLogger.categories.append('HLTrigReport')
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

