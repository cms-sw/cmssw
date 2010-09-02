import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(200)
        )

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")


# Generate event
process.load("SLHCUpgradeSimulations.L1CaloTrigger.QQH1352T_cfi")

# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")

# Famos sequences
process.load("FastSimulation.Configuration.FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# HLT paths - defined by configDB
# This one is created on the fly by FastSimulation/Configuration/test/IntegrationTestWithHLT_py.csh
process.load("FastSimulation.Configuration.HLT_8E29_cff")

# Only event accepted by L1 + HLT are reconstructed
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

# Schedule the HLT paths
process.schedule = cms.Schedule()
process.schedule.extend(process.HLTSchedule)

# Simulation sequence
#process.simulation = cms.Sequence(process.ProductionFilterSequence*process.simulationWithFamos)
process.source = cms.Source("EmptySource")
process.simulation = cms.Sequence(process.generator*process.simulationWithFamos)

# You many not want to simulate everything
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
# Parameterized magnetic field
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# Number of pileup events per crossing
process.famosPileUp.PileUpSimulator.averageNumber = 25

# Get frontier conditions   - not applied in the HCAL, see below
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

# Apply ECAL and HCAL miscalibration 
process.ecalRecHit.doMiscalib = True
process.hbhereco.doMiscalib = True
process.horeco.doMiscalib = True
process.hfreco.doMiscalib = True

# Apply Tracker misalignment
process.famosSimHits.ApplyAlignment = True
process.misalignedTrackerGeometry.applyAlignment = True
process.misalignedDTGeometry.applyAlignment = True
process.misalignedCSCGeometry.applyAlignment = True

# Attention ! for the HCAL IDEAL==STARTUP
# process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
# process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_0.0.xml"

process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysisCalibrated_cfi")

# Famos with everything !
#Load Scales
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")
process.ecalRecHit.doDigis = True
process.hbhereco.doDigis = True
process.horeco.doDigis = True
process.hfreco.doDigis = True

process.p1 = cms.Path(process.generator+
                      process.famosWithEverything+
                      process.simEcalTriggerPrimitiveDigis+
                      process.simHcalTriggerPrimitiveDigis+
                      process.SLHCCaloTrigger+
                      process.mcSequence+
                      process.analysisSequenceCalibrated
)

#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything)
process.source = cms.Source("EmptySource")
process.simulation = cms.Sequence(process.generator*process.simulationWithFamos)

process.schedule.append(process.p1)
# END OF SLHC STUFF

# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
                              outputCommands = cms.untracked.vstring('drop *_*_*_*',
                                                                     'keep *_L1Calo*_*_*',
                                                                     'keep *_*SLHCL1ExtraParticles*_*_*',
                                                                     'keep *_*l1extraParticles*_*_*',
                                                                     'keep *_genParticles_*_*',
                                                                     'keep recoGen*_*_*_*',
                                                                     'keep *_simEcalTriggerPrimitiveDigis_*_*',
                                                                     'keep *_simHcalTriggerPrimitiveDigis_*_*' 
                                                                     ),
                     
                              fileName = cms.untracked.string('SLHC_LHC_Output.root')
                              )
process.outpath = cms.EndPath(process.o1)

# Add endpaths to the schedule
process.schedule.append(process.outpath)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('histograms.root')
                                   )


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



#CALO TRIGGER CONFIGURATION OVERRIDE

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")
process.RCTConfigProducers.eMaxForHoECut = cms.double(60.0)
process.RCTConfigProducers.hOeCut = cms.double(0.05)
process.RCTConfigProducers.eGammaECalScaleFactors = cms.vdouble(1.0, 1.01, 1.02, 1.02, 1.02,
                                                      1.06, 1.04, 1.04, 1.05, 1.09,
                                                      1.1, 1.1, 1.15, 1.2, 1.27,
                                                      1.29, 1.32, 1.52, 1.52, 1.48,
                                                      1.4, 1.32, 1.26, 1.21, 1.17,
                                                      1.15, 1.15, 1.15)
process.RCTConfigProducers.eMinForHoECut = cms.double(3.0)
process.RCTConfigProducers.hActivityCut = cms.double(4.0)
process.RCTConfigProducers.eActivityCut = cms.double(4.0)
process.RCTConfigProducers.jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eicIsolationThreshold = cms.uint32(7)
process.RCTConfigProducers.etMETLSB = cms.double(0.5)
process.RCTConfigProducers.jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eMinForFGCut = cms.double(100.0)
process.RCTConfigProducers.eGammaLSB = cms.double(0.5)

process.L1GctConfigProducers = cms.ESProducer("L1GctConfigProducers",
                                          JetFinderCentralJetSeed = cms.double(0.5),
                                          JetFinderForwardJetSeed = cms.double(0.5),
                                          TauIsoEtThreshold = cms.double(2.0),
                                          HtJetEtThreshold = cms.double(10.0),
                                          MHtJetEtThreshold = cms.double(10.0),
                                          RctRegionEtLSB = cms.double(0.25),
                                          GctHtLSB = cms.double(0.25),
                                          # The CalibrationStyle should be "none", "PowerSeries", or "ORCAStyle
                                          CalibrationStyle = cms.string('None'),
                                          ConvertEtValuesToEnergy = cms.bool(False)
)                                      
