import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Get frontier conditions   - not applied in the HCAL, see below
# Common inputs, with fake conditions
process.load("FastSimulation.Configuration.CommonInputs_cff")

from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']


# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.HLTSetup_cff")


#process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysis_cfi")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysisCalibrated_cfi")
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")



process.p1 = cms.Path(process.rctDigis+
                      process.gctDigis+
                      process.gtDigis+
                      process.l1extraParticles+
                      process.SLHCCaloTrigger+
                      process.mcSequence+
                      process.analysisSequenceCalibrated
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:SLHCOutput.root')
                           )


process.schedule = cms.Schedule(process.p1)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histograms_c.root")
                                   )
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
process.RCTConfigProducers.etMETLSB = cms.double(0.25)
process.RCTConfigProducers.jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eMinForFGCut = cms.double(100.0)
process.RCTConfigProducers.eGammaLSB = cms.double(0.25)

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
