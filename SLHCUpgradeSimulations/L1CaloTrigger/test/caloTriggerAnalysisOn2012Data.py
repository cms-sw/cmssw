#####Emulator to run Raw Data
#####Written/Modified by Isobel Ojalvo
#####July 4, 2011
#############################START RAW DATA INPUT HERE#############################

import FWCore.ParameterSet.Config as cms

process = cms.Process("RCTofflineTEST")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_P_V32::All'
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

# UNCOMMENT THIS LINE TO RUN ON SETTINGS FROM THE DATABASE
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource', 'GlobalTag')

process.load("Configuration.StandardSequences.Geometry_cff")

#unpacking
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/06DC9A42-0986-E111-92FC-001D09F2A465.root',
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/746E1889-0886-E111-BC83-0025901D629C.root',
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/92AF22F6-0986-E111-A2FA-BCAEC518FF54.root',
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/38A85564-0B86-E111-9830-BCAEC518FF30.root',
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/8C8414A3-0F86-E111-A9F7-BCAEC5364C62.root',
'/store/data/Run2012A/DoubleElectron/AOD/PromptReco-v1/000/190/945/B278DEAF-0A86-E111-AA18-BCAEC5329719.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/2675AA20-F683-E111-BB99-003048F11DE2.root',
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/96578865-FE83-E111-B129-003048F11112.root',
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/D6CEDCD6-F983-E111-98DF-001D09F2A49C.root',
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/80C2940A-F883-E111-9086-BCAEC532970D.root',
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/D64269F2-F683-E111-B25C-00237DDC5C24.root',
'/store/data/Run2012A/DoubleElectron/RAW/v1/000/190/945/F0343B82-F783-E111-AD45-001D09F2B30B.root'
      )
)

process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")

#process.load("RecoParticleFlow.PFProducer.particleFlow_cff") ##added particle flow cff
 
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTriggerAnalysisOnData_cfi")

process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi")

process.p1 = cms.Path(
                      process.RawToDigi+
                      process.SLHCCaloTrigger+
                      process.l1extraParticles+
                     # process.particleFlow+  ##added for particleFlow
                      process.analysisSequenceCalibrated
)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("SLHCTausRatesMinbiasSkim.root")
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
process.RCTConfigProducers.eicIsolationThreshold = cms.uint32(6)
process.RCTConfigProducers.etMETLSB = cms.double(0.25)
process.RCTConfigProducers.jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0, 1.0, 1.0,
                                                                1.0, 1.0, 1.0)
process.RCTConfigProducers.eMinForFGCut = cms.double(100.0)
process.RCTConfigProducers.eGammaLSB = cms.double(0.25)








                                                     
