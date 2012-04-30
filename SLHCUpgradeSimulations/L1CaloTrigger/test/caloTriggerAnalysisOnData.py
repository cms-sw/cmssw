#####Emulator to run Raw Data
#####Written/Modified by Isobel Ojalvo
#####July 4, 2011
#############################START RAW DATA INPUT HERE#############################

import FWCore.ParameterSet.Config as cms

process = cms.Process("RCTofflineTEST")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_H_V14::All'#GR_H_V14
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
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/00142CEE-F6ED-E011-9E48-0024E86E8DA7.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/002D23D0-01EC-E011-9E56-001D096B0DB4.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/00384F26-E3E1-E011-86E1-0024E8768224.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0047EDD1-A9EA-E011-880A-0024E876A889.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0055F108-94F3-E011-AE09-001D0967D39B.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/00747E8D-DEF5-E011-8C15-001D0967DD0A.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0091CDBF-6BDE-E011-A36A-002618943922.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/00C0A6A2-D8EA-E011-9FF8-0024E86E8CFE.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/00E690D6-E403-E111-AFA2-0024E8768D41.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/023D5B61-18F0-E011-A4C3-0015178C6ADC.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0269AD8B-69F1-E011-B185-00151796D7F4.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/026BBE1E-A8F3-E011-A64F-0026B94D1ABB.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0278BA38-9EEE-E011-9EAB-0015178C4A90.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0296EC33-51FA-E011-9714-0015178C66B8.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/02BEEDA9-FEF8-E011-BC7F-0024E87680F4.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/02C50F06-0BF9-E011-8E4F-0026B94D1B23.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/02C6A448-A300-E111-881A-0024E8768072.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/02FA085A-8A01-E111-8ADC-0026B94D1ABB.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04065BF4-28F5-E011-95AB-00151796C088.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04111F29-AD03-E111-9FC9-0015178C4CDC.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04150CD9-40E8-E011-95D6-001D096B0F80.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/043B9E47-26F5-E011-AAA6-001D0967DAC1.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04B9E5BF-31E8-E011-89D8-0024E8766408.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04C9497F-5EF1-E011-B017-00151796D5F8.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04DAE6A0-32EE-E011-B229-0015178C6BA4.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/04FB9888-03FE-E011-9A0D-0015178C49D4.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0619A845-59DF-E011-B321-00266CFAEA68.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0637423F-BBF3-E011-BEA0-00151796D5C0.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/06460CA5-8800-E111-BFAB-0026B94D1AE2.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/06532423-46E6-E011-B2F9-0024E86E8DA7.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/065443F5-AAE4-E011-8F08-001D0967DA44.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/065615F1-12DF-E011-9F5D-0024E8768D41.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/0669929C-23DE-E011-A59E-001D0967BC3E.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/06B9BF98-F2EA-E011-B05E-0015178C15DC.root',
       '/store/data/Run2011B/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/0000/06D8D174-DB03-E111-BE25-001D0967DAC1.root'

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
                                   fileName = cms.string("analysis.root")
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








                                                     
