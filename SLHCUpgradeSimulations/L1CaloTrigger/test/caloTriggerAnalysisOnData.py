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
#        'file:/scratch/ojalvo/TauSkim/TauRawReco.root',
#        'file:/scratch/ojalvo/TauSkim/TauRawReco001.root',
#        'file:/scratch/ojalvo/TauSkim/TauRawReco002.root'    
##    '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/00257D72-F9DC-E011-A401-00261894392C.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/002CD2CA-4BF9-E011-8377-001A92971B04.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/0086BA65-33EE-E011-9BEC-0018F3D09652.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/00A7606D-22F9-E011-8437-001A92971BDC.root',
##       '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/00C112D0-ABE6-E011-98B7-0018F3D096C6.root',
##       '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/00C632B2-04EB-E011-BD0A-0018F3D095EC.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/0203D9EF-6AEE-E011-8FC4-003048678BAA.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02094552-C2EA-E011-8D86-0018F3D096F0.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/022787E9-0BF0-E011-88E4-00261894392F.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02350F5F-37EB-E011-B5B9-001A92810ADC.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02431D41-85F9-E011-89A0-001A92971AD0.root',
##     '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/024B3FB7-88F6-E011-B556-00304867920C.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/0268062B-C200-E111-8BDF-00248C0BE018.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/0270A6AE-0BEB-E011-A87A-001A928116B2.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02768AE9-ABE4-E011-909D-0026189438FC.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/027B85ED-24FE-E011-8D87-001A92810AAE.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02A38E0C-C3FB-E011-9EC1-002618943916.root',
##      '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02CB1413-EFDE-E011-9075-0026189438F3.root',
##    '/store/data/Run2011B/MinimumBias/RAW-RECO/ValSkim-PromptSkim-v1/0000/02CF7AFB-F4E0-E011-9999-001A92810AA4.root'
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/186/8A8C9F33-3BF3-E011-951D-003048CF99BA.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/104/C078BFCF-30F2-E011-8300-E0CB4E4408C4.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/189/12B3284C-3BF3-E011-94BF-003048F118C2.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/191/54F1521F-40F3-E011-8AA6-BCAEC518FF8E.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/198/20BFEA48-49F3-E011-A2F7-003048F11C58.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/198/2AF282A0-BBF4-E011-8A46-001D09F253C0.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/198/2E2EAEF3-49F3-E011-A3AD-003048F118C2.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/198/AAF80FA6-BBF4-E011-BDB8-001D09F24024.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/020113E7-4FF3-E011-A1E3-001D09F25217.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/021B0678-4DF3-E011-96AB-001D09F2424A.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/04E86175-4DF3-E011-B981-002481E0D7D8.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/1205B758-4EF3-E011-88B1-001D09F253D4.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/128EE876-4DF3-E011-817C-001D09F2915A.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/1E32DB50-4EF3-E011-B2DF-BCAEC5329716.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/1E948CE6-4FF3-E011-9FA6-001D09F2924F.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/26B7C0E6-4FF3-E011-888A-0019B9F581C9.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/3047B9E5-4FF3-E011-B7EE-001D09F24D67.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/36E499E4-4FF3-E011-88A9-003048D2C0F0.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/36EA00E6-4FF3-E011-B929-001D09F253C0.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/3AD089DC-4FF3-E011-9E1A-00237DDC5BBC.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/3C59FC4E-4EF3-E011-AF14-BCAEC5329714.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/44BD3059-4EF3-E011-8692-001D09F24047.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/44C71C4F-4EF3-E011-ABB5-BCAEC5364C4C.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/48080A4F-4EF3-E011-85A0-BCAEC518FF3C.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/4CC55D77-4DF3-E011-AA53-003048D37538.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/4EF1E5E6-4FF3-E011-B691-001D09F252F3.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/62C6FDE5-4FF3-E011-B4B2-001D09F2905B.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/7C410050-4EF3-E011-84D2-BCAEC5329703.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/8281B4E5-4FF3-E011-B754-001D09F24763.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/8441674F-4EF3-E011-83B8-BCAEC518FF7A.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/8E38C450-4EF3-E011-A50C-BCAEC5329718.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/9AAE23E6-4FF3-E011-AC9D-001D09F2983F.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/9E9C6B5B-4BF3-E011-96D6-003048F1182E.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/AC468950-4EF3-E011-A2C0-003048D37456.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/BC487651-4EF3-E011-80D5-BCAEC5364C42.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/C4996AE4-4FF3-E011-B39E-002481E0D790.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/C6612D4F-4EF3-E011-9115-BCAEC518FF41.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/C6C8AEEB-50F3-E011-8995-BCAEC53296FB.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/CC202C4F-4EF3-E011-BB74-BCAEC5329732.root',
##     '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/D88FA5E5-4FF3-E011-8A8B-001D09F290CE.root',
##    '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/178/203/EC11FFE6-4FF3-E011-A79D-001D09F25456.root'
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








                                                     
