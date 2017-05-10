import FWCore.ParameterSet.Config as cms

process = cms.Process("PROTPGD")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

# IdealGeometryRecord
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")


#process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_readDBOffline_cff")
#process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_GT_cff")
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_DB_cff")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
connect = cms.string('frontier://FrontierProd/CMS_COND_34X_ECAL'),
process.GlobalTag.globaltag = "START61_V11::All"
process.prefer("GlobalTag")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_1_0_pre6/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS171_V5-v1/00000/5A95EC7E-25C7-E311-B160-003048D15E2C.root'
    #'/store/relval/CMSSW_7_1_0_pre6/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS171_V5-v1/00000/F42142AD-25C7-E311-A8DC-0025905A609A.root'
    #'/store/relval/CMSSW_4_3_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V1-v1/0050/025B2A70-D672-E011-A8E1-003048679076.root',
    #'/store/relval/CMSSW_4_3_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V1-v1/0050/0A98BD7A-D572-E011-9301-0018F3D09630.root',
    #'/store/relval/CMSSW_4_3_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V1-v1/0050/1839EBF9-D372-E011-BDFD-001A92971AD8.root',
    #'/store/relval/CMSSW_4_3_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V1-v1/0050/30F39E6F-CE72-E011-9600-002618FDA250.root',
    #'/store/relval/CMSSW_4_3_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V1-v1/0050/48E66ADB-CC72-E011-A887-003048678B8E.root'
    
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

        
    
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*', 
        'keep *_ecalDigis_*_*', 
        'keep *_ecalRecHit_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep PCaloHits_*_EcalHitsEB_*', 
        'keep PCaloHits_*_EcalHitsEE_*', 
        'keep edmHepMCProduct_*_*_*'),
    fileName = cms.untracked.string('/tmp/ebecheva/TrigPrim_Em_DBOffline3.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('simEcalTriggerPrimitiveDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalTPG = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        )
    ),
    categories = cms.untracked.vstring('EcalTPG'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.out)


