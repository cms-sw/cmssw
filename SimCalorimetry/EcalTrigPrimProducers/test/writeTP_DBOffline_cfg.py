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


process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_readDBOffline_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/uberthon/tpg/elec_unsupp_pt10-100.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")


process.ecalTPConditions = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        tag = cms.string('EcalTPGPedestals_craft')
    ), 
        cms.PSet(
            record = cms.string('EcalTPGLinearizationConstRcd'),
            tag = cms.string('EcalTPGLinearizationConst_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGSlidingWindowRcd'),
            tag = cms.string('EcalTPGSlidingWindow_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
            tag = cms.string('EcalTPGFineGrainEBIdMap_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainStripEERcd'),
            tag = cms.string('EcalTPGFineGrainStripEE_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainTowerEERcd'),
            tag = cms.string('EcalTPGFineGrainTowerEE_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutIdMapRcd'),
            tag = cms.string('EcalTPGLutIdMap_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightIdMapRcd'),
            tag = cms.string('EcalTPGWeightIdMap_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightGroupRcd'),
            tag = cms.string('EcalTPGWeightGroup_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutGroupRcd'),
            tag = cms.string('EcalTPGLutGroup_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBGroupRcd'),
            tag = cms.string('EcalTPGFineGrainEBGroup_craft')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGPhysicsConstRcd'),
            tag = cms.string('EcalTPGPhysicsConst_craft')
        ),
	cms.PSet(    
	    record = cms.string('EcalTPGCrystalStatusRcd'),
	    tag = cms.string('EcalTPGCrystalStatus_craft')
	),
	cms.PSet(    
	    record = cms.string('EcalTPGTowerStatusRcd'),
	    tag = cms.string('EcalTPGTowerStatus_craft')
	)),    
    messagelevel = cms.untracked.uint32(3),
    timetype = cms.string('runnumber'),
#    connect = cms.string('oracle://ecalh4db/TEST02'),
    connect = cms.string('sqlite_file:DB_craft.db'),
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True)
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
    fileName = cms.untracked.string('TrigPrim_Em_DBOffline30x.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalTPG = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('simEcalTriggerPrimitiveDigis')
)

process.p = cms.Path(process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.out)


