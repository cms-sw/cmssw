import FWCore.ParameterSet.Config as cms

process = cms.Process("TPDBAn")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

#----------
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')
#-----------

process.ecalTPConditions = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGPedestalsRcd'),
        tag = cms.string('EcalTPGPedestals_beamv5_startup_mc')
    ), 
        cms.PSet(
            record = cms.string('EcalTPGLinearizationConstRcd'),
            tag = cms.string('EcalTPGLinearizationConst_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGSlidingWindowRcd'),
            tag = cms.string('EcalTPGSlidingWindow_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
            tag = cms.string('EcalTPGFineGrainEBIdMap_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainStripEERcd'),
            tag = cms.string('EcalTPGFineGrainStripEE_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainTowerEERcd'),
            tag = cms.string('EcalTPGFineGrainTowerEE_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutIdMapRcd'),
            tag = cms.string('EcalTPGLutIdMap_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightIdMapRcd'),
            tag = cms.string('EcalTPGWeightIdMap_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGWeightGroupRcd'),
            tag = cms.string('EcalTPGWeightGroup_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGLutGroupRcd'),
            tag = cms.string('EcalTPGLutGroup_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGFineGrainEBGroupRcd'),
            tag = cms.string('EcalTPGFineGrainEBGroup_beamv5_startup_mc')
        ), 
        cms.PSet(
            record = cms.string('EcalTPGPhysicsConstRcd'),
            tag = cms.string('EcalTPGPhysicsConst_beamv5_startup_mc')
        ),
	cms.PSet(
	    record = cms.string('EcalTPGSpikeRcd'),
	    tag = cms.string('EcalTPGSpike_beamv5_startup_mc')
	)
	#,
	#cms.PSet(
        #    record = cms.string('EcalTPGCrystalStatusRcd'),
        #    tag = cms.string('EcalTPGCrystalStatus_beamv5_startup_mc')
        #),
	#cms.PSet(
        #    record = cms.string('EcalTPGTowerStatusRcd'),
        #    tag = cms.string('EcalTPGTowerStatus_beamv5_startup_mc')
        #)
	),
    messagelevel = cms.untracked.uint32(3),
    timetype = cms.string('runnumber'),
#    connect = cms.string('oracle://ecalh4db/TEST02'),
#    connect = cms.string('sqlite_file:../../../CalibCalorimetry/EcalTPGTools/test/DB_beamv5_test_mc.db'),
    #connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_ECAL'),
    connect = cms.string('frontier://FrontierPrep/CMS_COND_ECAL'),
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True)
)

process.tpDBAnalyzer = cms.EDAnalyzer("EcalTPCondAnalyzer")

process.p = cms.Path(process.tpDBAnalyzer)
