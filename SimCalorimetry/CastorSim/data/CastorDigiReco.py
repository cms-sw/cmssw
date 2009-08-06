import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDIGIVALIDATION")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("SimCalorimetry.CastorSim.castordigi_cfi")

process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("CondCore.DBCommon.CondDBCommon_cfi")


process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(100),
    fileNames = cms.untracked.vstring('file:simevent.root')
)


process.es_pool = cms.ESSource(
     "PoolDBESSource",
     process.CondDBSetup,
     timetype = cms.string('runnumber'),
     connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_COND_30X_HCAL'),
     authenticationMethod = cms.untracked.uint32(0),
     toGet = cms.VPSet(
         cms.PSet(
             record = cms.string('CastorPedestalsRcd'),
             tag = cms.string('castor_pedestals_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorPedestalWidthsRcd'),
             tag = cms.string('castor_widths_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorGainsRcd'),
             tag = cms.string('castor_gains_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorQIEDataRcd'),
             tag = cms.string('castor_qie_v1.0_test')
             ),
         cms.PSet(
             record = cms.string('CastorElectronicsMapRcd'),
             tag = cms.string('castor_emap_v1.0_test')
             )
     )
)


process.digiDumper = cms.EDFilter("HcalDigiDump")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simCastorDigis = cms.untracked.uint32(1)
    )
)

process.hitDumper = cms.EDFilter("HcalRecHitDump")

process.hitAnalyzer = cms.EDAnalyzer("CastorHitAnalyzer")

process.digiAnalyzer = cms.EDAnalyzer("CastorDigiAnalyzer")

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('./digiout.root')
)

process.p = cms.Path(process.mix*process.simCastorDigis*process.castorreco*process.hitAnalyzer*process.hitDumper*process.digiAnalyzer)
process.outpath = cms.EndPath(process.o1)

