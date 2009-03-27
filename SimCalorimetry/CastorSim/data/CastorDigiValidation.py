import FWCore.ParameterSet.Config as cms

process = cms.Process("CASTORDIGIVALIDATION")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("SimCalorimetry.CastorSim.castordigi_cfi")

process.load("RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi")

process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")

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

process.castor_db_producer = cms.ESProducer("CastorDbProducer",
    dump = cms.untracked.vstring('Pedestals', 
        'PedestalWidths', 
        'Gains', 
        'GainWidths', 
        'QIEData', 
        'ChannelQuality', 
        'ElectronicsMap'),
    file = cms.untracked.string('/tmp/katsas/dump_castor.txt')
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ElectronicsMap'),
        file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_emap_tb08.txt')
    ), 
        cms.PSet(
            object = cms.string('Pedestals'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_pedestals_test.txt')
        ), 
        cms.PSet(
            object = cms.string('Gains'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_gains_TB_cor.txt')
        ), 
        cms.PSet(
            object = cms.string('QIEData'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_qie_test2.txt')
        ), 
        cms.PSet(
            object = cms.string('ChannelQuality'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_quality.txt')
        ), 
        cms.PSet(
            object = cms.string('GainWidths'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_gain_widths_test.txt')
        ), 
        cms.PSet(
            object = cms.string('PedestalWidths'),
            file = cms.FileInPath('SimCalorimetry/CastorSim/data/castor_pedestal_widths_test.txt')
        ))
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

