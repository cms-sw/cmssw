import FWCore.ParameterSet.Config as cms

process = cms.Process("mtdSimHitsValidation")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtended2023D38_cff")

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi")
process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdGeometry_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cfi")

process.mtdGeometry = cms.ESProducer("MTDDigiGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.MessageLogger.cerr.FwkReport  = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100),
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

# --- BTL Validation
process.load("Validation.MtdValidation.btlSimHits_cfi")
process.load("Validation.MtdValidation.btlDigiHits_cfi")
process.load("Validation.MtdValidation.btlRecHits_cfi")
btlValidation = cms.Sequence(process.btlSimHits + process.btlDigiHits + process.btlRecHits)

# --- ETL Validation
process.load("Validation.MtdValidation.etlSimHits_cfi")
process.load("Validation.MtdValidation.etlDigiHits_cfi")
process.load("Validation.MtdValidation.etlRecHits_cfi")
etlValidation = cms.Sequence(process.etlSimHits + process.etlDigiHits + process.etlRecHits)

process.DQMStore = cms.Service("DQMStore")

process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")

process.p = cms.Path( btlValidation + etlValidation + process.dqmSaver )
