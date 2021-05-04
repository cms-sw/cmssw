import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('mtdValidation',Phase2C9)


process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtended2026D49_cff")

process.load('SimGeneral.MixingModule.mixNoPU_cfi')

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

process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)

# --- BTL Validation
process.load("Validation.MtdValidation.btlSimHits_cfi")
process.load("Validation.MtdValidation.btlDigiHits_cfi")
process.load("Validation.MtdValidation.btlLocalReco_cfi")
btlValidation = cms.Sequence(process.btlSimHits + process.btlDigiHits + process.btlLocalReco)

# --- ETL Validation
process.load("Validation.MtdValidation.etlSimHits_cfi")
process.load("Validation.MtdValidation.etlDigiHits_cfi")
process.load("Validation.MtdValidation.etlLocalReco_cfi")
etlValidation = cms.Sequence(process.etlSimHits + process.etlDigiHits + process.etlLocalReco)

# --- Global Validation
process.load("Validation.MtdValidation.mtdTracks_cfi")

process.DQMStore = cms.Service("DQMStore")

process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")

process.p = cms.Path( process.mix + btlValidation + etlValidation + process.globalReco + process.dqmSaver)
