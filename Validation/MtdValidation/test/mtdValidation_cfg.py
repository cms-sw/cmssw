import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('mtdValidation',Phase2C11I13M9)


process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryExtended2026D76Reco_cff")

process.load('SimGeneral.MixingModule.mixNoPU_cfi')

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

process.btlDigiHits.LocalPositionDebug = True
process.etlDigiHits.LocalPositionDebug = True
process.btlLocalReco.LocalPositionDebug = True
process.etlLocalReco.LocalPositionDebug = True

process.DQMStore = cms.Service("DQMStore")

process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")

process.p = cms.Path( process.mix + btlValidation + etlValidation + process.mtdTracks + process.dqmSaver)
