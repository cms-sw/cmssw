import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load('Configuration/StandardSequences/Digi_cff')

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_30X::All'


process.load("Validation.TrackerDigis.trackerDigisValidation_cff")
process.pixelDigisValid.outputFile="pixeldigihisto.root"
process.stripDigisValid.outputFile="stripdigihisto.root"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/38E34C97-E8DD-DD11-8327-000423D94534.root'

))

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.digis = cms.Sequence(process.trDigi*process.trackerDigisValidation)
process.p1 = cms.Path(process.mix*process.digis)


