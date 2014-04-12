import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiValidation")
process.load("Configuration.StandardSequences.GeometryHCAL_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring("file:RAW.root")
fileNames = cms.untracked.vstring(
                                  )
)

process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigiTester",
    digiLabel = cms.InputTag("hcalDigis"),
    outputFile = cms.untracked.string('HcalDigisValidation.root'),
    hcalselector = cms.untracked.string('all'),
    zside = cms.untracked.string('*'),
    mode = cms.untracked.string('multi'),
    mc   = cms.untracked.string('no') # 'yes' for MC
)

#--- to force RAW->Digi
#process.hcalDigis.InputLabel = 'source'             # data
process.hcalDigis.InputLabel = 'rawDataCollector'  # MC

process.p = cms.Path( process.hcalDigis * process.hcalDigiAnalyzer)
