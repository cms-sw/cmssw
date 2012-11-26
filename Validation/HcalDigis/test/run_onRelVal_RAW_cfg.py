import FWCore.ParameterSet.Config as cms

process = cms.Process("DigiValidation")
process.load("Configuration.StandardSequences.GeometryHCAL_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
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
'/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0004/746D2F6B-1F74-DF11-B664-001A928116B0.root',
        '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/F2181911-EE73-DF11-86BC-0030486791BA.root',
        '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/F063C904-F273-DF11-831C-001A92971B64.root',
        '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/D4B25A5D-F373-DF11-99E7-00304867918E.root',
        '/store/relval/CMSSW_3_8_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V0-v1/0003/8AA2EF2A-F073-DF11-82F7-001A92971B32.root'
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
