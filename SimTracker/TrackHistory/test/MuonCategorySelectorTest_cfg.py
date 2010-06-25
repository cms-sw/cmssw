import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonHistoryTest")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.MuonClassifier_cff")

from SimTracker.TrackHistory.CategorySelectors_cff import *

process.muonSelector = MuonCategorySelector(
    src = cms.InputTag('standAloneMuons'),
    cut = cms.string("is('FromChargePionMuon') || is('FromChargeKaonMuon')"),
)

process.muonHistoryAnalyzer = cms.EDAnalyzer("TrackHistoryAnalyzer",
    process.MuonClassifier,
    pset = process.MuonClassifier
)

process.muonHistoryAnalyzer.trackProducer = 'muonSelector'

process.GlobalTag.globaltag = 'MC_36Y_V7A::All'

process.p = cms.Path(process.playback * process.muonSelector * process.muonHistoryAnalyzer)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(200) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0007/F0BFE5CD-0F38-DF11-96B4-003048678BAA.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/F6B3EDA0-BD37-DF11-AF4A-0026189438D3.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/F06E2154-C237-DF11-A470-003048678FB8.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/ECB4C50A-BC37-DF11-964C-0018F3D09600.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/54050715-BD37-DF11-BB52-002618943969.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/482C601C-BF37-DF11-922E-0026189437F5.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/3224259E-BE37-DF11-9C9A-003048678FF2.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/2EC6285C-C337-DF11-BEB1-001A92971B64.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-RECO/MC_3XY_V25-v1/0006/021F29CA-C337-DF11-8CC9-001A92971B68.root' ] );

secFiles.extend( [
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/E40A1EDF-C337-DF11-85F0-002618943914.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/D6ECDAD3-C537-DF11-BF18-003048678F6C.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/D468599F-C037-DF11-8347-00304867915A.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/BAD0BF11-BD37-DF11-AA5E-00304867D446.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/B6ADA7A5-BD37-DF11-AC85-0026189438B4.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/9CAD2294-BE37-DF11-A722-001A92810AE6.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/8C82A38F-BE37-DF11-B4D5-00261894391B.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/86E02A4D-C237-DF11-A946-003048678FA0.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/6C8362A5-BD37-DF11-AD4D-001BFCDBD190.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/60A58719-BE37-DF11-A7A8-002618943972.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/5E27BFCF-C337-DF11-8ED9-00248C55CC62.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/5C3FD9C8-C237-DF11-8EEF-0018F3D09612.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/4A69B14B-C337-DF11-B83C-001A92810AE6.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/3CC0955A-C337-DF11-8582-001A928116EC.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/2E5A0502-BC37-DF11-AF06-0026189438B3.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/14BEDF1B-BF37-DF11-AE8F-0026189438ED.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/1412A2A2-BD37-DF11-ADA7-001A928116E0.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/10B6E600-BC37-DF11-BE0C-002618943953.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/0CEFAA10-BD37-DF11-9023-00248C0BE013.root',
       '/store/relval/CMSSW_3_5_5/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V25-v1/0006/0A8E4683-BC37-DF11-9FE2-00248C0BE005.root' ] );


