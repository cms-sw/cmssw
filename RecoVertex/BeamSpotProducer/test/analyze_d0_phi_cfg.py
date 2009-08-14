import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0003/08C63285-F66B-DE11-84EA-001D09F2438A.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/F0EA2A75-E36B-DE11-AC89-001D09F2512C.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/DCCD9F1B-EA6B-DE11-A6C0-000423D6CA42.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/B6B3FCA7-EB6B-DE11-81A7-000423D99CEE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/72060DE0-E46B-DE11-9DE8-001D09F29619.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/5C9FA456-EF6B-DE11-A71D-001D09F2905B.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/52BFCA85-E36B-DE11-A1B5-001D09F2AD84.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V2-v2/0002/08E55DDA-E36B-DE11-8778-001D09F24353.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1500)
)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']
process.d0_phi_analyzer.OutputFileName = 'EarlyCollision.root'


