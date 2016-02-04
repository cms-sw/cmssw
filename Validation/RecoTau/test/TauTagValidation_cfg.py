import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
<<<<<<< TauTagValidation_cfg.py
<<<<<<< TauTagValidation_cfg.py
       '/store/relval/CMSSW_3_1_0_pre6/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0002/FE76B34E-8F32-DE11-8208-000423D9853C.root',
       '/store/relval/CMSSW_3_1_0_pre6/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0002/C4077B7A-1733-DE11-84AA-001617C3B77C.root',
       '/store/relval/CMSSW_3_1_0_pre6/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0002/B4C86624-8B32-DE11-A2B3-000423D98B6C.root',
       '/store/relval/CMSSW_3_1_0_pre6/RelValZTT/GEN-SIM-RECO/STARTUP_31X_v1/0002/4AB04CA2-9032-DE11-AD66-000423D992DC.root'
=======
        '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/0C9E3984-57E8-DD11-B89C-001D09F291D2.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/4CBAC9FC-56E8-DD11-90F5-000423D986C4.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/B0AF7439-57E8-DD11-BC2D-001617C3B77C.root',
        '/store/relval/CMSSW_3_0_0_pre7/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0006/C6AEDDC0-6AE8-DD11-8381-001D09F231B0.root'
>>>>>>> 1.6
=======
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/102A7D74-300A-DE11-B318-000423D6006E.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/C6345BA8-300A-DE11-A5F2-000423D6CA42.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/E4F6CB66-300A-DE11-BAF4-000423D60FF6.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/FE324A2E-800A-DE11-A3A3-000423D99A8E.root'
>>>>>>> 1.10
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
#    input = cms.untracked.int32(100)
)

process.DQMStore = cms.Service("DQMStore")

process.load("Validation.RecoTau.TauTagValidationProducer_cff")
process.load("Validation.RecoTau.TauTagValidation_cfi")
process.load("Validation.RecoTau.RelValHistogramEff_cfi")
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")


###################################################################################################
#
#    Name of the file that gets all the DQMMonitorElements and saves them, please modify accordingly
#
###################################################################################################

process.saveTauEff = cms.EDAnalyzer("DQMSimpleFileSaver",
<<<<<<< TauTagValidation_cfg.py
<<<<<<< TauTagValidation_cfg.py
  outputFileName = cms.string('CMSSW_3_1_0_pre6_ZTT.root')
=======
  outputFileName = cms.string('CMSSW_3_0_0_pre7_tauGenJets_TEST.root')
>>>>>>> 1.6
=======
  outputFileName = cms.string('CMSSW_3_1_0_pre3_tauGenJets.root')
>>>>>>> 1.10
)
                                 
process.p = cms.Path(
    process.PFTau+
    process.tauGenJetProducer +
    process.tauTagValidationWithTanc +
    process.TauEfficiencies +
    process.saveTauEff)

process.schedule = cms.Schedule(process.p)
