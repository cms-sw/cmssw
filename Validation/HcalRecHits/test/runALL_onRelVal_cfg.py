import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitsValidationRelVal")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/00BCD825-6E57-DD11-8C1F-000423D98EA8.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/0A1018B5-6E57-DD11-83DF-000423D6C8EE.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/0CE8BD2B-6E57-DD11-94C3-001617C3B70E.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/1A24E712-6E57-DD11-A15C-000423D98834.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/2E660F31-6D57-DD11-B781-000423D94534.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/307FF178-6E57-DD11-B085-000423D9880C.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/3E0D47AA-6E57-DD11-B760-000423D99264.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/4E251861-6E57-DD11-988F-001617C3B77C.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/6623B35A-6E57-DD11-A234-000423D986A8.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/688D5EE4-6D57-DD11-B921-000423D6B358.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/6A3DD0CF-6D57-DD11-8861-001617C3B77C.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/701FEC0C-6E57-DD11-A623-000423D991F0.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/742325B2-6E57-DD11-90D1-000423D6AF24.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/82769E01-6D57-DD11-8418-000423DD2F34.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/8A650215-6D57-DD11-8F8C-000423D6BA18.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/9CF4037C-6D57-DD11-9D22-000423D9517C.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/B420F018-6E57-DD11-8E96-000423D98804.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/C800B9FD-6C57-DD11-B288-000423D98BE8.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/C8060E10-6E57-DD11-AD31-000423D6C8EE.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/CC91E017-6E57-DD11-8F25-001617C3B6FE.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/D6E0FDE2-6D57-DD11-A2D1-000423D98800.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/EC88274B-6E57-DD11-B8CD-000423D991D4.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/ECDB23F3-6C57-DD11-AD1F-001617C3B76E.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/EE228395-6D57-DD11-9386-000423D94908.root',
       '/store/relval/2008/7/21/RelVal-RelValTTbar-1216579481-IDEAL_V5-2nd/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/FA21A44A-6E57-DD11-A92F-000423D9517C.root'
      )
)

process.hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
    eventype = cms.untracked.string('multi'),
    outputFile = cms.untracked.string('HcalRecHitsValidationALL_RelVal.root'),
    ecalselector = cms.untracked.string('yes'),
    mc = cms.untracked.string('no'),
    hcalselector = cms.untracked.string('all')
)


process.p = cms.Path(process.hcalRecoAnalyzer)

