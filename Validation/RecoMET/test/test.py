import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("RecoMET.Configuration.CaloTowersOptForMET_cff")

process.load("RecoMET.Configuration.RecoMET_cff")

process.load("RecoMET.Configuration.RecoHTMET_cff")

process.load("RecoMET.Configuration.RecoGenMET_cff")

process.load("RecoMET.Configuration.GenMETParticles_cff")

process.load("RecoMET.Configuration.RecoPFMET_cff")

process.load("RecoJets.Configuration.CaloTowersRec_cff")

process.load("Validation.RecoMET.CaloMET_cff")

process.load("Validation.RecoMET.GenMET_cff")

process.load("Validation.RecoMET.HTMET_cff")

process.load("Validation.RecoMET.GenMETFromGenJets_cff")
process.load("DQMOffline.JetMET.caloTowers_cff")
process.towerSchemeBAnalyzer.FineBinning = cms.untracked.bool(True)
process.towerSchemeBAnalyzer.FolderName =  cms.untracked.string("RecoMETV/MET_CaloTowers/SchemeB")
process.towerOptAnalyzer.FineBinning = cms.untracked.bool(True)
process.towerOptAnalyzer.FolderName =  cms.untracked.string("RecoMETV/MET_CaloTowers/Optimized")


process.load("DQMOffline.JetMET.RecHits_cff")
process.ECALAnalyzer.FineBinning = cms.untracked.bool(True)
process.ECALAnalyzer.FolderName =  cms.untracked.string("RecoMETV/MET_ECAL/data")
process.HCALAnalyzer.FineBinning = cms.untracked.bool(True)
process.HCALAnalyzer.FolderName =  cms.untracked.string("RecoMETV/MET_HCAL/data")


process.load("Validation.RecoMET.PFMET_cff")

process.load("Validation.RecoMET.TCMET_cff")

process.load("Validation.RecoMET.MuonCorrectedCaloMET_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = cms.string("MC_31X_V8::All")

process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")

process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(

    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/CA73755A-84B5-DE11-B054-000423D98634.root',
    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/FA75655E-FAB4-DE11-8741-001D09F2462D.root',
    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/DC5EB98F-FAB4-DE11-B072-001D09F25208.root',
    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/B018653F-02B5-DE11-B0E9-000423D98EC4.root',
    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/A66A6322-06B5-DE11-B986-000423D98EC4.root',
    '/store/relval/CMSSW_3_4_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/68579E06-FCB4-DE11-A4F6-000423D990CC.root'
    
    )
                            

)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(8000) )


process.fileSaver = cms.EDFilter("METFileSaver",
    OutputFile = cms.untracked.string('METTester_data_QCD_Pt_80_120.root')
) 
process.p = cms.Path(process.fileSaver*
                     process.pfMet*
                     process.calotoweroptmaker*
                     process.analyzeRecHits*
                     process.analyzecaloTowers*
                     process.analyzeGenMET*
                     process.analyzeGenMETFromGenJets*
                     process.analyzeHTMET*
                     process.analyzeCaloMET*
                     process.analyzePFMET*
                     process.analyzeTCMET*
                     process.analyzeMuonCorrectedCaloMET
)
process.schedule = cms.Schedule(process.p)


