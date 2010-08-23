import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.Reconstruction_cff")

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
process.load("RecoMET/Configuration/RecoMET_BeamHaloId_cff")
process.DQMStore = cms.Service("DQMStore")
process.load("DQMOffline/JetMET/BeamHaloAnalyzer_cfi")

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

#process.GlobalTag.globaltag = cms.string("MC_31X_V8::All")
#process.GlobalTag.globaltag = cms.string("MC_3XY_V12::All")   
process.GlobalTag.globaltag = cms.string("MC_38Y_V9::All")

process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")

process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

    #'/store/relval/CMSSW_3_7_0_pre2/RelValZMM/GEN-SIM-RECO/START37_V1-v1/0017/78448BF0-A252-DF11-B3DC-00261894392B.root'
    '/store/relval/CMSSW_3_9_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0014/14B76A1A-FEA7-DF11-8046-00261894384F.root'
    )
                            

)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(8000) )


process.fileSaver = cms.EDAnalyzer("METFileSaver",
    OutputFile = cms.untracked.string('Test.root') )
process.p = cms.Path(process.fileSaver*
                     process.calotoweroptmaker*
                     process.calotoweroptmakerWithHO*
                     process.towerMakerWithHO*
                     process.genJetParticles*
                     process.genMETParticles*
                     process.metreco*
                     process.analyzeRecHits*
                     process.analyzecaloTowers*
                     process.analyzeGenMET*
                     process.analyzeGenMETFromGenJets*
                     process.analyzeHTMET*
                     process.analyzeCaloMET*
                     process.analyzePFMET*
                     process.analyzeTCMET*
                     process.analyzeMuonCorrectedCaloMET*
                     process.AnalyzeBeamHalo
)
process.schedule = cms.Schedule(process.p)


