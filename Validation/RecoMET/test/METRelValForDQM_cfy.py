import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("RecoMET.Configuration.CaloTowersOptForMET_cff")

process.load("RecoMET.Configuration.RecoMET_cff")

process.load("RecoMET.Configuration.RecoHTMET_cff")

process.load("RecoMET.Configuration.RecoGenMET_cff")

process.load("RecoMET.Configuration.GenMETParticles_cff")

process.load("RecoMET.Configuration.RecoPFMET_cff")

process.load("RecoJets.Configuration.CaloTowersRec_cff")

process.load("Validation.RecoMET.METRelValForDQM_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(

#        '/store/relval/CMSSW_3_1_0_pre2/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0000/3AC3EF48-8203-DE11-AC5A-001617DC1F70.root'
         '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/083D0102-5C03-DE11-94D4-001617E30CC8.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/189440EA-5903-DE11-B00F-001617E30F48.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/2EAB172D-5903-DE11-AE3C-000423D174FE.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/52AB7EAA-5003-DE11-B6FA-0030487A322E.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/52FD5AF7-5703-DE11-A2FA-0030487A1FEC.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/60DCC57D-5B03-DE11-9C3A-000423D9880C.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/66599CB1-5A03-DE11-8111-001D09F2543D.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/C81DD1DC-5603-DE11-B464-000423D98EA8.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0000/FA3055B4-5B03-DE11-8279-000423D99AA2.root',
                '/store/relval/CMSSW_3_1_0_pre2/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_30X_v1/0001/480BEA03-DB03-DE11-9BB0-000423D99AA2.root'
        
    )


)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


process.fileSaver = cms.EDFilter("METFileSaver",
                                 OutputFile = cms.untracked.string('METTester_data_QCD_3000-3500.root')
) 
process.p = cms.Path(process.fileSaver*
#                     process.genMetTrue*
#                     process.genMetCalo*
#                     process.genMetCaloAndNonPrompt*
#                     process.tcMet*
                     process.METRelValSequence
)
process.schedule = cms.Schedule(process.p)


