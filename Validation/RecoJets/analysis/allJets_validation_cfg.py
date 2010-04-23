import FWCore.ParameterSet.Config as cms

#from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
#from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

#process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
#
#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

# check # of bins
#process.load("DQMServices.Components.DQMStoreStats_cfi")

#process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
#process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(2)
)

process.source = cms.Source("PoolSource",
#    debugFlag = cms.untracked.bool(True),
#    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/FC439FCE-D34C-DF11-8512-003048678B16.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/EEF3259E-D14C-DF11-ACC8-003048678BE6.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/EEC9ABA5-D24C-DF11-9064-003048678F84.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/DEAF1DA0-D14C-DF11-A4DC-003048678FE0.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/D08AEF28-D24C-DF11-BAF4-003048D42DC8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/CC2507B1-D24C-DF11-B05D-001A928116EC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/C673EAA3-D24C-DF11-88BF-002354EF3BDE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/BEE9E42C-D24C-DF11-A459-001A928116EC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/BCED4128-D24C-DF11-8D5B-003048678B3C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/B829CB34-D24C-DF11-BA25-002618943821.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/B213C934-D24C-DF11-BC10-00304867902E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/B08A9327-D24C-DF11-A915-001A92971B94.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/B07D2FA7-D24C-DF11-AED8-003048678C06.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/AC543736-D24C-DF11-A7C0-0018F3D096C8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/96704B64-DA4C-DF11-B7FD-00248C55CC7F.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/943B292F-D24C-DF11-928A-003048679044.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/8C7D1A27-D24C-DF11-A08A-001A928116CE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/8ABC6837-D24C-DF11-B17D-001A928116EC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/8A0B2BA9-D14C-DF11-B85A-0026189438C1.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/849E6932-D24C-DF11-B74E-0026189438B9.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/7E0F15A4-D14C-DF11-856C-00304867C1BC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/7445F032-D24C-DF11-A843-0026189438FE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/5CC78CAA-D14C-DF11-AF31-00261894388F.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/5C48692E-D24C-DF11-B0AF-0026189438D4.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/56296732-D24C-DF11-AA1D-00248C0BE012.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/3E54F0BF-E94C-DF11-98EC-001BFCDBD100.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/2CB98662-DB4C-DF11-BEBC-003048678F9C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/240F3925-D24C-DF11-ADF7-00261894386F.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/20C4F9CF-D34C-DF11-8985-003048678FB8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/1EC22A36-D24C-DF11-A75D-003048D15E14.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/1CA92D9F-D14C-DF11-96C7-00261894393E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/1A880BCE-D34C-DF11-8909-00304867900C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/18E3152B-D24C-DF11-9561-00261894388F.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/0E8EC629-D24C-DF11-82F5-002618943882.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/0E719044-D34C-DF11-954B-00261894395A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/0CEF4730-D24C-DF11-9E9B-003048678B7C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0015/0270712B-D24C-DF11-936E-0026189438CF.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/F8E80F06-D14C-DF11-94C7-003048678F74.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/F8A4D899-D14C-DF11-B559-002618943914.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/F626779E-D14C-DF11-88D7-0030486790FE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/E666EC99-D14C-DF11-B9B9-003048678B5E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/E29242F4-D04C-DF11-ACBF-003048D3C010.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/DE2FB2B3-D14C-DF11-B262-0030486792AC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/D6389E95-D04C-DF11-9EC8-00304867920A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/CADE1BA5-D14C-DF11-9C21-0026189438AA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/CACC45AB-D14C-DF11-B5FF-003048679044.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/CA322AA2-D14C-DF11-B81A-0018F3D0968E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/BEAD22F5-D04C-DF11-B696-002618FDA216.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/BA545F9C-D14C-DF11-8964-002618943969.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/B8604170-D04C-DF11-8DD9-002618943845.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/B65F7098-D14C-DF11-B6BE-00304867BFBC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/B60E5777-D04C-DF11-9021-002618943867.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/B46D88A1-D14C-DF11-9F51-002618943973.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/A04D3589-D04C-DF11-9022-003048678B38.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/9EE45B9A-D14C-DF11-AD8D-0026189438F7.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/9E418173-D04C-DF11-BFA9-0026189438F3.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/9E11ACA4-D14C-DF11-B96D-00304867915A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/9CDD68F6-D04C-DF11-8C1B-002618943833.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/9A0EEFB3-D14C-DF11-B709-003048679162.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/969C3D91-D04C-DF11-87AA-00304867906C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/964B17F4-D04C-DF11-A009-00261894392F.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/8CDD8304-D14C-DF11-B5AB-0026189438D7.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/8A7AA3FE-D04C-DF11-9FEA-003048678B38.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/860069E0-CF4C-DF11-98A3-001A928116CE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/84B81EAA-D14C-DF11-B8A7-001A92971B94.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/82CB42D1-CF4C-DF11-B442-002618943973.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/7CB2AFFA-D04C-DF11-8FC6-0026189438ED.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/78C746F4-CF4C-DF11-8C4C-00304867908C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/78B51FFF-D04C-DF11-AA51-003048678BEA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/76C3EEF6-D04C-DF11-BB5C-0018F3D0960A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/6AD18E8C-D04C-DF11-B525-0030486790A6.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/6ACD4191-D14C-DF11-964D-002618943834.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/5E93EFA1-D14C-DF11-A8C9-003048678FDE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/5CCE7C04-D14C-DF11-B85B-0030486792DE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/5C775D80-D04C-DF11-9C2D-0026189438B8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/5AD82797-D14C-DF11-A8D0-00248C55CC97.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/5A68829D-D14C-DF11-8F30-003048678B8E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/583113A5-D14C-DF11-B09D-002618943957.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/561DA600-D14C-DF11-A266-003048678AC8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/52479C98-D14C-DF11-B29B-003048679294.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/523C479E-D14C-DF11-8A24-001A928116CE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/4E6DEFE3-CF4C-DF11-88F8-003048678CA2.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/4C73596B-D04C-DF11-B396-0026189438FD.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/4A845ED5-CF4C-DF11-A4B2-0026189438E4.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/42E811A1-D14C-DF11-AD18-003048678FEA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/3A0E6379-D04C-DF11-8C3F-0026189438EF.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/36D8A5F7-D04C-DF11-88DA-002618943953.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/3037C696-D14C-DF11-99FC-003048679180.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/2CA13391-D14C-DF11-837D-0026189438E7.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/28F893F2-D04C-DF11-B835-002354EF3BDC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/2887A19B-D14C-DF11-9756-0018F3D09624.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/1C577397-D14C-DF11-BF12-00304867908C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/18CBB6E0-CF4C-DF11-ACBF-0018F3D096F6.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/18496896-D14C-DF11-83AB-002618943833.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/18100A01-D14C-DF11-8623-003048679080.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/14BAB398-D04C-DF11-A30A-0030486792B4.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/1064B4FB-D04C-DF11-A56F-0018F3D096E4.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/0CD81EF9-D04C-DF11-B990-00261894390C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/06E96084-D04C-DF11-9DB2-002354EF3BDC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/MC_37Y_V0_FastSim-v1/0014/003715F1-D04C-DF11-A528-00261894393C.root'

    )

)

process.fileSaver = cms.EDAnalyzer("JetFileSaver",
                                 OutputFile = cms.untracked.string('histo.root')
)

## Test for corrected jets - available only for 
#process.prefer("L2L3CorJetIC5Calo")

#process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeCone5CaloJets"),
#    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
#)


## AK5 Corrected jets
process.JetAnalyzerAK5Cor = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("L2L3CorJetAK5Calo"),
    srcGen = cms.InputTag("ak5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)


### IC5 JPT jets
#JetAnalyzerIC5JPT = cms.EDFilter("CaloJetTester",
#    src = cms.InputTag("ic5JPTJetsL2L3"),
#    srcGen = cms.InputTag("iterativeCone5GenJets"),
##    TurnOnEverything = cms.untracked.string('yes'),
##    TurnOnEverything = cms.untracked.string('no'),
##    outputFile = cms.untracked.string('histo.root'),
##    outputFile = cms.untracked.string('test.root'),
#    genEnergyFractionThreshold = cms.double(0.05),
#    genPtThreshold = cms.double(1.0),
#    RThreshold = cms.double(0.3),
#    reverseEnergyFractionThreshold = cms.double(0.5)
#)

### AntiKt5 JPT jets
#JetAnalyzerAk5JPT = cms.EDFilter("CaloJetTester",
#    src = cms.InputTag("ak5JPTJetsL2L3"),
#    srcGen = cms.InputTag("ak5GenJets"),
##    TurnOnEverything = cms.untracked.string('yes'),
##    TurnOnEverything = cms.untracked.string('no'),
##    outputFile = cms.untracked.string('histo.root'),
##    outputFile = cms.untracked.string('test.root'),
#    genEnergyFractionThreshold = cms.double(0.05),
#    genPtThreshold = cms.double(1.0),
#    RThreshold = cms.double(0.3),
#    reverseEnergyFractionThreshold = cms.double(0.5)
#)

process.p1 = cms.Path(process.fileSaver
                      #--- Non-Standard sequence (that involve Producers)
                      *process.L2L3CorJetAK5Calo
 #                     *process.ZSPJetCorrectionsIcone5
 #                     *process.ZSPJetCorrectionsAntiKt5
 #                     *process.JetPlusTrackCorrectionsIcone5
 #                     *process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerAK5Cor
#                      *process.JetAnalyzerIC5JPT
#                      *process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation
                      #--- DQM stats module
#                      *process.dqmStoreStats
)
process.DQM.collectorHost = ''

