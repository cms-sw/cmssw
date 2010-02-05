import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/FCDC1618-86B4-DE11-9E2A-000423D991F0.root',
#        '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/B86D4330-81B4-DE11-80F9-001D09F2512C.root',
#        '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/8881D8D8-82B4-DE11-B6EE-000423D94990.root',
#        '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/803AD5A0-7BB4-DE11-843E-001D09F2AD4D.root',
#        '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/56316077-7FB4-DE11-ABA0-001D09F2437B.root'

#'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y250-v1/0007/E06C2C15-69B4-DE11-97CA-001D09F28E80.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y250-v1/0007/A8C866B1-6EB4-DE11-97DE-000423D9997E.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y250-v1/0007/36335C47-BDB4-DE11-BD46-001D09F2AF96.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y250-v1/0007/2E14D0E7-6BB4-DE11-A959-000423D98B6C.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y250-v1/0007/04ABCE4B-65B4-DE11-8F2C-001D09F24024.root'

#'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y500-v1/0007/CEC22971-7BB4-DE11-BF07-001D09F2305C.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y500-v1/0007/785B0D8A-7DB4-DE11-AD48-001D09F290BF.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y500-v1/0007/6AE208D0-77B4-DE11-944B-001D09F24FBA.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y500-v1/0007/226399E3-BDB4-DE11-A2F1-000423D8F63C.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y500-v1/0007/08BFA188-79B4-DE11-831D-001D09F2424A.root'

#'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y1000-v1/0007/AC6D67FF-76B4-DE11-B83E-001D09F2906A.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y1000-v1/0007/7CE021E7-73B4-DE11-ACF8-000423D99A8E.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y1000-v1/0007/5EC6B9F8-75B4-DE11-AE6F-001D09F2983F.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y1000-v1/0007/2A5B6842-BDB4-DE11-9DCB-001D09F2915A.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y1000-v1/0007/185CEE58-78B4-DE11-9A00-001D09F2A690.root'

#'/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/961D5F56-89B4-DE11-8C30-000423D98EC4.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/8A2431FE-87B4-DE11-BEE6-000423D98800.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/8A23E975-BDB4-DE11-A0ED-0019B9F6C674.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/4479D311-87B4-DE11-A495-000423D9890C.root',
#       '/store/relval/CMSSW_3_1_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/1A471008-83B4-DE11-8CB8-000423D9890C.root'

#startup
#'/store/relval/CMSSW_3_1_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START_31X_V4A-v1/0000/F89B1E02-B1B9-DE11-A7AA-001D09F2447F.root',
#'/store/relval/CMSSW_3_1_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START_31X_V4A-v1/0000/6C7257CF-ABB9-DE11-9EC5-001D09F2960F.root',
#'/store/relval/CMSSW_3_1_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START_31X_V4A-v1/0000/30CE260E-C4B9-DE11-9AC8-001D09F28F0C.root',
#'/store/relval/CMSSW_3_1_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START_31X_V4A-v1/0000/127D5ED5-A7B9-DE11-890E-001D09F2437B.root'

# ZMuMu
#'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/AECE5D8B-50B4-DE11-8329-001D09F23C73.root',
#'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/602EC0DD-4DB4-DE11-97C9-0019B9F70468.root',
#'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/4A80D48E-D3B4-DE11-AAF5-001D09F28F25.root',
#'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y100-v1/0007/0EC15F0B-4FB4-DE11-AE78-001D09F23C73.root'

# ZMuMu y=10000
'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0007/6E179A2D-BEB4-DE11-A378-001D09F2A465.root',
'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0006/8E075F54-1DB4-DE11-9147-001D09F251FE.root',
'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0006/44255641-1FB4-DE11-ABFC-001D09F2B2CF.root',
'/store/relval/CMSSW_3_1_3/RelValZMM/GEN-SIM-RECO/MC_31X_V5_Early10TeVX322Y10000-v1/0006/24FACFDE-20B4-DE11-8331-001D09F2AF1E.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500) #1500
)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################
# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('standAloneMuons') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True
#process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 15
#process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
#process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 20
#process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 1
#########################

process.d0_phi_analyzer.BeamFitter.OutputFileName = 'bsZMMwithMuons10000debug.root' #AtVtx10000.root'
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

# fit as function of lumi sections
#process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
#process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 10
