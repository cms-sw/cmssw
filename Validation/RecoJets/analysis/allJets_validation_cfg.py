import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

#process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Simulation_cff")
#process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff")

# Validation module
process.load("Validation.RecoJets.JetValidation_cff")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/BEC68E16-45AB-DE11-A3AC-001D09F2525D.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/B68C9C5E-4AAB-DE11-97DC-001D09F29533.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/B200DC23-44AB-DE11-87A8-001D09F26C5C.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/A8AB06D2-88AB-DE11-9C07-0019B9F70468.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/A850153C-43AB-DE11-B571-001D09F2AD84.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/985EB6CB-46AB-DE11-8632-001D09F34488.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/88B0CD9B-45AB-DE11-854A-001D09F24637.root',
       #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0004/44D3D763-71AB-DE11-A7C8-001D09F28F0C.root'

        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0004/0452E86B-ABAB-DE11-A31F-000423D996C8.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/A0C7DA44-BBAA-DE11-BA7D-001D09F241B4.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/9A6F263A-BFAA-DE11-89A6-001D09F295A1.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/9283FF14-BEAA-DE11-8F46-0019B9F72CE5.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/7EB58716-C1AA-DE11-BAC9-001D09F282F5.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/7852640A-C3AA-DE11-880F-001D09F25208.root',
        #'/store/relval/CMSSW_3_3_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0003/10F3F6D6-BCAA-DE11-BC2F-001D09F2462D.root'
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/C0A7B148-7AA7-DE11-9E2E-001D09F252E9.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/AE3F97A3-81A7-DE11-9170-001D09F28EC1.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/AC7F7145-7CA7-DE11-B72E-001D09F2441B.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/AC2AE288-72A7-DE11-B70D-001D09F24D4E.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/9E34FBAB-80A7-DE11-9CCC-001D09F2A690.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/8EC9C834-7DA7-DE11-95C4-0019B9F7310E.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/6EA8334E-7BA7-DE11-9A7D-001D09F291D2.root',
        '/store/relval/CMSSW_3_3_0_pre4/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0000/1EE0668E-7AA7-DE11-8671-001D09F2424A.root'
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/F0003B41-AB6B-DE11-A751-001D09F2A690.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/E4E8B446-A06B-DE11-AF53-000423D6B2D8.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/E0CCCD38-B56B-DE11-B6D2-001D09F282F5.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/9C6C79B9-A56B-DE11-85BA-0019B9F707D8.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/608F5711-B66B-DE11-9A82-001D09F28E80.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/58E40A13-AF6B-DE11-B7F8-001D09F24D67.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/2204FCC2-976B-DE11-B1BB-000423D98844.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/2086BDA1-B36B-DE11-93A8-001D09F2AD4D.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/16E5D232-D66B-DE11-85E0-001D09F2924F.root',
        #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/000F823D-B86B-DE11-90C2-000423D98B6C.root'

    )

)

process.fileSaver = cms.EDFilter("JetFileSaver",
                                 OutputFile = cms.untracked.string('histo.root')
)

## Test for corrected jets - available only for 
process.prefer("L2L3JetCorrectorIC5Calo")

process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
)

## IC5 Corrected jets
process.JetAnalyzerIC5Cor = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("L2L3CorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

## IC5 JPT jets
process.JetAnalyzerIC5JPT = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

## AntiKt5 JPT jets
process.JetAnalyzerAk5JPT = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
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

process.p1 = cms.Path(process.fileSaver
                      #--- Non-Standard sequence (that involve Producers)
                      *process.L2L3CorJetIcone5
                      *process.ZSPJetCorrectionsIcone5
                      *process.ZSPJetCorrectionsAntiKt5
                      *process.JetPlusTrackCorrectionsIcone5
                      *process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerIC5Cor
                      *process.JetAnalyzerIC5JPT
                      *process.JetAnalyzerAk5JPT
                      #--- Standard sequence
                      *process.JetValidation)
                      #--- DQM stats module
                      #*process.dqmStoreStats)

process.DQM.collectorHost = ''

