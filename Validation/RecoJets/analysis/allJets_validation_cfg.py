import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")

#process.load("JetMETCorrections.Configuration.L2L3Corrections_iCSA08_S156_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_Summer08Redigi_cff")
#process.load("JetMETCorrections.Configuration.L2L3Corrections_Winter09_cff")

#process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

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

# IC5 Calo jets

process.JetAnalyzerIC5Calo = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("iterativeCone5CaloJets"),
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

# IC5 PFlow jets

process.JetAnalyzerIC5PF = cms.EDFilter("PFJetTester",
    src = cms.InputTag("iterativeCone5PFJets"),
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

# IC5 JPT jets

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

# kt4 Calo jets

process.JetAnalyzerKt4Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("kt4CaloJets"),                                   
    srcGen = cms.InputTag("kt4GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)
                                    
# kt6 Calo jets
                                    
process.JetAnalyzerKt6Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("kt6CaloJets"),                                   
    srcGen = cms.InputTag("kt6GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# Sisc5 jets
                                    
process.JetAnalyzerSc5Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("sisCone5CaloJets"),                                 
    srcGen = cms.InputTag("sisCone5GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# Sisc7 jets
                                    
process.JetAnalyzerSc7Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("sisCone7CaloJets"),
    srcGen = cms.InputTag("sisCone7GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# AntiKt5 jets
                                    
process.JetAnalyzerAk5Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("ak5CaloJets"),                                 
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

# AntiKt7 jets
                                    
process.JetAnalyzerAk7Calo = cms.EDFilter("CaloJetTester",
    src = cms.InputTag("ak7CaloJets"),
    srcGen = cms.InputTag("ak7GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
#    outputFile = cms.untracked.string('histo.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# AntiKt5 PFlow jets

process.JetAnalyzerAk5PF = cms.EDFilter("PFJetTester",
    src = cms.InputTag("ak5PFJets"),
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

# AntiKt5 JPT jets

process.JetAnalyzerIC5JPT = cms.EDFilter("CaloJetTester",
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

# IC5 Corrected jets

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


process.prefer("L2L3JetCorrectorIC5Calo")

process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIC5Calo')
)

process.p1 = cms.Path(process.fileSaver*process.L2L3CorJetIcone5
                      *process.ZSPJetCorrectionsIcone5
                      *process.ZSPJetCorrectionsAntiKt5
                      *process.JetPlusTrackCorrectionsIcone5
                      *process.JetPlusTrackCorrectionsAntiKt5
                      *process.JetAnalyzerIC5Calo
                      *process.JetAnalyzerIC5PF*process.JetAnalyzerIC5JPT
                      *process.JetAnalyzerKt4Calo*process.JetAnalyzerKt6Calo
                      *process.JetAnalyzerSc5Calo*process.JetAnalyzerSc7Calo
                      *process.JetAnalyzerAk5Calo*process.JetAnalyzerAk7Calo
                      *process.JetAnalyzerAk5PF
                      *process.JetAnalyzerIC5Cor)

