import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_30X::All'
#process.GlobalTag.globaltag = 'STARTUP_30X::All'
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

        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/F0003B41-AB6B-DE11-A751-001D09F2A690.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/E4E8B446-A06B-DE11-AF53-000423D6B2D8.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/E0CCCD38-B56B-DE11-B6D2-001D09F282F5.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/9C6C79B9-A56B-DE11-85BA-0019B9F707D8.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/608F5711-B66B-DE11-9A82-001D09F28E80.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/58E40A13-AF6B-DE11-B7F8-001D09F24D67.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/2204FCC2-976B-DE11-B1BB-000423D98844.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/2086BDA1-B36B-DE11-93A8-001D09F2AD4D.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/16E5D232-D66B-DE11-85E0-001D09F2924F.root',
        '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/MC_31X_V2-v1/0002/000F823D-B86B-DE11-90C2-000423D98B6C.root'


    )

)

process.fileSaver = cms.EDFilter("JetFileSaver",
                                 OutputFile = cms.untracked.string('histo.root')
)

# IC5 Calo jets

#process.JetAnalyzer1 = cms.EDFilter("CaloJetTester",
process.JetAnalyzer1 = cms.EDAnalyzer("CaloJetTester",
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

process.JetAnalyzer2 = cms.EDFilter("PFJetTester",
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

process.JetAnalyzer3 = cms.EDFilter("CaloJetTester",
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

process.JetAnalyzer4 = cms.EDFilter("CaloJetTester",
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
                                    
process.JetAnalyzer5 = cms.EDFilter("CaloJetTester",
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
                                    
process.JetAnalyzer6 = cms.EDFilter("CaloJetTester",
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
                                    
process.JetAnalyzer7 = cms.EDFilter("CaloJetTester",
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

# IC5 Corrected jets

process.JetAnalyzer8 = cms.EDAnalyzer("CaloJetTester",
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

process.p1 = cms.Path(process.fileSaver*process.L2L3CorJetIcone5*process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.JetAnalyzer1*process.JetAnalyzer2*process.JetAnalyzer3*process.JetAnalyzer4*process.JetAnalyzer5*process.JetAnalyzer6*process.JetAnalyzer7*process.JetAnalyzer8)

