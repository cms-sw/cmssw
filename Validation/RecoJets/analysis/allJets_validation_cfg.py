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
process.load("JetMETCorrections.Configuration.ZSPJetCorrections152_cff")
process.load("JetMETCorrections.Configuration.L2L3Corrections_iCSA08_S156_cff")

#process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(500)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/F29F469F-B141-DE11-A368-000423D98950.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/D29D4EA2-E943-DE11-900F-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/B0EDEE9D-E343-DE11-A4BC-001D09F27067.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/AC5C2C7E-B241-DE11-BA3B-001D09F2523A.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/A6C7ACD4-B241-DE11-AEB0-0030487D1BCC.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/9C3291E5-E143-DE11-9B02-001D09F2432B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/9686D97D-0344-DE11-991A-001D09F24024.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/881C487B-E243-DE11-9147-001D09F2432B.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/86D723CA-EA43-DE11-B9FA-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/80C0D453-E143-DE11-BB9D-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/80A38F88-EB43-DE11-AF6D-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/74EE6234-E743-DE11-9C52-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/54CCDC95-E641-DE11-AF26-001D09F28D4A.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/3A8A3554-E843-DE11-BE44-001D09F24DDA.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/301D4047-E943-DE11-9285-001D09F2AD4D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/2A14EB3A-F241-DE11-912A-001D09F28D4A.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/20CE9336-B241-DE11-A36F-001D09F2426D.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/1A037839-EA43-DE11-A9C4-001D09F29524.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/183B4656-5044-DE11-84AC-001D09F24FE7.root',
       '/store/relval/CMSSW_3_1_0_pre7/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0004/1452AD89-AF41-DE11-B669-000423D6A6F4.root'    )

)

# IC5 Calo jets

#process.JetAnalyzer1 = cms.EDFilter("CaloJetTester",
process.JetAnalyzer1 = cms.EDAnalyzer("CaloJetTester",
    src = cms.InputTag("iterativeCone5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),                                
#    TurnOnEverything = cms.untracked.string('yes'),
#    TurnOnEverything = cms.untracked.string('no'),
    outputFile = cms.untracked.string('histo.root'),
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


process.prefer("L2L3JetCorrectorIcone5")

process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIcone5')
)


#process.p1 = cms.Path(process.JetAnalyzer1*process.JetAnalyzer2*process.JetAnalyzer3*process.JetAnalyzer4*process.JetAnalyzer5*process.JetAnalyzer6*process.JetAnalyzer7*process.JetAnalyzer8)

process.p1 = cms.Path(process.L2L3CorJetIcone5*process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.JetAnalyzer1*process.JetAnalyzer2*process.JetAnalyzer3*process.JetAnalyzer4*process.JetAnalyzer5*process.JetAnalyzer6*process.JetAnalyzer7*process.JetAnalyzer8)
