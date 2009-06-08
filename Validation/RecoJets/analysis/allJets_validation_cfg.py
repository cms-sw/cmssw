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
       input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/EAE8043B-EB4E-DE11-8ADC-0019DB29C614.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/E0D2779F-ED4E-DE11-AAAF-000423D6B42C.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/CAC40E5E-524F-DE11-BBB6-001D09F251D1.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/C2B3C04F-EB4E-DE11-9A68-001617C3B6CC.root',
    #'/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/ACD61339-ED4E-DE11-A33B-001617C3B6CC.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/A21B95E1-EC4E-DE11-8EB8-001D09F2905B.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/96FF33DE-EC4E-DE11-80E9-001D09F2A465.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/90B4023B-EF4E-DE11-8258-001D09F24691.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/866EE0D7-EC4E-DE11-BAA7-001617C3B706.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/62FB60E2-EE4E-DE11-ACEE-001D09F25393.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/483DD622-EE4E-DE11-8136-000423D94534.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/38F94336-EE4E-DE11-8ADC-001617C3B710.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/3425A3DC-ED4E-DE11-BBAE-001D09F276CF.root',
    '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_31X_FastSim_v1/0007/10A0BBE1-EC4E-DE11-9C3C-000423D6B444.root'
    )

)

process.fileSaver = cms.EDFilter("JetFileSaver",
                                 OutputFile = cms.untracked.string('JetTester_FlatPt_15_3000_IDEAL_fastsim_310pre9.root')
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


process.prefer("L2L3JetCorrectorIcone5")

process.L2L3CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrectorIcone5')
)

process.p1 = cms.Path(process.fileSaver*process.L2L3CorJetIcone5*process.ZSPJetCorrections*process.JetPlusTrackCorrections*process.JetAnalyzer1*process.JetAnalyzer2*process.JetAnalyzer3*process.JetAnalyzer4*process.JetAnalyzer5*process.JetAnalyzer6*process.JetAnalyzer7*process.JetAnalyzer8)

