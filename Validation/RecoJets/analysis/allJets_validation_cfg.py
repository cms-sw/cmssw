import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("JETVALIDATION")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_30X::All'
process.GlobalTag.globaltag = 'STARTUP_30X::All'
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
       input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/26D95147-9E16-DE11-BEEE-0030486790BE.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/48DEB40C-9516-DE11-9234-0018F3D09616.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/9CB5FDE4-9316-DE11-AF3D-003048678C9A.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/9EDF6E0A-A416-DE11-9B51-001A92811700.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/A07F4531-A016-DE11-B4EA-0017312B58FF.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/B472B4A5-9216-DE11-BBD5-001A92810AC8.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/FA0C1551-A116-DE11-BE88-001A9281172A.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/FAA830B1-A216-DE11-9395-0018F3D09630.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0002/06D8A590-FE16-DE11-B398-003048678FAE.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/06072974-1916-DE11-80F2-001617DBD224.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/16F15C67-1116-DE11-92F9-000423D98FBC.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/22392399-5116-DE11-8BDB-001617DBD230.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/A26596FC-0F16-DE11-A0DE-000423D98834.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/B228E953-AB16-DE11-A83B-001617C3B6E8.root',
        '/store/relval/CMSSW_3_1_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0003/E650DA99-1316-DE11-B057-000423D9A2AE.root'


    )
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
    outputFile = cms.untracked.string('histo.root'),                                    
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
    outputFile = cms.untracked.string('histo.root'),
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
    outputFile = cms.untracked.string('histo.root'),
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
    outputFile = cms.untracked.string('histo.root'),
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
    outputFile = cms.untracked.string('histo.root'),
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
    outputFile = cms.untracked.string('histo.root'),
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
    outputFile = cms.untracked.string('histo.root'),                                
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
