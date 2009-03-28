import FWCore.ParameterSet.Config as cms

JetAnalyzer1 = cms.EDAnalyzer(
    "CaloJetTester",
    src = cms.InputTag("iterativeCone5CaloJets"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)

# IC5 PFlow jets

JetAnalyzer2 = cms.EDFilter(
    "PFJetTester",
    src = cms.InputTag("iterativeCone5PFJets"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),                                    
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),                                    
    genPtThreshold = cms.double(1.0),           
    RThreshold = cms.double(0.3),              
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# IC5 JPT jets

JetAnalyzer3 = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("JetPlusTrackZSPCorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                 
)

# kt4 Calo jets

JetAnalyzer4 = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("kt4CaloJets"),                                   
    srcGen = cms.InputTag("kt4GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)
                                    
# kt6 Calo jets
                                    
JetAnalyzer5 = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("kt6CaloJets"),                                   
    srcGen = cms.InputTag("kt6GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# Sisc5 jets
                                    
JetAnalyzer6 = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("sisCone5CaloJets"),                                 
    srcGen = cms.InputTag("sisCone5GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# Sisc7 jets
                                    
JetAnalyzer7 = cms.EDFilter(
    "CaloJetTester",
    src = cms.InputTag("sisCone7CaloJets"),
    srcGen = cms.InputTag("sisCone7GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)
)

# IC5 Corrected jets

JetAnalyzer8 = cms.EDAnalyzer(
    "CaloJetTester",
    src = cms.InputTag("L2L3CorJetIcone5"),
    srcGen = cms.InputTag("iterativeCone5GenJets"),                                
    outputFile = cms.untracked.string('JetValidation.root'),                                
#    outputFile = cms.untracked.string('test.root'),
    genEnergyFractionThreshold = cms.double(0.05),
    genPtThreshold = cms.double(1.0),
    RThreshold = cms.double(0.3),
    reverseEnergyFractionThreshold = cms.double(0.5)                                    
)
