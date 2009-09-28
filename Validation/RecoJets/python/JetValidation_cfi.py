import FWCore.ParameterSet.Config as cms

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


