import FWCore.ParameterSet.Config as cms

# IC5 Calo jets

JetAnalyzerIC5Calo = cms.EDAnalyzer("CaloJetTester",
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

JetAnalyzerIC5PF = cms.EDAnalyzer("PFJetTester",
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

JetAnalyzerKt4Calo = cms.EDAnalyzer("CaloJetTester",
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
                                    
JetAnalyzerKt6Calo = cms.EDAnalyzer("CaloJetTester",
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

# AntiKt5 jets
                                    
JetAnalyzerAk5Calo = cms.EDAnalyzer("CaloJetTester",
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
                                    
JetAnalyzerAk7Calo = cms.EDAnalyzer("CaloJetTester",
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

JetAnalyzerAk5PF = cms.EDAnalyzer("PFJetTester",
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

## IC5 JPT jets
JetAnalyzerIC5JPT = cms.EDAnalyzer("JPTJetTester",
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
JetAnalyzerAk5JPT = cms.EDAnalyzer("JPTJetTester",
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


