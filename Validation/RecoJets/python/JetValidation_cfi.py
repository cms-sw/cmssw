import FWCore.ParameterSet.Config as cms

# kt6 PF jets - UnCorr
#-------------------------------------------------------------------------------
JetAnalyzerKt6PF = cms.EDAnalyzer("JetTester",
                                  src            = cms.InputTag("kt6PFJets"),
                                  srcRho         = cms.InputTag("kt6PFJets","rho"),
                                  srcGen         = cms.InputTag("kt6GenJets"),
                                  JetCorrections = cms.string(""),
                                  recoJetPtThreshold = cms.double(40),
                                  genEnergyFractionThreshold     = cms.double(0.05),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3),
                                  reverseEnergyFractionThreshold = cms.double(0.5)
                                  )

# kt6 Calo jets - UnCorr
#-------------------------------------------------------------------------------
JetAnalyzerKt6Calo = cms.EDAnalyzer("JetTester",
                                    src            = cms.InputTag("kt6CaloJets"),                                   
                                    srcRho         = cms.InputTag("kt6CaloJets","rho"),
                                    srcGen         = cms.InputTag("kt6GenJets"),
                                    JetCorrections = cms.string(""),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )

# AntiKt5 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5Calo = cms.EDAnalyzer("JetTester",
                                    src            = cms.InputTag("ak5CaloJets"),                                 
                                    srcRho         = cms.InputTag("ak5CaloJets","rho"),
                                    srcGen         = cms.InputTag("ak5GenJets"),
                                    JetCorrections = cms.string("newAk5CaloL2L3"),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )

# AntiKt7 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk7Calo = cms.EDAnalyzer("JetTester",
                                    src            = cms.InputTag("ak7CaloJets"),
                                    srcRho         = cms.InputTag("ak7CaloJets","rho"),
                                    srcGen         = cms.InputTag("ak7GenJets"),
                                    JetCorrections = cms.string("newAk7CaloL2L3"),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )

# AntiKt5 PF jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5PF = cms.EDAnalyzer("JetTester",
                                  src            = cms.InputTag("ak5PFJets"),
                                  srcRho         = cms.InputTag("ak5PFJets","rho"),
                                  srcGen         = cms.InputTag("ak5GenJets"),
                                  JetCorrections = cms.string("newAk5PFL1FastL2L3"),
                                  recoJetPtThreshold = cms.double(40),
                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
                                  matchGenPtThreshold                 = cms.double(20.0),           
                                  RThreshold                     = cms.double(0.3),              
                                  reverseEnergyFractionThreshold = cms.double(0.5)
                                  )

# AntiKt5 JPT jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5JPT = cms.EDAnalyzer("JetTester",
                                   src            = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
                                   srcRho         = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5","rho"),
                                   srcGen         = cms.InputTag("ak5GenJets"),
                                   JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
                                   recoJetPtThreshold = cms.double(40),
                                   genEnergyFractionThreshold     = cms.double(0.05),
                                   matchGenPtThreshold                 = cms.double(20.0),
                                   RThreshold                     = cms.double(0.3),
                                   reverseEnergyFractionThreshold = cms.double(0.5)
                                   )
