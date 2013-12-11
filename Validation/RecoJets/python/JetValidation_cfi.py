import FWCore.ParameterSet.Config as cms

### kt6 PF jets - UnCorr
###-------------------------------------------------------------------------------
#JetAnalyzerKt6PF = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('calo'),
#                                  OutputFile = cms.untracked.string('JetTester.root'),
#                                  src            = cms.InputTag("kt6PFJets"),
#                                  srcRho         = cms.InputTag("kt6PFJets","rho"),
#                                  srcGen         = cms.InputTag("kt6GenJets"),
#                                  JetCorrections = cms.string(""),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),
#                                  matchGenPtThreshold                 = cms.double(20.0),
#                                  RThreshold                     = cms.double(0.3),
#                                  reverseEnergyFractionThreshold = cms.double(0.5)
#                                  )

## kt6 Calo jets - UnCorr
##-------------------------------------------------------------------------------
#JetAnalyzerKt6Calo = cms.EDAnalyzer("JetTester",
#                                    JetType = cms.untracked.string('calo'),
#                                    OutputFile = cms.untracked.string('JetTester.root'),
#                                    src            = cms.InputTag("kt6CaloJets"),                                   
#                                    srcRho         = cms.InputTag("kt6CaloJets","rho"),
#                                    srcGen         = cms.InputTag("kt6GenJets"),
#                                    JetCorrections = cms.string(""),
#                                    recoJetPtThreshold = cms.double(40),
#                                    genEnergyFractionThreshold     = cms.double(0.05),
#                                    matchGenPtThreshold                 = cms.double(20.0),
#                                    RThreshold                     = cms.double(0.3),
#                                    reverseEnergyFractionThreshold = cms.double(0.5)
#                                    )

# AntiKt5 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    OutputFile = cms.untracked.string('JetTester.root'),
                                    src            = cms.InputTag("ak4CaloJets"),                                 
#                                    srcRho         = cms.InputTag("rho", "ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak4GenJets"),
                                    JetCorrections = cms.string("newAk5CaloL2L3"),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3),
                                    reverseEnergyFractionThreshold = cms.double(0.5)
                                    )

## AntiKt7 Calo jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk7Calo = cms.EDAnalyzer("JetTester",
#                                    JetType = cms.untracked.string('calo'),
#                                    OutputFile = cms.untracked.string('JetTester.root'),
#                                    src            = cms.InputTag("ak8CaloJets"),
#                                    srcRho         = cms.InputTag("ak8CaloJets","rho"),
#                                    srcGen         = cms.InputTag("ak8GenJets"),
#                                    JetCorrections = cms.string("newAk7CaloL2L3"),
#                                    recoJetPtThreshold = cms.double(40),
#                                    genEnergyFractionThreshold     = cms.double(0.05),
#                                    matchGenPtThreshold                 = cms.double(20.0),
#                                    RThreshold                     = cms.double(0.3),
#                                    reverseEnergyFractionThreshold = cms.double(0.5)
#                                    )
#

# AntiKt5 PF jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5PF = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  OutputFile = cms.untracked.string('JetTester.root'),
                                  src            = cms.InputTag("ak4PFJets"),
#                                  srcRho         = cms.InputTag("ak4PFJets","rho"),
                                  srcGen         = cms.InputTag("ak4GenJets"),
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
                                   JetType = cms.untracked.string('jpt'),
                                   OutputFile = cms.untracked.string('JetTester.root'),
                                   src            = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
#                                   srcRho         = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5","rho"),
                                   srcGen         = cms.InputTag("ak4GenJets"),
                                   JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
                                   recoJetPtThreshold = cms.double(40),
                                   genEnergyFractionThreshold     = cms.double(0.05),
                                   matchGenPtThreshold                 = cms.double(20.0),
                                   RThreshold                     = cms.double(0.3),
                                   reverseEnergyFractionThreshold = cms.double(0.5)
                                   )
# AntiKt5 PF CHS jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5PFCHS = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  OutputFile = cms.untracked.string('JetTester.root'),
                                  src            = cms.InputTag("ak4PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ak4PFJetsCHS","rho"),
                                  srcGen         = cms.InputTag("ak4GenJets"),
                                  JetCorrections = cms.string("newAk5PFchsL1FastL2L3"),
                                  recoJetPtThreshold = cms.double(40),
                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
                                  matchGenPtThreshold                 = cms.double(20.0),           
                                  RThreshold                     = cms.double(0.3),              
                                  reverseEnergyFractionThreshold = cms.double(0.5)
                                  )
## AntiKt8 PF  jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk8PF = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string('JetTester.root'),
#                                  src            = cms.InputTag("ak8PFJets"),
#                                  srcRho         = cms.InputTag("ak8PFJets","rho"),
##                                  srcGen         = cms.InputTag("ak8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("Ak8PFL1FastL2L3"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3),              
#                                  reverseEnergyFractionThreshold = cms.double(0.5)
#                                  )
## AntiKt8 PF CHS jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk8PFCHS = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string('JetTester.root'),
#                                  src            = cms.InputTag("ak8PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ak8PFJetsCHS","rho"),
##                                  srcGen         = cms.InputTag("ak8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("Ak8PFL1FastL2L3CHS"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3),              
#                                  reverseEnergyFractionThreshold = cms.double(0.5)
#                                  )
## CA8 PF CHS jets
##-------------------------------------------------------------------------------
#JetAnalyzerCA8PFCHS = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string('JetTester.root'),
#                                  src            = cms.InputTag("ca8PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ca8PFJetsCHS","rho"),
##                                  srcGen         = cms.InputTag("ca8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("CA8PFL1FastL2L3CHS"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3),              
#                                  reverseEnergyFractionThreshold = cms.double(0.5)
#                                  )
#

