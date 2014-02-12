import FWCore.ParameterSet.Config as cms

### kt6 PF jets - UnCorr
###-------------------------------------------------------------------------------
#JetAnalyzerKt6PF = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('calo'),
#                                  OutputFile = cms.untracked.string(''),
#                                  src            = cms.InputTag("kt6PFJets"),
#                                  srcRho         = cms.InputTag("fixedGridRhoFastjetAll"),
#                                  srcGen         = cms.InputTag("kt6GenJets"),
#                                  JetCorrections = cms.string(""),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),
#                                  matchGenPtThreshold                 = cms.double(20.0),
#                                  RThreshold                     = cms.double(0.3)
#                                  )

## kt6 Calo jets - UnCorr
##-------------------------------------------------------------------------------
#JetAnalyzerKt6Calo = cms.EDAnalyzer("JetTester",
#                                    JetType = cms.untracked.string('calo'),
#                                    OutputFile = cms.untracked.string(''),
#                                    src            = cms.InputTag("kt6CaloJets"),                                   
#                                    srcRho         = cms.InputTag("fixedGridRhoFastjetAllCalo"),
#                                    srcGen         = cms.InputTag("kt6GenJets"),
#                                    JetCorrections = cms.string(""),
#                                    recoJetPtThreshold = cms.double(40),
#                                    genEnergyFractionThreshold     = cms.double(0.05),
#                                    matchGenPtThreshold                 = cms.double(20.0),
#                                    RThreshold                     = cms.double(0.3)
#                                    )

# AntiKt5 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    OutputFile = cms.untracked.string(''),
                                    src            = cms.InputTag("ak5CaloJets"),                                 
#                                    srcRho         = cms.InputTag("rho", "ak5CaloJets"),
                                    srcGen         = cms.InputTag("ak5GenJets"),
                                    JetCorrections = cms.string("newAk5CaloL2L3"),
                                    recoJetPtThreshold = cms.double(40),
                                    genEnergyFractionThreshold     = cms.double(0.05),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )

## AntiKt7 Calo jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk7Calo = cms.EDAnalyzer("JetTester",
#                                    JetType = cms.untracked.string('calo'),
#                                    OutputFile = cms.untracked.string(''),
#                                    src            = cms.InputTag("ak7CaloJets"),
#                                    srcRho         = cms.InputTag("ak7CaloJets","rho"),
#                                    srcGen         = cms.InputTag("ak7GenJets"),
#                                    JetCorrections = cms.string("newAk7CaloL2L3"),
#                                    recoJetPtThreshold = cms.double(40),
#                                    genEnergyFractionThreshold     = cms.double(0.05),
#                                    matchGenPtThreshold                 = cms.double(20.0),
#                                    RThreshold                     = cms.double(0.3)
#                                    )
#

# AntiKt5 PF jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5PF = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  OutputFile = cms.untracked.string(''),
                                  src            = cms.InputTag("ak5PFJets"),
#                                  srcRho         = cms.InputTag("ak5PFJets","rho"),
                                  srcGen         = cms.InputTag("ak5GenJets"),
                                  JetCorrections = cms.string("newAk5PFL1FastL2L3"),
                                  recoJetPtThreshold = cms.double(40),
                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
                                  matchGenPtThreshold                 = cms.double(20.0),           
                                  RThreshold                     = cms.double(0.3)
                                  )

# AntiKt5 JPT jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5JPT = cms.EDAnalyzer("JetTester",
                                   JetType = cms.untracked.string('jpt'),
                                   OutputFile = cms.untracked.string(''),
                                   src            = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5"),
#                                   srcRho         = cms.InputTag("JetPlusTrackZSPCorJetAntiKt5","rho"),
                                   srcGen         = cms.InputTag("ak5GenJets"),
                                   JetCorrections = cms.string("newAk5JPTL1FastL2L3"),
                                   recoJetPtThreshold = cms.double(40),
                                   genEnergyFractionThreshold     = cms.double(0.05),
                                   matchGenPtThreshold                 = cms.double(20.0),
                                   RThreshold                     = cms.double(0.3)
                                   )
# AntiKt5 PF CHS jets
#-------------------------------------------------------------------------------
JetAnalyzerAk5PFCHS = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  OutputFile = cms.untracked.string(''),
                                  src            = cms.InputTag("ak5PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ak5PFJetsCHS","rho"),
                                  srcGen         = cms.InputTag("ak5GenJets"),
                                  JetCorrections = cms.string("newAk5PFchsL1FastL2L3"),
                                  recoJetPtThreshold = cms.double(40),
                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
                                  matchGenPtThreshold                 = cms.double(20.0),           
                                  RThreshold                     = cms.double(0.3) 
                                  )
## AntiKt8 PF  jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk8PF = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string(''),
#                                  src            = cms.InputTag("ak8PFJets"),
#                                  srcRho         = cms.InputTag("ak8PFJets","rho"),
##                                  srcGen         = cms.InputTag("ak8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("Ak8PFL1FastL2L3"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3)
#                                  )
## AntiKt8 PF CHS jets
##-------------------------------------------------------------------------------
#JetAnalyzerAk8PFCHS = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string(''),
#                                  src            = cms.InputTag("ak8PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ak8PFJetsCHS","rho"),
##                                  srcGen         = cms.InputTag("ak8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("Ak8PFL1FastL2L3CHS"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3)
#                                  )
## CA8 PF CHS jets
##-------------------------------------------------------------------------------
#JetAnalyzerCA8PFCHS = cms.EDAnalyzer("JetTester",
#                                  JetType = cms.untracked.string('pf'),
#                                  OutputFile = cms.untracked.string(''),
#                                  src            = cms.InputTag("ca8PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ca8PFJetsCHS","rho"),
##                                  srcGen         = cms.InputTag("ca8GenJets"),
#                                  srcGen         = cms.InputTag(""),
#                                  JetCorrections = cms.string("CA8PFL1FastL2L3CHS"),
#                                  recoJetPtThreshold = cms.double(40),
#                                  genEnergyFractionThreshold     = cms.double(0.05),                                    
#                                  matchGenPtThreshold                 = cms.double(20.0),           
#                                  RThreshold                     = cms.double(0.3)
#                                  )


