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
JetAnalyzerAk4Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    src            = cms.InputTag("ak4CaloJets"),
#                                    srcRho         = cms.InputTag("rho", "ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak4GenJetsNoNu"),
                                    JetCorrections = cms.InputTag("newAk4CaloL2L3Corrector"),
                                    primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                    recoJetPtThreshold = cms.double(40),
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
JetAnalyzerAk4PF = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  src            = cms.InputTag("ak4PFJets"),
#                                  srcRho         = cms.InputTag("ak4PFJets","rho"),
                                  srcGen         = cms.InputTag("ak4GenJetsNoNu"),
                                  JetCorrections = cms.InputTag("newAk4PFL1FastL2L3Corrector"),
                                  primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

# AntiKt5 JPT jets
#-------------------------------------------------------------------------------
#JetAnalyzerAk4JPT = cms.EDAnalyzer("JetTester",
#                                   JetType = cms.untracked.string('jpt'),
#                                   OutputFile = cms.untracked.string(''),
#                                   src            = cms.InputTag("JetPlusTrackZSPCorJetAntiKt4"),
##                                   srcRho         = cms.InputTag("JetPlusTrackZSPCorJetAntiKt4","rho"),
#                                   srcGen         = cms.InputTag("ak4GenJetsNoNu"),
#                                   JetCorrections = cms.string("newAk4JPTL1FastL2L3"),
#                                   recoJetPtThreshold = cms.double(40),
#                                   genEnergyFractionThreshold     = cms.double(0.05),
#                                   matchGenPtThreshold                 = cms.double(20.0),
#                                   RThreshold                     = cms.double(0.3)
#                                   )
# AntiKt5 PF CHS jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4PFCHS = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  src            = cms.InputTag("ak4PFJetsCHS"),
#                                  srcRho         = cms.InputTag("ak4PFJetsCHS","rho"),
                                  srcGen         = cms.InputTag("ak4GenJetsNoNu"),
                                  JetCorrections = cms.InputTag("newAk4PFCHSL1FastL2L3Corrector"),
                                  primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
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
JetAnalyzerAk4PFCHSMiniAOD = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('miniaod'),
                                  src            = cms.InputTag("slimmedJets"),
                                  srcGen         = cms.InputTag("slimmedGenJets"),
                                  JetCorrections = cms.InputTag(""),#not called for MiniAOD
                                  primVertex     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

