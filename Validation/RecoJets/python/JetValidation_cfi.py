import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# AntiKt5 Calo jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4Calo = DQMEDAnalyzer('JetTester',
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

# AntiKt5 PF jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4PF = DQMEDAnalyzer('JetTester',
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

# AntiKt5 PF CHS jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4PFCHS = DQMEDAnalyzer('JetTester',
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

# AntiKt5 PF Puppi jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4PFPUPPI = DQMEDAnalyzer('JetTester',
                                  JetType = cms.untracked.string('pf'),
                                  src            = cms.InputTag("ak4PFJetsPuppi"),
                                  srcGen         = cms.InputTag("ak4GenJetsNoNu"),
                                  JetCorrections = cms.InputTag("newAk4PFPUPPIL1FastL2L3Corrector"),
                                  primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

JetAnalyzerAk4PFCHSMiniAOD = DQMEDAnalyzer('JetTester',
                                  JetType = cms.untracked.string('miniaod'),
                                  src            = cms.InputTag("slimmedJets"),
                                  srcGen         = cms.InputTag("slimmedGenJets"),
                                  JetCorrections = cms.InputTag(""),#not called for MiniAOD
                                  primVertex     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

JetAnalyzerAk4PFPUPPIMiniAOD = DQMEDAnalyzer('JetTester',
                                  JetType = cms.untracked.string('miniaod'),
                                  src            = cms.InputTag("slimmedJetsPuppi"),
                                  srcGen         = cms.InputTag("slimmedGenJets"),
                                  JetCorrections = cms.InputTag(""),#not called for MiniAOD
                                  primVertex     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

JetAnalyzerAk8PFPUPPIMiniAOD = DQMEDAnalyzer('JetTester',
                                  JetType = cms.untracked.string('miniaod'),
                                  src            = cms.InputTag("slimmedJetsAK8"),
                                  srcGen         = cms.InputTag("slimmedGenJetsAK8"),
                                  JetCorrections = cms.InputTag(""),#not called for MiniAOD
                                  primVertex     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )


