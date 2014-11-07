import FWCore.ParameterSet.Config as cms

#-------------------------------------------------------------------------------
JetAnalyzerAk4Calo = cms.EDAnalyzer("JetTester",
                                    JetType = cms.untracked.string('calo'),
                                    src            = cms.InputTag("ak4CaloJets"),
                                    srcGen         = cms.InputTag("ak4GenJets"),
                                    JetCorrections = cms.InputTag("ak4CaloL2L3Corrector"),
                                    primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                    recoJetPtThreshold = cms.double(40),
                                    matchGenPtThreshold                 = cms.double(20.0),
                                    RThreshold                     = cms.double(0.3)
                                    )
#-------------------------------------------------------------------------------
JetAnalyzerAk4PF = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  src            = cms.InputTag("ak4PFJets"),
                                  srcGen         = cms.InputTag("ak4GenJets"),
                                  JetCorrections = cms.InputTag("ak4PFL1FastL2L3Corrector"),
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
#                                   srcGen         = cms.InputTag("ak4GenJets"),
#                                   JetCorrections = cms.string("newAk4JPTL1FastL2L3"),
#                                 primVertex     = cms.InputTag("offlinePrimaryVertices"),
#                                   recoJetPtThreshold = cms.double(40),
#                                   matchGenPtThreshold            = cms.double(20.0),
#                                   RThreshold                     = cms.double(0.3)
#                                   )
# AntiKt5 PF CHS jets
#-------------------------------------------------------------------------------
JetAnalyzerAk4PFCHS = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('pf'),
                                  src            = cms.InputTag("ak4PFJetsCHS"),
                                  srcGen         = cms.InputTag("ak4GenJets"),
                                  JetCorrections = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"),
                                  primVertex     = cms.InputTag("offlinePrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold           = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )
JetAnalyzerAk4PFCHSMiniAOD = cms.EDAnalyzer("JetTester",
                                  JetType = cms.untracked.string('miniaod'),
                                  src            = cms.InputTag("slimmedJets"),
                                  srcGen         = cms.InputTag("slimmedGenJets"),
                                  primVertex     = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                  recoJetPtThreshold = cms.double(40),
                                  matchGenPtThreshold                 = cms.double(20.0),
                                  RThreshold                     = cms.double(0.3)
                                  )

