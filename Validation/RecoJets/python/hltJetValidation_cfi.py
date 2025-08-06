import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

hltJetAnalyzerAk4PFPuppi = DQMEDAnalyzer('JetTester',
                            JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
                            src = cms.InputTag("hltAK4PFPuppiJets"),
                            srcGen = cms.InputTag("ak4GenJetsNoNu"),
                            JetCorrections = cms.InputTag("hltAK4PFPuppiJetCorrector"),
                            primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
                            recoJetPtThreshold = cms.double(30), # defines the "lowPt" jets between 20 and 40 GeV
                            matchGenPtThreshold = cms.double(20.0),
                            RThreshold = cms.double(0.2),
)

hltJetAnalyzerAk4PFCluster = DQMEDAnalyzer('JetTester',
                            JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
                            src = cms.InputTag("hltAK4PFClusterJets"),
                            srcGen = cms.InputTag("ak4GenJetsNoNu"),
                            JetCorrections = cms.InputTag(""),
                            primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
                            recoJetPtThreshold = cms.double(30), # defines the "lowPt" jets between 20 and 40 GeV
                            matchGenPtThreshold = cms.double(20.0),
                            RThreshold = cms.double(0.2),
)

hltJetAnalyzerAk4PF = DQMEDAnalyzer('JetTester',
                            JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
                            src = cms.InputTag("hltAK4PFJets"),
                            srcGen = cms.InputTag("ak4GenJetsNoNu"),
                            JetCorrections = cms.InputTag("hltAK4PFJetCorrector"),
                            primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
                            recoJetPtThreshold = cms.double(30), # defines the "lowPt" jets between 20 and 40 GeV
                            matchGenPtThreshold = cms.double(20.0),
                            RThreshold = cms.double(0.2),
)

hltJetAnalyzerAk4PFCHS = DQMEDAnalyzer('JetTester',
                            JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
                            src = cms.InputTag("hltAK4PFCHSJets"),
                            srcGen = cms.InputTag("ak4GenJetsNoNu"),
                            JetCorrections = cms.InputTag("hltAK4PFCHSJetCorrector"),
                            primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
                            recoJetPtThreshold = cms.double(30), # defines the "lowPt" jets between 20 and 40 GeV
                            matchGenPtThreshold = cms.double(20.0),
                            RThreshold = cms.double(0.2),
)
