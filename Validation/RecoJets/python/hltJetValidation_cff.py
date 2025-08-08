from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

hltJetPreValidSeq = cms.Sequence(
    prunedGenParticlesWithStatusOne
    + prunedGenParticles
    + finalGenParticles
    + genParticlesForJetsNoNu
    + ak4GenJetsNoNu
)

from Validation.RecoJets.jetTester_cfi import jetTester as _jetTester
_hltJetTester = _jetTester.clone(
    isHLT = cms.untracked.bool(True),
    recoJetPtThreshold = cms.double(30),
    matchGenPtThreshold = cms.double(20.0),
    RThreshold = cms.double(0.2)
)

hltJetAnalyzerAk4PFPuppi = _hltJetTester.clone(
    JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
    src = cms.InputTag("hltAK4PFPuppiJets"),
    srcGen = cms.InputTag("ak4GenJetsNoNu"),
    JetCorrections = cms.InputTag("hltAK4PFPuppiJetCorrector"),
    primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
)

hltJetAnalyzerAk4PFCluster = _hltJetTester.clone(
    JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
    src = cms.InputTag("hltAK4PFClusterJets"),
    srcGen = cms.InputTag("ak4GenJetsNoNu"),
    JetCorrections = cms.InputTag(""),
    primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
)

hltJetAnalyzerAk4PF = _hltJetTester.clone(
    JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
    src = cms.InputTag("hltAK4PFJets"),
    srcGen = cms.InputTag("ak4GenJetsNoNu"),
    JetCorrections = cms.InputTag("hltAK4PFJetCorrector"),
    primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
)

hltJetAnalyzerAk4PFCHS = _hltJetTester.clone(
    JetType = cms.untracked.string('pf'), # requires "pf", "calo", or "miniaod"
    src = cms.InputTag("hltAK4PFCHSJets"),
    srcGen = cms.InputTag("ak4GenJetsNoNu"),
    JetCorrections = cms.InputTag("hltAK4PFCHSJetCorrector"),
    primVertex = cms.InputTag("hltGoodOfflinePrimaryVertices"),
)

