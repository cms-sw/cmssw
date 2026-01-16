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
    isHLT=True,
    recoJetPtThreshold=30.0,
    matchGenPtThreshold=20.0,
    RThreshold=0.4,
    srcGen="ak4GenJetsNoNu",
    JetType='pf',  # requires "pf", "calo", or "miniaod"
    primVertex="hltGoodOfflinePrimaryVertices",
)

hltJetAnalyzerAK4PFPuppi = _hltJetTester.clone(
    src = "hltAK4PFPuppiJets",
    JetCorrections = "hltAK4PFPuppiJetCorrector",
)

hltJetAnalyzerAK4PF = _hltJetTester.clone(
    src = "hltAK4PFJets",
    JetCorrections = "hltAK4PFJetCorrector",
)

hltJetAnalyzerAK4PFCHS = _hltJetTester.clone(
    src = "hltAK4PFCHSJets",
    JetCorrections = "hltAK4PFCHSJetCorrector",
)

