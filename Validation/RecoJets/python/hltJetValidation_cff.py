from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFPuppiJetCorrector_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFJetCorrector_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL1_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL2_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrectorL3_cfi import *
from HLTrigger.Configuration.HLT_75e33.modules.hltAK4PFCHSJetCorrector_cfi import *

# reproduce JetCorrector modules when not available
hltJetCorrectionSequence = cms.Sequence(
    hltAK4PFPuppiJetCorrectorL1
    + hltAK4PFPuppiJetCorrectorL2
    + hltAK4PFPuppiJetCorrectorL3
    + hltAK4PFPuppiJetCorrector
    + hltAK4PFJetCorrectorL1
    + hltAK4PFJetCorrectorL2
    + hltAK4PFJetCorrectorL3
    + hltAK4PFJetCorrector
    + hltAK4PFCHSJetCorrectorL1
    + hltAK4PFCHSJetCorrectorL2
    + hltAK4PFCHSJetCorrectorL3
    + hltAK4PFCHSJetCorrector
)

hltJetPreValidSeq = cms.Sequence(
    prunedGenParticlesWithStatusOne
    + prunedGenParticles
    + finalGenParticles
    + genParticlesForJetsNoNu
    + ak4GenJetsNoNu
    + hltJetCorrectionSequence
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

