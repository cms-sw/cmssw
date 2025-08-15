from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

hltMetPreValidSeq = cms.Sequence()

from Validation.RecoMET.metTester_cfi import metTester as _metTester
_hltMetTester = _metTester.clone(
    primaryVertices = cms.InputTag("hltPhase2PixelVertices"),
    genMetTrue = cms.InputTag("genMetTrue"),
    genMetCalo = cms.InputTag("genMetCalo"),
)

hltMetAnalyzerPF = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFMET'), 
    METType = cms.untracked.string('pf'), 
)

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMET'),
    METType = cms.untracked.string('pf'),
)

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMET'),
    METType = cms.untracked.string('pf'),
)

hltMetTypeOneAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMETTypeOne'),
    METType = cms.untracked.string('pf'),
)

hltMetAnalyzerPFCalo = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltCaloMET'),
    METType = cms.untracked.string('calo'),
)

