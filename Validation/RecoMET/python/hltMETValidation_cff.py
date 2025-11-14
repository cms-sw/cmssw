from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

hltMetPreValidSeq = cms.Sequence()

from Validation.RecoMET.metTesterPostProcessor_cfi import metTesterPostProcessor as _metTesterPostProcessor
hltMetPostProcessor = _metTesterPostProcessor.clone(
    runDir = cms.untracked.string("HLT/JetMET/METValidation/"),
)

from Validation.RecoMET.metTester_cfi import metTester as _metTester
_hltMetTester = _metTester.clone(
    runDir = "HLT/JetMET/METValidation/",
    primaryVertices = 'hltPixelVertices',
    genMetTrue = 'genMetTrue',
    genMetCalo = 'genMetCalo',
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(_hltMetTester, primaryVertices = 'hltPhase2PixelVertices')

hltMetAnalyzerPF = _hltMetTester.clone(
    inputMETLabel = 'hltPFMETProducer', 
    METType = 'pf',
)
phase2_common.toModify(hltMetAnalyzerPF, inputMETLabel = 'hltPFMET')

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = 'dummy (phase2-only)',
    METType = 'pf',
)
phase2_common.toModify(hltMetAnalyzerPFPuppi, inputMETLabel = 'hltPFPuppiMET')

hltMetTypeOneAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = 'dummy (phase2-only)',
    METType = 'pf',
)
phase2_common.toModify(hltMetTypeOneAnalyzerPFPuppi, inputMETLabel = 'hltPFPuppiMETTypeOne')

hltMetAnalyzerPFCalo = _hltMetTester.clone(
    inputMETLabel = 'hltMet',
    METType = 'calo',
)
phase2_common.toModify(hltMetAnalyzerPFCalo, inputMETLabel = 'hltCaloMET')
