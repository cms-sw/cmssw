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
    genMetTrueLabel = 'genMetTrue',
    genMetCaloLabel = 'genMetCalo',
    genMhtLabel = 'genMht',
    inputMHTLabel = '',
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(_hltMetTester, primaryVertices = 'hltPhase2PixelVertices')

# `outFolder` keeps the DQM folders named after the validated collection (used by makeHLTMETValidationPlots.py).
# MET vs MHT plots are filled by the MHT analyzers (`inputMETLabel` defines the MET companion),
# in the `METvsMHT/<MET>_<MHT>` folder. Each MET-MHT pair must be configured in a single module.

####### MET #######
hltMetAnalyzerPF = _hltMetTester.clone(
    inputMETLabel = 'hltPFMETProducer',
    outFolder = 'hltPFMETProducer',
    METType = 'pf',
)
phase2_common.toModify(hltMetAnalyzerPF, inputMETLabel = 'hltPFMET', outFolder = 'hltPFMET')

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = 'dummy (phase2-only)',
    METType = 'pf',
)
phase2_common.toModify(hltMetAnalyzerPFPuppi, inputMETLabel = 'hltPFPuppiMET', outFolder = 'hltPFPuppiMET')

hltMetTypeOneAnalyzerPF = _hltMetTester.clone(
    inputMETLabel = 'hltPFMETTypeOne',
    outFolder = 'hltPFMETTypeOne',
    METType = 'pf',
)
phase2_common.toModify(hltMetTypeOneAnalyzerPF, inputMETLabel = 'hltPFPuppiMETTypeOne', outFolder = 'hltPFPuppiMETTypeOne')

hltMetAnalyzerPFCalo = _hltMetTester.clone(
    inputMETLabel = 'hltMet',
    outFolder = 'hltMet',
    METType = 'calo',
)
phase2_common.toModify(hltMetAnalyzerPFCalo, inputMETLabel = 'hltCaloMET', outFolder = 'hltCaloMET')

# Run 3 only
hltMetAnalyzerPFNoMu = _hltMetTester.clone(
    inputMETLabel = 'hltPFMETNoMuProducer',
    outFolder = 'hltPFMETNoMuProducer',
    METType = 'pf',
)

####### MHT #######
# Run 3 only
hltMhtAnalyzer = _hltMetTester.clone(
    inputMHTLabel = 'hltMht',
    outFolder = 'hltMht',
    inputMETLabel = 'hltMet',
    METType = 'mht',
)

# Run 3 only
hltMhtAnalyzerPFTightID = _hltMetTester.clone(
    inputMHTLabel = 'hltPFMHTTightID',
    outFolder = 'hltPFMHTTightID',
    inputMETLabel = 'hltPFMETProducer',
    METType = 'mht',
)

hltMhtAnalyzerPFPuppi = _hltMetTester.clone(
    inputMHTLabel = 'dummy (phase2-only)',
    inputMETLabel = 'dummy (phase2-only)',
    METType = 'mht',
)
phase2_common.toModify(hltMhtAnalyzerPFPuppi, inputMHTLabel = 'hltPFPuppiMHT', inputMETLabel = 'hltPFPuppiMET', outFolder = 'hltPFPuppiMHT')
