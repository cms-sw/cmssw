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

# `outFolder` keeps the DQM folders named after the validated collection (used by makeMETValidationPlots.py).
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

####### MHT scan #######
from Validation.RecoMET.mhtScanUtils import cutTag as _cutTag, moduleTag as _moduleTag, scanCuts as _scanCuts

def addMHTScanHLT(process, minPtJetMht=(20., 30., 40., 50.), maxEtaJetMht=None, genWDecayPdgIds=(), lightMode=True):
    """
    Validates `hltPFPuppiMHT` (phase-2) for several jet kinematic cuts.
    `minPtJetMht` and `maxEtaJetMht` are lists with the same length: the i-th MHT producer uses
    the i-th value of each list. If `maxEtaJetMht` is not given, it is fixed at 5 for all `minPtJetMht` values.
    The distributions are stored in `<runDir>/hltPFPuppiMHT/<cut tag>`.
    If `genWDecayPdgIds` is not empty (e.g. (11, 13, 15) for W' -> l nu), the same distributions are
    additionally stored per |pdgId| in `<runDir>/hltPFPuppiMHT/<cut tag>/pdgid<N>`; they require `genParticles`.
    With `lightMode`, only MEx, MEy and METPhi are filled (see METTester); otherwise full METTester
    instances (as hltMhtAnalyzerPFPuppi) are used.
    """
    minPtJetMht, maxEtaJetMht = _scanCuts('addMHTScanHLT', minPtJetMht, maxEtaJetMht)
    if not hasattr(process, 'hltPFPuppiMHT'):
        raise RuntimeError("addMHTScanHLT: process has no hltPFPuppiMHT (phase-2 HLT menu required).")

    modules = []
    for minPt, maxEta in zip(minPtJetMht, maxEtaJetMht):
        tag = _cutTag(minPt, maxEta)
        ltag = _moduleTag(tag)
        # producer labels must not match hltTriggerSummaryAOD's 'hlt*' pattern, which would create a scheduling cycle
        label = 'scanPFPuppiMHT' + ltag
        setattr(process, label, process.hltPFPuppiMHT.clone(minPtJetMht=minPt, maxEtaJetMht=maxEta))
        modules.append(getattr(process, label))

        selections = [(None, 'hltPFPuppiMHT/' + tag)]
        selections += [(pdg, f'hltPFPuppiMHT/{tag}/pdgid{pdg}') for pdg in genWDecayPdgIds]
        for pdg, folder in selections:
            analyzer = _hltMetTester.clone(
                inputMHTLabel = label,
                inputMETLabel = '' if lightMode else 'hltPFPuppiMET',
                outFolder = folder,
                METType = 'mht',
                lightMode = lightMode,
                genWDecayPdgIds = cms.vint32(pdg) if pdg is not None else cms.vint32(),
            )
            aname = 'hltMhtAnalyzer' + ltag + ('' if pdg is None else f'Pdgid{pdg}')
            setattr(process, aname, analyzer)
            modules.append(analyzer)

    scan = cms.Sequence(sum(modules[1:], modules[0]))
    process.hltMhtScanSeq = scan
    process.HLTJetMETValSeq += scan
