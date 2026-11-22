import FWCore.ParameterSet.Config as cms

# Fill validation histograms for MET
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
metAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = cms.string("caloMet"),
    outFolder = cms.untracked.string("caloMet"),
    METType = cms.untracked.string("calo"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices")
)

pfMetAnalyzer = metAnalyzer.clone(
    inputMETLabel = "pfMet",
    outFolder = "pfMet",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
) 

# Phase-2: PUPPI MET (scheduled in METRelValForDQM_cff.py)
pfMetPuppiAnalyzer = metAnalyzer.clone(
    inputMETLabel = "pfMetPuppi",
    outFolder = "pfMetPuppi",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)

# Phase-2: PUPPI MHT from the (uncorrected) AK4 PUPPI jets, with the same selection as hltPFPuppiMHT.
# The MET companion defines the MET vs MHT plots, in the METvsMHT/<MET>_<MHT> folder.
pfPuppiMHT = cms.EDProducer("HLTHtMhtProducer",
    excludePFMuons = cms.bool(False),
    jetsLabel = cms.InputTag("ak4PFJetsPuppi"),
    maxEtaJetHt = cms.double(5.0),
    maxEtaJetMht = cms.double(5.0),
    minNJetHt = cms.int32(0),
    minNJetMht = cms.int32(0),
    minPtJetHt = cms.double(30.0),
    minPtJetMht = cms.double(30.0),
    pfCandidatesLabel = cms.InputTag(""),
    usePt = cms.bool(False)
)

pfPuppiMHTAnalyzer = metAnalyzer.clone(
    inputMHTLabel = cms.string("pfPuppiMHT"),
    inputMETLabel = "pfMetPuppi",
    outFolder = "pfPuppiMHT",
    METType = "mht",
    primaryVertices = "offlinePrimaryVertices"
)

genMetTrueAnalyzer = metAnalyzer.clone(
    inputMETLabel = "genMetTrue",
    outFolder = "genMetTrue",
    METType = "gen",
    primaryVertices = "offlinePrimaryVertices"
)

pfType0CorrectedMetAnalyzer = metAnalyzer.clone(
    inputMETLabel = "pfMetT0pc",
    outFolder = "pfMetT0pc",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)
pfType1CorrectedMetAnalyzer = metAnalyzer.clone(
    inputMETLabel = "PfMetT1",
    outFolder = "PfMetT1",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)

pfType01CorrectedMetAnalyzer = metAnalyzer.clone(
    inputMETLabel = "PfMetT0pcT1",
    outFolder = "PfMetT0pcT1",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)

pfType1CorrectedMetAnalyzerMiniAOD = metAnalyzer.clone(
    inputMETLabel = "slimmedMETs",
    outFolder = "slimmedMETs",
    METType = "miniaod",
    primaryVertices = "offlineSlimmedPrimaryVertices"
)

pfPuppiMetAnalyzerMiniAOD = metAnalyzer.clone(
    inputMETLabel = "slimmedMETsPuppi",
    outFolder = "slimmedMETsPuppi",
    METType = "miniaod",
    primaryVertices = "offlineSlimmedPrimaryVertices"
)

####### MHT scan #######
from Validation.RecoMET.mhtScanUtils import cutTag as _cutTag, moduleTag as _moduleTag, scanCuts as _scanCuts

def addMHTScanOffline(process, minPtJetMht=(20., 30., 40., 50.), maxEtaJetMht=None, genWDecayPdgIds=(),
                      lightMode=True, scanOnly=False):
    """
    Validates `pfPuppiMHT` (phase-2, uncorrected `ak4PFJetsPuppi` jets) for several jet kinematic cuts: the
    offline counterpart of `addMHTScanHLT`. `minPtJetMht` and `maxEtaJetMht` are lists with the same length
    (`maxEtaJetMht` defaults to 5), the i-th MHT producer using the i-th value of each list.
    The distributions are stored in `<runDir>/pfPuppiMHT/<cut tag>`, and additionally in
    `<runDir>/pfPuppiMHT/<cut tag>/pdgid<N>` for each |pdgId| of `genWDecayPdgIds` (e.g. (11, 13, 15) for
    W' -> l nu; requires `genParticles`). With `lightMode`, only MEx, MEy and METPhi are filled (see METTester).
    The modules are appended to the validation sequence `METValidation`. With `scanOnly`, the
    `validation_step` path is instead replaced by the scan alone, to run it on already reconstructed events.
    """
    minPtJetMht, maxEtaJetMht = _scanCuts('addMHTScanOffline', minPtJetMht, maxEtaJetMht)
    if not hasattr(process, 'pfPuppiMHT'):
        raise RuntimeError("addMHTScanOffline: process has no pfPuppiMHT (phase-2 offline validation required).")

    modules = []
    for minPt, maxEta in zip(minPtJetMht, maxEtaJetMht):
        tag = _cutTag(minPt, maxEta)
        ltag = _moduleTag(tag)
        label = 'scanPfPuppiMHT' + ltag
        setattr(process, label, process.pfPuppiMHT.clone(minPtJetMht=minPt, maxEtaJetMht=maxEta))
        modules.append(getattr(process, label))

        selections = [(None, 'pfPuppiMHT/' + tag)]
        selections += [(pdg, f'pfPuppiMHT/{tag}/pdgid{pdg}') for pdg in genWDecayPdgIds]
        for pdg, folder in selections:
            analyzer = process.pfPuppiMHTAnalyzer.clone(
                inputMHTLabel = label,
                inputMETLabel = '' if lightMode else 'pfMetPuppi',
                outFolder = folder,
                METType = 'mht',
                lightMode = cms.bool(lightMode),
                genWDecayPdgIds = cms.vint32(pdg) if pdg is not None else cms.vint32(),
            )
            aname = 'pfMhtAnalyzer' + ltag + ('' if pdg is None else f'Pdgid{pdg}')
            setattr(process, aname, analyzer)
            modules.append(analyzer)

    scan = cms.Sequence(sum(modules[1:], modules[0]))
    process.pfMhtScanSeq = scan
    if scanOnly:
        process.validation_step = cms.EndPath(scan)
    else:
        process.METValidation += scan
