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
