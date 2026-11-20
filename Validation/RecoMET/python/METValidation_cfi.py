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
