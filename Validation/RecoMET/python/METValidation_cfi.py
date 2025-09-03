import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
metAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "caloMet",
    METType = "calo",
    primaryVertices = "offlinePrimaryVertices"
)

pfMetAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "pfMet",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
) 

genMetTrueAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "genMetTrue",
    METType = "gen",
    primaryVertices = "offlinePrimaryVertices"
)

#genMetCaloAnalyzer = DQMEDAnalyzer(
#    "METTester",
#    OutputFile = '',
#    inputMETLabel = "genMetCalo"
#    )
#
#genMptCaloAnalyzer = DQMEDAnalyzer(
#    "METTester",
#    OutputFile = '',
#    inputMETLabel = "genMptCalo"
#    )
#
#
#genMetCaloAndNonPromptAnalyzer = DQMEDAnalyzer(
#    "METTester",
#    OutputFile = '',
#    inputMETLabel = "genMetCaloAndNonPrompt"
#    )

pfType0CorrectedMetAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "pfMetT0pc",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)
pfType1CorrectedMetAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "PfMetT1",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)
pfType01CorrectedMetAnalyzer = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "PfMetT0pcT1",
    METType = "pf",
    primaryVertices = "offlinePrimaryVertices"
)
pfType1CorrectedMetAnalyzerMiniAOD = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "slimmedMETs",
    METType = "miniaod",
    primaryVertices = "offlineSlimmedPrimaryVertices"
)

pfPuppiMetAnalyzerMiniAOD = DQMEDAnalyzer(
    "METTester",
    inputMETLabel = "slimmedMETsPuppi",
    METType = "miniaod",
    primaryVertices = "offlineSlimmedPrimaryVertices"
)
