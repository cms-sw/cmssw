import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of muons
#
analyzeMuon = cms.EDAnalyzer("TopMuonAnalyzer",
    inputMuon = cms.InputTag("selectedLayer1Muons"),
    inputElec = cms.InputTag("selectedLayer1Electrons")
)


