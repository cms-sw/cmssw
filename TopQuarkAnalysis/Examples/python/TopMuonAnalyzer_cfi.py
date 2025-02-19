import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of muons
#
analyzeMuon = cms.EDAnalyzer("TopMuonAnalyzer",
    input   = cms.InputTag("selectedPatMuons"),
    verbose = cms.bool(True)
)


