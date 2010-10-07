import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of tautrons
#
analyzeTau = cms.EDAnalyzer("TopTauAnalyzer",
    input   = cms.InputTag("selectedPatTaus"),
    verbose = cms.bool(True)
)


