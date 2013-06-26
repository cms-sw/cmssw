import FWCore.ParameterSet.Config as cms

#
# module to make simple analyses of electrons
#
analyzeElec = cms.EDAnalyzer("TopElecAnalyzer",
    input   = cms.InputTag("selectedPatElectrons"),
    verbose = cms.bool(True)
)


