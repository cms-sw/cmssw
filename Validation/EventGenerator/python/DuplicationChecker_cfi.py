import FWCore.ParameterSet.Config as cms

duplicationChecker = cms.EDAnalyzer("DuplicationChecker",
    hepmcCollection = cms.InputTag("generator",""),
    searchForLHE = cms.bool(False),
    UseWeightFromHepMC = cms.bool(True)
)
