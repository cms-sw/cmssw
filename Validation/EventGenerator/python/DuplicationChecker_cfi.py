import FWCore.ParameterSet.Config as cms

duplicationChecker = cms.EDAnalyzer("DuplicationChecker",
    hepmcCollection = cms.InputTag("generatorSmeared"),
    searchForLHE = cms.bool(False),
    UseWeightFromHepMC = cms.bool(True)
)
