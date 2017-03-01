import FWCore.ParameterSet.Config as cms

basicHepMCValidation = cms.EDAnalyzer("BasicHepMCValidation",
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
