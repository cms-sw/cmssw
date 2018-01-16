import FWCore.ParameterSet.Config as cms

basicHepMCValidation = DQMStep1Module('BasicHepMCValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
