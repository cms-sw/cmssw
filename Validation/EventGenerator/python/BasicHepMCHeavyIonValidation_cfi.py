import FWCore.ParameterSet.Config as cms

basicHepMCHeavyIonValidation = DQMStep1Module('BasicHepMCHeavyIonValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
