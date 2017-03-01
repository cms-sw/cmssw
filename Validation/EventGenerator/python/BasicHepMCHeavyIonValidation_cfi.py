import FWCore.ParameterSet.Config as cms

basicHepMCHeavyIonValidation = cms.EDAnalyzer("BasicHepMCHeavyIonValidation",
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
