import FWCore.ParameterSet.Config as cms

basicHepMCHeavyIonValidation = cms.EDAnalyzer("BasicHepMCHeavyIonValidation",
    hepmcCollection = cms.InputTag("generator",""),
    UseWeightFromHepMC = cms.bool(True)
)
