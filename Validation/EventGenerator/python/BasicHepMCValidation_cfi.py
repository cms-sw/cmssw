import FWCore.ParameterSet.Config as cms

basicHepMCValidation = cms.EDAnalyzer("BasicHepMCValidation",
    hepmcCollection = cms.InputTag("generator",""),
    UseWeightFromHepMC = cms.bool(True)
)
