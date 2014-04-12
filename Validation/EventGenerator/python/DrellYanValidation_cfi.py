import FWCore.ParameterSet.Config as cms

drellYanValidation = cms.EDAnalyzer("DrellYanValidation",
    hepmcCollection = cms.InputTag("generator",""),
    decaysTo = cms.int32(11),
    name = cms.string("Electrons"),
    UseWeightFromHepMC = cms.bool(True)
)
