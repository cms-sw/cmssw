import FWCore.ParameterSet.Config as cms

wValidation = cms.EDAnalyzer("WValidation",
    hepmcCollection = cms.InputTag("generatorSmeared"),
    decaysTo = cms.int32(11),
    name = cms.string("Electrons"),
    UseWeightFromHepMC = cms.bool(True)
)
