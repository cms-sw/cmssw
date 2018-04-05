import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
drellYanValidation = DQMEDAnalyzer('DrellYanValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    decaysTo = cms.int32(11),
    name = cms.string("Electrons"),
    UseWeightFromHepMC = cms.bool(True)
)
