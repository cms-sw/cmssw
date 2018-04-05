import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
wValidation = DQMEDAnalyzer('WValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    decaysTo = cms.int32(11),
    name = cms.string("Electrons"),
    UseWeightFromHepMC = cms.bool(True)
)
