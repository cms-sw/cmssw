import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
basicHepMCValidation = DQMEDAnalyzer('BasicHepMCValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
