import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
basicHepMCHeavyIonValidation = DQMEDAnalyzer('BasicHepMCHeavyIonValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
