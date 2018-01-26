import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
duplicationChecker = DQMEDAnalyzer('DuplicationChecker',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    searchForLHE = cms.bool(False),
    UseWeightFromHepMC = cms.bool(True)
)
