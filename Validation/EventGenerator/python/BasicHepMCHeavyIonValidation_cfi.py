import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
basicHepMCHeavyIonValidation = DQMEDAnalyzer('BasicHepMCHeavyIonValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    UseWeightFromHepMC = cms.bool(True)
)
# foo bar baz
# 69gqUTJEXqF7w
# 5Pg0b9Ma0o8sK
