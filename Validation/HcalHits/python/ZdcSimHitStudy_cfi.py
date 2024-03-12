import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
zdcSimHitStudy = DQMEDAnalyzer('ZdcSimHitStudy',
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('ZDCHITS')
)



# foo bar baz
# 8mP95TUfk6gy6
