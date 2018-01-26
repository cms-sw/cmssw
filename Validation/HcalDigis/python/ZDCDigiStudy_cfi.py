import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ZDCDigiStudy = DQMEDAnalyzer('ZDCDigiStudy',
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('ZDCHITS')
)



