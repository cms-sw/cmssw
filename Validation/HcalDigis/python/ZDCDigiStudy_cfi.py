import FWCore.ParameterSet.Config as cms

ZDCDigiStudy = DQMStep1Module('ZDCDigiStudy',
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('ZDCHITS')
)



