import FWCore.ParameterSet.Config as cms

ZDCDigiStudy = cms.EDAnalyzer("ZDCDigiStudy",
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('ZDCHITS')
)



