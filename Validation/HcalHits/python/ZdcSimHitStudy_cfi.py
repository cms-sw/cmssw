import FWCore.ParameterSet.Config as cms

zdcSimHitStudy = cms.EDAnalyzer("ZdcSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('ZDCHITS')
)



