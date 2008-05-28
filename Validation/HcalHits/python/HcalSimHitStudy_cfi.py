import FWCore.ParameterSet.Config as cms

hcalSimHitStudy = cms.EDFilter("HcalSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    OutputFile = cms.untracked.string('valid_HB.root'),
    Verbose = cms.untracked.bool(True),
    HitCollection = cms.untracked.string('HcalHits')
)



