import FWCore.ParameterSet.Config as cms

hcalHitValid = cms.EDFilter("HcalHitValidation",
      ModuleLabel   = cms.untracked.string('g4SimHits'),
      HitCollection = cms.untracked.string('HcalHits'),
      LayerInfo     = cms.untracked.string('HcalInfoLayer'),
      NxNInfo       = cms.untracked.string('HcalInfoNxN'),
      JetsInfo      = cms.untracked.string('HcalInfoJets'),
      OutputFile    = cms.untracked.string('valid_HB.root'),
      Verbose       = cms.untracked.bool(True),
      TestNumbering = cms.untracked.bool(True),
      ValidHits     = cms.untracked.bool(True),
      ValidLayer    = cms.untracked.bool(True),
      ValidNxN      = cms.untracked.bool(True),
      ValidJets     = cms.untracked.bool(True)
)
