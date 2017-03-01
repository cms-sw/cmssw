import FWCore.ParameterSet.Config as cms

ftSimHitTest = cms.EDAnalyzer('FTSimHitTest',
                              ModuleLabel   = cms.untracked.string('g4SimHits'),
                              HitBarrelLabel= cms.untracked.string('FastTimerHitsBarrel'),
                              HitEndcapLabel= cms.untracked.string('FastTimerHitsEndcap'),
                              )
