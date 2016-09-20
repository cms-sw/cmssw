import FWCore.ParameterSet.Config as cms

hgcalBHValidation = cms.EDAnalyzer('HGCalBHValidation',
                                   ModuleLabel   = cms.untracked.string('g4SimHits'),
                                   HitCollection = cms.untracked.string('HcalHits'),
                                   DigiCollection= cms.untracked.InputTag('simHcalUnsuppressedDigis','HBHEUpgradeDigiCollection'),
                                   Sample        = cms.untracked.int32(5),
                                   Threshold     = cms.untracked.double(15.0)
                                   )
