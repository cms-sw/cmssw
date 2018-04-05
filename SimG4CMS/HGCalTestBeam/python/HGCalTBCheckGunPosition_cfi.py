import FWCore.ParameterSet.Config as cms

HGCalTBCheckGunPostion = cms.EDFilter("HGCalTBCheckGunPostion",
                                      HepMCProductLabel = cms.InputTag('generatorSmeared'),
                                      Verbosity         = cms.untracked.bool(False),
                                      Method2           = cms.untracked.bool(False),
)
