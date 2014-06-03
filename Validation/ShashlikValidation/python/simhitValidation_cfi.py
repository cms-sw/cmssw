import FWCore.ParameterSet.Config as cms

shashlikSimHitValidation = cms.EDAnalyzer('ShashlikSimHitValidation',
                                          CaloHitSource = cms.string("EcalHitsEK"),
                                          Verbosity     = cms.untracked.int32(1)
)
