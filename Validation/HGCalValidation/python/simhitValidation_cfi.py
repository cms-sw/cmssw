import FWCore.ParameterSet.Config as cms

hgcalSimHitValidationEE = cms.EDAnalyzer('HGCalSimHitValidation',
                                         DetectorName = cms.string("HGCalEESensitive"),
                                         CaloHitSource = cms.string("HGCHitsEE"),
                                         HERebuild     = cms.untracked.bool(False),
                                         TestNumber    = cms.untracked.bool(True),
                                         Verbosity     = cms.untracked.int32(0)
                                         )
