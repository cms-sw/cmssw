import FWCore.ParameterSet.Config as cms

hgcalSimHitValidationEE = cms.EDAnalyzer('HGCalSimHitValidation',
                                         DetectorName = cms.string("HGCalEESensitive"),
                                         CaloHitSource = cms.string("HGCHitsEE"),
                                         TimeSlices    = cms.vdouble(25,1000),
                                         Verbosity     = cms.untracked.int32(0),
                                         TestNumber    = cms.untracked.bool(True)
                                         )
