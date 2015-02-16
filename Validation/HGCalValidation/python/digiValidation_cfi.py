import FWCore.ParameterSet.Config as cms

hgcalDigiValidationEE = cms.EDAnalyzer('HGCalDigiValidation',
                                       DetectorName  = cms.string("HGCalEESensitive"),
                                       DigiSource    = cms.InputTag("mix", "HGCDigisEE"),
                                       Verbosity     = cms.untracked.int32(0),
                                       SampleIndx    = cms.untracked.int32(0)
                                       )
