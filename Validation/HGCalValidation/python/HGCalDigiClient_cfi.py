import FWCore.ParameterSet.Config as cms

hgcalDigiClientEE = cms.EDAnalyzer("HGCalDigiClient", 
                                   outputFile   = cms.untracked.string(''),
                                   DetectorName = cms.string("HGCalEESensitive"),
                                   Verbosity    = cms.untracked.int32(0)
                                   )
