import FWCore.ParameterSet.Config as cms

hgcalDigiClientEE = cms.EDAnalyzer("HGCalDigiClient", 
                                   DetectorName = cms.string("HGCalEESensitive"),
                                   Verbosity    = cms.untracked.int32(0)
                                   )
