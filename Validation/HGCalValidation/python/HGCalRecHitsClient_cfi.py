import FWCore.ParameterSet.Config as cms

hgcalRecHitClientEE = cms.EDAnalyzer("HGCalRecHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
