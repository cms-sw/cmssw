import FWCore.ParameterSet.Config as cms

hgcalRecHitClientEE = cms.EDAnalyzer("HGCalRecHitsClient", 
                                     outputFile = cms.untracked.string(''),
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
