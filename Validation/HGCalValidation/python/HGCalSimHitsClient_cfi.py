import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDAnalyzer("HGCalSimHitsClient", 
                                     outputFile = cms.untracked.string(''),
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
