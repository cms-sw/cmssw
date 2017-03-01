import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDAnalyzer("HGCalSimHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
