import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDAnalyzer("HGCalSimHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     TimeSlices   = cms.int32(2),
                                     Verbosity    = cms.untracked.int32(0),
)
