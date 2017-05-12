import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDProducer("HGCalSimHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     TimeSlices   = cms.int32(2),
                                     Verbosity    = cms.untracked.int32(0),
)
