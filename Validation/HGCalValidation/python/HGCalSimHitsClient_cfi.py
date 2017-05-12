import FWCore.ParameterSet.Config as cms

hgcalSimHitClientEE = cms.EDProducer("HGCalSimHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
