import FWCore.ParameterSet.Config as cms

hgcalRecHitClientEE = cms.EDProducer("HGCalRecHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
