import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hgcalSimHitClientEE = DQMEDHarvester("HGCalSimHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     TimeSlices   = cms.int32(2),
                                     Verbosity    = cms.untracked.int32(0),
)
