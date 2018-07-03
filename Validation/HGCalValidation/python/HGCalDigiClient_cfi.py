import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hgcalDigiClientEE = DQMEDHarvester("HGCalDigiClient", 
                                   DetectorName = cms.string("HGCalEESensitive"),
                                   Verbosity    = cms.untracked.int32(0)
                                   )
