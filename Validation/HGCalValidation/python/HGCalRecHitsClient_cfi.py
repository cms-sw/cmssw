import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hgcalRecHitClientEE = DQMEDHarvester("HGCalRecHitsClient", 
                                     DetectorName = cms.string("HGCalEESensitive"),
                                     Verbosity     = cms.untracked.int32(0)
                                     )
# foo bar baz
# UFHN6Qknr70zN
# RL9qnlBuXTsLO
