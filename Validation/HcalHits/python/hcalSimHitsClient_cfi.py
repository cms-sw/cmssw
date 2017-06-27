import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hcalsimhitsClient = DQMEDHarvester("HcalSimHitsClient", 
     DQMDirName = cms.string("/"), # root directory
     Verbosity  = cms.untracked.bool(False),
)
