import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hcaldigisClient = DQMEDHarvester("HcalDigisClient",
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
