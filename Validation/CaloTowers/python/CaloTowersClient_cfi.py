import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

calotowersClient = DQMEDHarvester("CaloTowersClient",
#     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
