import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hgcalHitClient = DQMEDHarvester("HGCalHitClient", 
                                DirectoryName = cms.string("HitValidation"),
                                )
# foo bar baz
# vzOheRm4Xp6jM
