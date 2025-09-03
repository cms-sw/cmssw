import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltMETPostProcessor = DQMEDHarvester(
    'METTesterPostProcessor',
    isHLT = cms.untracked.bool(True)
)
