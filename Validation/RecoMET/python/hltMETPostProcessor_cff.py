import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hltMETPostProcessor = DQMEDHarvester(
    'METTesterPostProcessor',
    runDir = cms.untracked.string("HLT/JetMET/METValidation/")
)
