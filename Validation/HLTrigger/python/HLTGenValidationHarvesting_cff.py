import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
hltGenValidationClient = DQMEDHarvester("HLTGenValClient",
    outputFileName = cms.untracked.string(''),
    subDirs        = cms.untracked.vstring("HLTGenVal"),
)
