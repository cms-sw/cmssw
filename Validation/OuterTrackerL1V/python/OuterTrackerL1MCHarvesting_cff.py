import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
OuterTrackerL1HarvesterV = DQMEDHarvester("OuterTrackerMCHarvester",)

OuterTracker_harvestingV = cms.Sequence(
    OuterTrackerL1HarvesterV
                                 )
