import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
OuterTrackerHarvesterV = DQMEDHarvester("OuterTrackerMCHarvester",)

OuterTracker_harvestingV = cms.Sequence(OuterTrackerHarvesterV)
