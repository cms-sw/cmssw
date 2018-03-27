import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
SiOuterTrackerHarvesterV = DQMEDHarvester("OuterTrackerMCHarvester",)

OuterTracker_harvestingV = cms.Sequence(
    SiOuterTrackerHarvesterV
                                 )
