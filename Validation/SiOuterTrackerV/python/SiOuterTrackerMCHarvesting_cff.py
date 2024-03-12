import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
OuterTrackerHarvesterV = DQMEDHarvester("OuterTrackerMCHarvester",)

OuterTracker_harvestingV = cms.Sequence(OuterTrackerHarvesterV)
# foo bar baz
# uFn8RQrq2aUqs
