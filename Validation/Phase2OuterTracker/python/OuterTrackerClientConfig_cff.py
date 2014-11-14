import FWCore.ParameterSet.Config as cms

from Validation.Phase2OuterTracker.OuterTrackerClusterClient_cfi import *
from Validation.Phase2OuterTracker.OuterTrackerMCTruthClient_cfi import *
from Validation.Phase2OuterTracker.OuterTrackerStubClient_cfi import *

OuterTrackerClient = cms.Sequence(OuterTrackerClusterClient *
				  OuterTrackerMCTruthClient *
				  OuterTrackerStubClient)

