import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *
from Validation.Phase2OuterTracker.OuterTrackerCluster_cfi import *
from Validation.Phase2OuterTracker.OuterTrackerMCTruth_cfi import *
from Validation.Phase2OuterTracker.OuterTrackerStub_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerCluster * 
				  OuterTrackerMCTruth *
				  OuterTrackerStub)
