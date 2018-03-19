import FWCore.ParameterSet.Config as cms

from Validation.OuterTrackerL1V.OuterTrackerMonitorTrackingParticles_cfi import *
OuterTrackerSourceV = cms.Sequence(
                                OuterTrackerMonitorTrackingParticles
                                 )
