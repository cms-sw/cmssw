import FWCore.ParameterSet.Config as cms

from Validation.SiTrackerPhase2V.Phase2OTValidateTrackingParticles_cfi import * 
from Validation.SiTrackerPhase2V.Phase2OTValidateTTStub_cfi import *

trackingParticleValidOT = Phase2OTValidateTrackingParticles.clone()

stubValidOT = Phase2OTValidateTTStub.clone()
