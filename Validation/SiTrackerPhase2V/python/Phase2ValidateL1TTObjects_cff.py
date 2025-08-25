import FWCore.ParameterSet.Config as cms

from Validation.SiTrackerPhase2V.Phase2OTValidateTracks_cfi import * 
from Validation.SiTrackerPhase2V.Phase2OTValidateStub_cfi import *

trackingParticleValidOT = Phase2OTValidateTracks.clone()
stubValidOT = Phase2OTValidateStub.clone()
