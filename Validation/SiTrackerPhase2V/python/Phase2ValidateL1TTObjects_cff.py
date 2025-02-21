import FWCore.ParameterSet.Config as cms

from Validation.SiTrackerPhase2V.Phase2OTValidateReconstruction_cfi import * 
from Validation.SiTrackerPhase2V.Phase2OTValidateTTStub_cfi import *

trackingParticleValidOT = Phase2OTValidateReconstruction.clone()

stubValidOT = Phase2OTValidateTTStub.clone()
