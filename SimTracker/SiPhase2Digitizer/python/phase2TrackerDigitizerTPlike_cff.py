import FWCore.ParameterSet.Config as cms

import SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi

phase2TrackerDigitizerTPlike = SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi.phase2TrackerDigitizer.clone()

phase2TrackerDigitizerTPlike.isOTreadoutAnalog = cms.bool(True)
